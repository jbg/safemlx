//! Compilation of functions.

// TODO: there's plenty boilerplate code here but it's not clear how to reduce it

use std::marker::PhantomData;

use crate::{error::Exception, Array, Stream};

use super::{
    transform_guard, type_id_to_usize, Closure, Compiled, CompiledState, Guarded, VectorArray,
};

/// Options used when compiling reusable functions.
#[derive(Debug, Clone, Default)]
pub struct CompileOptions {
    shapeless: bool,
    constants: Vec<u64>,
}

impl CompileOptions {
    /// Create compile options with shape-specialized caching.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure whether cache entries ignore input shapes.
    pub fn shapeless(mut self, shapeless: bool) -> Self {
        self.shapeless = shapeless;
        self
    }

    /// Configure cache-key constants for non-array specialization.
    pub fn constants(mut self, constants: impl Into<Vec<u64>>) -> Self {
        self.constants = constants.into();
        self
    }

    /// Add one cache-key constant.
    pub fn constant(mut self, constant: u64) -> Self {
        self.constants.push(constant);
        self
    }
}

impl From<bool> for CompileOptions {
    fn from(shapeless: bool) -> Self {
        Self::new().shapeless(shapeless)
    }
}

impl From<Option<bool>> for CompileOptions {
    fn from(shapeless: Option<bool>) -> Self {
        Self::new().shapeless(shapeless.unwrap_or(false))
    }
}

/// Returns a compiled function that produces the same output as `f`.
///
/// Please refer to the [swift binding
/// documentation](https://swiftpackageindex.com/ml-explore/mlx-swift/main/documentation/mlx/compilation)
/// for more information.
pub fn compile<F, A, O, E>(
    f: F,
    shapeless: impl Into<Option<bool>>,
) -> impl for<'a> FnMut(F::Args<'a>) -> Result<O, Exception> + 'static
where
    F: Compile<A, O, E> + 'static + Copy,
{
    let shapeless = shapeless.into().unwrap_or(false);
    move |args| {
        // NOTE: we have to place this here to avoid the lifetime issue
        // `f.compile` will look up the cached compiled function so it shouldn't result in re-compilation
        let mut compiled = f.compile(shapeless);
        compiled.call_mut(args)
    }
}

/// Returns a reusable compiled unary function whose body receives the caller's
/// explicit stream.
///
/// MLX compile only treats arrays as graph inputs, but many safemlx functions
/// require an explicit [`Stream`]. This wrapper keeps the stream explicit at the
/// public call site while adapting the single array argument into MLX's compiled
/// input vector.
pub fn compile_unary_with_stream<F>(
    f: F,
    shapeless: impl Into<Option<bool>>,
) -> CompiledUnaryWithStream<F>
where
    F: FnMut(&Array, &Stream) -> Result<Array, Exception> + 'static,
{
    CompiledUnaryWithStream::new(f, shapeless)
}

/// Returns a reusable compiled function with hidden array captures.
///
/// Captures are appended to the compiled graph's array inputs, so they are
/// traced safely without being closed over as implicit constants. This is useful
/// for model weights or other arrays that should participate in the graph but
/// should not be part of the public call signature.
pub fn compile_with_captures<F>(
    f: F,
    captures: impl IntoIterator<Item = Array>,
    options: impl Into<CompileOptions>,
) -> CompiledWithCaptures<F>
where
    F: FnMut(&[Array], &[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    CompiledWithCaptures::new(f, captures, options)
}

/// Returns a reusable compiled function with hidden array captures and an
/// explicit stream at the call site.
pub fn compile_with_stream_and_captures<F>(
    f: F,
    captures: impl IntoIterator<Item = Array>,
    options: impl Into<CompileOptions>,
) -> CompiledWithStreamAndCaptures<F>
where
    F: FnMut(&[Array], &[Array], &Stream) -> Result<Vec<Array>, Exception> + 'static,
{
    CompiledWithStreamAndCaptures::new(f, captures, options)
}

/// Returns a reusable compiled unary function with hidden array captures and an
/// explicit stream at the call site.
pub fn compile_unary_with_stream_and_captures<F>(
    f: F,
    captures: impl IntoIterator<Item = Array>,
    options: impl Into<CompileOptions>,
) -> CompiledUnaryWithStreamAndCaptures<F>
where
    F: FnMut(&Array, &[Array], &Stream) -> Result<Array, Exception> + 'static,
{
    CompiledUnaryWithStreamAndCaptures::new(f, captures, options)
}

/// Returns a reusable compiled binary function with hidden array captures and
/// an explicit stream at the call site.
pub fn compile_binary_with_stream_and_captures<F>(
    f: F,
    captures: impl IntoIterator<Item = Array>,
    options: impl Into<CompileOptions>,
) -> CompiledBinaryWithStreamAndCaptures<F>
where
    F: FnMut((&Array, &Array), &[Array], &Stream) -> Result<Array, Exception> + 'static,
{
    CompiledBinaryWithStreamAndCaptures::new(f, captures, options)
}

/// A reusable compiled function with hidden array captures.
pub struct CompiledWithCaptures<F> {
    f: F,
    captures: Vec<Array>,
    options: CompileOptions,
    id: usize,
}

impl<F> CompiledWithCaptures<F>
where
    F: 'static,
{
    fn new(
        f: F,
        captures: impl IntoIterator<Item = Array>,
        options: impl Into<CompileOptions>,
    ) -> Self {
        let id = type_id_to_usize(&f);
        Self {
            f,
            captures: captures.into_iter().collect(),
            options: options.into(),
            id,
        }
    }

    /// Replace the hidden array captures used on subsequent calls.
    pub fn set_captures(&mut self, captures: impl IntoIterator<Item = Array>) {
        self.captures = captures.into_iter().collect();
    }

    /// Return the currently configured hidden array captures.
    pub fn captures(&self) -> &[Array] {
        &self.captures
    }
}

impl<F> CompiledWithCaptures<F>
where
    F: FnMut(&[Array], &[Array]) -> Result<Vec<Array>, Exception>,
{
    /// Call the compiled function with public array inputs.
    pub fn call(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        let args_len = args.len();
        let captures_len = self.captures.len();
        let f = &mut self.f;
        let mut inner = move |inputs: &[Array]| -> Result<Vec<Array>, Exception> {
            let (args, captures) = inputs.split_at(args_len);
            debug_assert_eq!(captures.len(), captures_len);
            f(args, captures)
        };
        let inner_closure = Closure::new_fallible(&mut inner);
        let inputs = combined_inputs(args, &self.captures);
        call_mut_inner(
            inner_closure,
            self.id,
            self.options.shapeless,
            &self.options.constants,
            &inputs,
        )
    }
}

impl<F> Clone for CompiledWithCaptures<F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            captures: self.captures.clone(),
            options: self.options.clone(),
            id: self.id,
        }
    }
}

impl<F> std::fmt::Debug for CompiledWithCaptures<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledWithCaptures")
            .field("captures", &self.captures.len())
            .field("options", &self.options)
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<F> Drop for CompiledWithCaptures<F> {
    fn drop(&mut self) {
        erase_compiled(self.id);
    }
}

/// A reusable compiled function with hidden array captures and an explicit
/// stream at the call site.
pub struct CompiledWithStreamAndCaptures<F> {
    f: F,
    captures: Vec<Array>,
    options: CompileOptions,
    id: usize,
}

impl<F> CompiledWithStreamAndCaptures<F>
where
    F: 'static,
{
    fn new(
        f: F,
        captures: impl IntoIterator<Item = Array>,
        options: impl Into<CompileOptions>,
    ) -> Self {
        let id = type_id_to_usize(&f);
        Self {
            f,
            captures: captures.into_iter().collect(),
            options: options.into(),
            id,
        }
    }

    /// Replace the hidden array captures used on subsequent calls.
    pub fn set_captures(&mut self, captures: impl IntoIterator<Item = Array>) {
        self.captures = captures.into_iter().collect();
    }

    /// Return the currently configured hidden array captures.
    pub fn captures(&self) -> &[Array] {
        &self.captures
    }
}

impl<F> CompiledWithStreamAndCaptures<F>
where
    F: FnMut(&[Array], &[Array], &Stream) -> Result<Vec<Array>, Exception>,
{
    /// Call the compiled function with public array inputs and an explicit
    /// stream.
    pub fn call(&mut self, args: &[Array], stream: &Stream) -> Result<Vec<Array>, Exception> {
        let args_len = args.len();
        let captures_len = self.captures.len();
        let f = &mut self.f;
        let mut inner = move |inputs: &[Array]| -> Result<Vec<Array>, Exception> {
            let (args, captures) = inputs.split_at(args_len);
            debug_assert_eq!(captures.len(), captures_len);
            f(args, captures, stream)
        };
        let inner_closure = Closure::new_fallible(&mut inner);
        let inputs = combined_inputs(args, &self.captures);
        call_mut_inner(
            inner_closure,
            self.id,
            self.options.shapeless,
            &self.options.constants,
            &inputs,
        )
    }
}

impl<F> Clone for CompiledWithStreamAndCaptures<F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            captures: self.captures.clone(),
            options: self.options.clone(),
            id: self.id,
        }
    }
}

impl<F> std::fmt::Debug for CompiledWithStreamAndCaptures<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledWithStreamAndCaptures")
            .field("captures", &self.captures.len())
            .field("options", &self.options)
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<F> Drop for CompiledWithStreamAndCaptures<F> {
    fn drop(&mut self) {
        erase_compiled(self.id);
    }
}

/// A reusable compiled unary function with hidden array captures and an
/// explicit stream at the call site.
pub struct CompiledUnaryWithStreamAndCaptures<F> {
    f: F,
    captures: Vec<Array>,
    options: CompileOptions,
    id: usize,
}

impl<F> CompiledUnaryWithStreamAndCaptures<F>
where
    F: 'static,
{
    fn new(
        f: F,
        captures: impl IntoIterator<Item = Array>,
        options: impl Into<CompileOptions>,
    ) -> Self {
        let id = type_id_to_usize(&f);
        Self {
            f,
            captures: captures.into_iter().collect(),
            options: options.into(),
            id,
        }
    }

    /// Replace the hidden array captures used on subsequent calls.
    pub fn set_captures(&mut self, captures: impl IntoIterator<Item = Array>) {
        self.captures = captures.into_iter().collect();
    }

    /// Return the currently configured hidden array captures.
    pub fn captures(&self) -> &[Array] {
        &self.captures
    }
}

impl<F> CompiledUnaryWithStreamAndCaptures<F>
where
    F: FnMut(&Array, &[Array], &Stream) -> Result<Array, Exception>,
{
    /// Call the compiled function with one public array input and an explicit
    /// stream.
    pub fn call(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let captures_len = self.captures.len();
        let f = &mut self.f;
        let mut inner = move |inputs: &[Array]| -> Result<Vec<Array>, Exception> {
            let (args, captures) = inputs.split_at(1);
            debug_assert_eq!(captures.len(), captures_len);
            Ok(vec![f(&args[0], captures, stream)?])
        };
        let inner_closure = Closure::new_fallible(&mut inner);
        let inputs = combined_inputs(std::slice::from_ref(x), &self.captures);
        let result = call_mut_inner(
            inner_closure,
            self.id,
            self.options.shapeless,
            &self.options.constants,
            &inputs,
        )?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<F> Clone for CompiledUnaryWithStreamAndCaptures<F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            captures: self.captures.clone(),
            options: self.options.clone(),
            id: self.id,
        }
    }
}

impl<F> std::fmt::Debug for CompiledUnaryWithStreamAndCaptures<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledUnaryWithStreamAndCaptures")
            .field("captures", &self.captures.len())
            .field("options", &self.options)
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<F> Drop for CompiledUnaryWithStreamAndCaptures<F> {
    fn drop(&mut self) {
        erase_compiled(self.id);
    }
}

/// A reusable compiled binary function with hidden array captures and an
/// explicit stream at the call site.
pub struct CompiledBinaryWithStreamAndCaptures<F> {
    f: F,
    captures: Vec<Array>,
    options: CompileOptions,
    id: usize,
}

impl<F> CompiledBinaryWithStreamAndCaptures<F>
where
    F: 'static,
{
    fn new(
        f: F,
        captures: impl IntoIterator<Item = Array>,
        options: impl Into<CompileOptions>,
    ) -> Self {
        let id = type_id_to_usize(&f);
        Self {
            f,
            captures: captures.into_iter().collect(),
            options: options.into(),
            id,
        }
    }

    /// Replace the hidden array captures used on subsequent calls.
    pub fn set_captures(&mut self, captures: impl IntoIterator<Item = Array>) {
        self.captures = captures.into_iter().collect();
    }

    /// Return the currently configured hidden array captures.
    pub fn captures(&self) -> &[Array] {
        &self.captures
    }
}

impl<F> CompiledBinaryWithStreamAndCaptures<F>
where
    F: FnMut((&Array, &Array), &[Array], &Stream) -> Result<Array, Exception>,
{
    /// Call the compiled function with two public array inputs and an explicit
    /// stream.
    pub fn call(&mut self, x: &Array, y: &Array, stream: &Stream) -> Result<Array, Exception> {
        let captures_len = self.captures.len();
        let f = &mut self.f;
        let mut inner = move |inputs: &[Array]| -> Result<Vec<Array>, Exception> {
            let (args, captures) = inputs.split_at(2);
            debug_assert_eq!(captures.len(), captures_len);
            Ok(vec![f((&args[0], &args[1]), captures, stream)?])
        };
        let inner_closure = Closure::new_fallible(&mut inner);
        let inputs = combined_inputs(&[x, y], &self.captures);
        let result = call_mut_inner(
            inner_closure,
            self.id,
            self.options.shapeless,
            &self.options.constants,
            &inputs,
        )?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<F> Clone for CompiledBinaryWithStreamAndCaptures<F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            captures: self.captures.clone(),
            options: self.options.clone(),
            id: self.id,
        }
    }
}

impl<F> std::fmt::Debug for CompiledBinaryWithStreamAndCaptures<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledBinaryWithStreamAndCaptures")
            .field("captures", &self.captures.len())
            .field("options", &self.options)
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<F> Drop for CompiledBinaryWithStreamAndCaptures<F> {
    fn drop(&mut self) {
        erase_compiled(self.id);
    }
}

/// A reusable compiled unary function that is called with an explicit stream.
pub struct CompiledUnaryWithStream<F> {
    f: F,
    shapeless: bool,
    id: usize,
}

impl<F> CompiledUnaryWithStream<F>
where
    F: 'static,
{
    fn new(f: F, shapeless: impl Into<Option<bool>>) -> Self {
        let id = type_id_to_usize(&f);
        Self {
            f,
            shapeless: shapeless.into().unwrap_or(false),
            id,
        }
    }
}

impl<F> CompiledUnaryWithStream<F>
where
    F: FnMut(&Array, &Stream) -> Result<Array, Exception>,
{
    /// Call the compiled function with one array input and an explicit stream.
    pub fn call(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        let f = &mut self.f;
        let mut inner = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            Ok(vec![f(&args[0], stream)?])
        };
        let inner_closure = Closure::new_fallible(&mut inner);
        let result = call_mut_inner(inner_closure, self.id, self.shapeless, &[], &[x])?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<F> Clone for CompiledUnaryWithStream<F>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            f: self.f.clone(),
            shapeless: self.shapeless,
            id: self.id,
        }
    }
}

impl<F> std::fmt::Debug for CompiledUnaryWithStream<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledUnaryWithStream")
            .field("shapeless", &self.shapeless)
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<F> Drop for CompiledUnaryWithStream<F> {
    fn drop(&mut self) {
        erase_compiled(self.id);
    }
}

/// A trait for functions that can be compiled.
///
/// # Generic parameters
///
/// - `A`: The type of the array arguments
/// - `O`: The type of the output
/// - `E`: The type of the error
pub trait Compile<A, O, E>: Sized {
    /// The type of the arguments that the returned closure takes.
    ///
    /// This is needed to relax the lifetime requirements of the returned
    /// closure. Otherwise, the arguments to the returned closure would have to
    /// live longer than the closure itself.
    type Args<'a>;

    /// Compiles the function.
    fn compile<'args, 'compiled>(
        self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, O, E> + 'compiled
    where
        Self: 'compiled;
}

impl<F> Compile<&[Array], Vec<Array>, ()> for F
where
    F: FnMut(&[Array]) -> Vec<Array> + 'static,
{
    type Args<'a> = &'a [Array];

    fn compile<'args, 'compiled>(
        self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, Vec<Array>, ()> + 'compiled
    where
        Self: 'compiled,
    {
        let id = type_id_to_usize(&self);
        let state = CompiledState {
            f: self,

            shapeless,
            id,
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<&Array, Array, ()> for F
where
    F: FnMut(&Array) -> Array + 'static,
{
    type Args<'a> = &'a Array;

    fn compile<'args, 'compiled>(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, Array, ()> + 'compiled
    where
        Self: 'compiled,
    {
        let id = type_id_to_usize(&self);
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)(&args[0]);
            vec![result]
        };
        let state = CompiledState { f, shapeless, id };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<(&Array, &Array), Array, ()> for F
where
    F: FnMut((&Array, &Array)) -> Array + 'static,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn compile<'args, 'compiled>(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, Array, ()> + 'compiled
    where
        Self: 'compiled,
    {
        let id = type_id_to_usize(&self);
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)((&args[0], &args[1]));
            vec![result]
        };
        let state = CompiledState { f, shapeless, id };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<(&Array, &Array, &Array), Array, ()> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Array + 'static,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn compile<'args, 'compiled>(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, Array, ()> + 'compiled
    where
        Self: 'compiled,
    {
        let id = type_id_to_usize(&self);
        let f = move |args: &[Array]| -> Vec<Array> {
            let result = (self)((&args[0], &args[1], &args[2]));
            vec![result]
        };
        let state = CompiledState { f, shapeless, id };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<&[Array], Vec<Array>, Exception> for F
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'static,
{
    type Args<'a> = &'a [Array];

    fn compile<'args, 'compiled>(
        self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, Vec<Array>, Exception> + 'compiled
    where
        Self: 'compiled,
    {
        let id = type_id_to_usize(&self);
        let state = CompiledState {
            f: self,
            shapeless,
            id,
        };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<&Array, Array, Exception> for F
where
    F: FnMut(&Array) -> Result<Array, Exception> + 'static,
{
    type Args<'a> = &'a Array;

    fn compile<'args, 'compiled>(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, Array, Exception> + 'compiled
    where
        Self: 'compiled,
    {
        let id = type_id_to_usize(&self);
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)(&args[0])?;
            Ok(vec![result])
        };
        let state = CompiledState { f, shapeless, id };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<(&Array, &Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array);

    fn compile<'args, 'compiled>(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, Array, Exception> + 'compiled
    where
        Self: 'compiled,
    {
        let id = type_id_to_usize(&self);
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)((&args[0], &args[1]))?;
            Ok(vec![result])
        };
        let state = CompiledState { f, shapeless, id };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

impl<F> Compile<(&Array, &Array, &Array), Array, Exception> for F
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> + 'static,
{
    type Args<'a> = (&'a Array, &'a Array, &'a Array);

    fn compile<'args, 'compiled>(
        mut self,
        shapeless: bool,
    ) -> impl CallMut<Self::Args<'args>, Array, Exception> + 'compiled
    where
        Self: 'compiled,
    {
        let id = type_id_to_usize(&self);
        let f = move |args: &[Array]| -> Result<Vec<Array>, Exception> {
            let result = (self)((&args[0], &args[1], &args[2]))?;
            Ok(vec![result])
        };
        let state = CompiledState { f, shapeless, id };
        Compiled {
            f_marker: PhantomData::<F>,
            state,
        }
    }
}

/// A trait for a compiled function that can be called.
pub trait CallMut<A, O, E> {
    /// Calls the compiled function with the given arguments.
    fn call_mut(&mut self, args: A) -> Result<O, Exception>;
}

impl<'a, F, G> CallMut<&'a [Array], Vec<Array>, ()> for Compiled<F, G>
where
    F: FnMut(&[Array]) -> Vec<Array> + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.call_mut(args)
    }
}

impl<'a, F, G> CallMut<&'a Array, Array, ()> for Compiled<F, G>
where
    F: FnMut(&Array) -> Array + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: &Array) -> Result<Array, Exception> {
        let args = std::slice::from_ref(args);
        let result = self.state.call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array), Array, ()> for Compiled<F, G>
where
    F: FnMut((&Array, &Array)) -> Array + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1];
        let result = self.state.call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array, &'a Array), Array, ()> for Compiled<F, G>
where
    F: FnMut((&Array, &Array, &Array)) -> Array + 'a,
    G: FnMut(&[Array]) -> Vec<Array> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array, &Array)) -> Result<Array, Exception> {
        // Is there any way to avoid this shallow clone?
        let args = &[args.0, args.1, args.2];
        let result = self.state.call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<&'a [Array], Vec<Array>, Exception> for Compiled<F, G>
where
    F: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: &[Array]) -> Result<Vec<Array>, Exception> {
        self.state.fallible_call_mut(args)
    }
}

impl<'a, F, G> CallMut<&'a Array, Array, Exception> for Compiled<F, G>
where
    F: FnMut(&Array) -> Result<Array, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: &Array) -> Result<Array, Exception> {
        let args = &[args];
        let result = self.state.fallible_call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array), Array, Exception> for Compiled<F, G>
where
    F: FnMut((&Array, &Array)) -> Result<Array, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1];
        let result = self.state.fallible_call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

impl<'a, F, G> CallMut<(&'a Array, &'a Array, &'a Array), Array, Exception> for Compiled<F, G>
where
    F: FnMut((&Array, &Array, &Array)) -> Result<Array, Exception> + 'a,
    G: FnMut(&[Array]) -> Result<Vec<Array>, Exception> + 'a,
{
    fn call_mut(&mut self, args: (&Array, &Array, &Array)) -> Result<Array, Exception> {
        let args = &[args.0, args.1, args.2];
        let result = self.state.fallible_call_mut(args)?;
        Ok(result.into_iter().next().unwrap())
    }
}

#[inline]
fn call_mut_inner(
    inner_closure: Closure,
    fun_id: usize,
    shapeless: bool,
    constants: &[u64],
    args: &[impl AsRef<Array>],
) -> crate::error::Result<Vec<Array>> {
    let _transform_guard = transform_guard::enter();

    // note: this will use the cached compile (via the id)
    // but will be able to re-evaluate with fresh state if needed
    let compiled = Closure::try_from_op(|res| unsafe {
        safemlx_sys::mlx_detail_compile(
            res,
            inner_closure.as_ptr(),
            fun_id,
            shapeless,
            constants.as_ptr(),
            0,
        )
    })?;

    let inner_inputs_vector = VectorArray::try_from_iter(args.iter())?;

    // will compile the function (if needed) and evaluate the
    // compiled graph
    let result_vector = VectorArray::try_from_op(|res| unsafe {
        safemlx_sys::mlx_closure_apply(res, compiled.as_ptr(), inner_inputs_vector.as_ptr())
    })?;
    let result_plus_state_output: Vec<Array> = result_vector.try_into_values()?;

    let result_len = result_plus_state_output.len();
    Ok(result_plus_state_output
        .into_iter()
        .take(result_len)
        .collect())
}

fn combined_inputs(args: &[impl AsRef<Array>], captures: &[Array]) -> Vec<Array> {
    args.iter()
        .map(|arg| arg.as_ref().clone())
        .chain(captures.iter().cloned())
        .collect()
}

fn erase_compiled(id: usize) {
    let _transform_guard = transform_guard::enter();
    unsafe {
        safemlx_sys::mlx_detail_compile_erase(id);
    }
}

impl<F> CompiledState<F> {
    fn call_mut(&mut self, args: &[impl AsRef<Array>]) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Vec<Array>,
    {
        let inner_closure = Closure::new(&mut self.f);

        call_mut_inner(inner_closure, self.id, self.shapeless, &[], args)
    }

    fn fallible_call_mut(&mut self, args: &[impl AsRef<Array>]) -> Result<Vec<Array>, Exception>
    where
        F: FnMut(&[Array]) -> Result<Vec<Array>, Exception>,
    {
        let inner_closure = Closure::new_fallible(&mut self.f);

        call_mut_inner(inner_closure, self.id, self.shapeless, &[], args)
    }
}

#[cfg(test)]
mod tests {
    use core::panic;

    use crate::{
        array,
        error::Exception,
        ops::{multiply, ones},
        Array,
    };

    use super::{
        compile, compile_binary_with_stream_and_captures, compile_unary_with_stream,
        compile_unary_with_stream_and_captures, compile_with_stream_and_captures,
    };

    fn example_fn_0(x: f32) -> f32 {
        x + 1.0
    }

    fn example_fn_3(x: f32) -> f32 {
        x + 1.0
    }

    fn explicit_stream_square(x: &Array, stream: &crate::Stream) -> Result<Array, Exception> {
        x.multiply(x, stream)
    }

    fn explicit_stream_captured_affine(
        x: &Array,
        captures: &[Array],
        stream: &crate::Stream,
    ) -> Result<Array, Exception> {
        x.multiply(&captures[0], stream)?.add(&captures[1], stream)
    }

    fn explicit_stream_binary_captured_affine(
        (x, y): (&Array, &Array),
        captures: &[Array],
        stream: &crate::Stream,
    ) -> Result<Array, Exception> {
        x.multiply(&captures[0], stream)?
            .add(y.multiply(&captures[1], stream)?, stream)
    }

    fn explicit_stream_multi_captured(
        args: &[Array],
        captures: &[Array],
        stream: &crate::Stream,
    ) -> Result<Vec<Array>, Exception> {
        let sum = args[0].add(&args[1], stream)?.add(&captures[0], stream)?;
        let product = args[0]
            .multiply(&args[2], stream)?
            .multiply(&captures[1], stream)?;
        Ok(vec![sum, product])
    }

    #[test]
    fn test_type_id_to_usize() {
        // We would like to check that different functions that share the same signature can produce
        // different ids

        let example_fn_1 = |x: f32| x + 1.0;
        let example_fn_2 = |x: f32| x + 1.0;

        let mut ids = Vec::new();

        ids.push(super::type_id_to_usize(&example_fn_0));

        let id1 = super::type_id_to_usize(&example_fn_1);
        if ids.contains(&id1) {
            panic!("id1 already exists");
        }
        ids.push(id1);

        let id2 = super::type_id_to_usize(&example_fn_2);
        if ids.contains(&id2) {
            panic!("id2 already exists");
        }
        ids.push(id2);

        let id3 = super::type_id_to_usize(&example_fn_3);
        if ids.contains(&id3) {
            panic!("id3 already exists");
        }
        ids.push(id3);

        assert_eq!(ids.len(), 4);
    }

    #[test]
    fn test_compile() {
        // This unit test is modified from the mlx-swift codebase
        let stream = crate::test_stream();

        let f = move |inputs: &[Array]| -> Vec<Array> {
            vec![inputs[0].multiply(&inputs[1], stream).unwrap()]
        };
        let mut compiled = compile(f, None);

        let i1 = ones::<f32>(&[20, 20], stream).unwrap();
        let i2 = ones::<f32>(&[20, 20], stream).unwrap();

        let args = [i1, i2];

        // evaluate directly
        let r1 = f(&args).drain(0..1).next().unwrap();
        // evaluate compiled
        let r2 = compiled(&args).unwrap().drain(0..1).next().unwrap();

        assert!(crate::array::eval_equal_values(&r1, &r2));

        let r3 = compiled(&args).unwrap().drain(0..1).next().unwrap();
        assert!(crate::array::eval_equal_values(&r1, &r3));
    }

    #[test]
    fn test_compile_with_error() {
        let stream = crate::test_stream();
        let f = move |inputs: &[Array]| -> Result<Vec<Array>, Exception> {
            multiply(&inputs[0], &inputs[1], stream).map(|x| vec![x])
        };

        // Success case
        let i1 = ones::<f32>(&[20, 20], stream).unwrap();
        let i2 = ones::<f32>(&[20, 20], stream).unwrap();
        let args = [i1, i2];

        // evaluate directly
        let r1 = f(&args).unwrap().drain(0..1).next().unwrap();

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled(&args).unwrap().drain(0..1).next().unwrap();

        assert!(crate::array::eval_equal_values(&r1, &r2));

        let r3 = compiled(&args).unwrap().drain(0..1).next().unwrap();
        assert!(crate::array::eval_equal_values(&r1, &r3));

        // Error case
        let a = array!([1.0, 2.0, 3.0]);
        let b = array!([4.0, 5.0]);
        let args = [a, b];

        // The cache is keyed by function pointer and argument shapes
        let c = array!([4.0, 5.0, 6.0]);
        let d = array!([7.0, 8.0]);
        let another_args = [c, d];

        // evaluate directly
        let result = f(&args);
        assert!(result.is_err());

        // evaluate compiled
        let mut compiled = compile(f, None);
        let result = compiled(&args);
        assert!(result.is_err());

        let result = compiled(&args);
        assert!(result.is_err());

        let result = compiled(&another_args);
        assert!(result.is_err());
    }

    #[test]
    fn test_compile_with_one_arg() {
        let stream = crate::test_stream();
        let f = move |x: &Array| x.multiply(x, stream).unwrap();

        let i = ones::<f32>(&[20, 20], stream).unwrap();

        // evaluate directly
        let r1 = f(&i);

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled(&i).unwrap();

        assert!(crate::array::eval_equal_values(&r1, &r2));

        let r3 = compiled(&i).unwrap();
        assert!(crate::array::eval_equal_values(&r1, &r3));
    }

    #[test]
    fn test_compile_unary_with_explicit_stream() {
        let stream = crate::test_stream();
        let i = ones::<f32>(&[20, 20], stream).unwrap();
        let r1 = explicit_stream_square(&i, stream).unwrap();

        let mut compiled = compile_unary_with_stream(explicit_stream_square, None);
        let r2 = compiled.call(&i, stream).unwrap();
        assert!(crate::array::eval_equal_values(&r1, &r2));

        let r3 = compiled.call(&i, stream).unwrap();
        assert!(crate::array::eval_equal_values(&r1, &r3));
    }

    #[test]
    fn test_compile_unary_with_explicit_stream_and_captures() {
        let stream = crate::test_stream();
        let x = ones::<f32>(&[20, 20], stream).unwrap();
        let one = ones::<f32>(&[20, 20], stream).unwrap();
        let two = one.add(&one, stream).unwrap();
        let three = two.add(&one, stream).unwrap();

        let mut compiled = compile_unary_with_stream_and_captures(
            explicit_stream_captured_affine,
            vec![one.clone(), one.clone()],
            None,
        );
        let expected =
            explicit_stream_captured_affine(&x, &[one.clone(), one.clone()], stream).unwrap();
        let actual = compiled.call(&x, stream).unwrap();
        assert!(crate::array::eval_equal_values(&expected, &actual));

        compiled.set_captures(vec![two.clone(), three.clone()]);
        let expected = explicit_stream_captured_affine(&x, &[two, three], stream).unwrap();
        let actual = compiled.call(&x, stream).unwrap();
        assert!(crate::array::eval_equal_values(&expected, &actual));
    }

    #[test]
    fn test_compile_binary_with_explicit_stream_and_captures() {
        let stream = crate::test_stream();
        let one = ones::<f32>(&[20, 20], stream).unwrap();
        let two = one.add(&one, stream).unwrap();
        let three = two.add(&one, stream).unwrap();

        let expected = explicit_stream_binary_captured_affine(
            (&one, &two),
            &[two.clone(), three.clone()],
            stream,
        )
        .unwrap();
        let mut compiled = compile_binary_with_stream_and_captures(
            explicit_stream_binary_captured_affine,
            vec![two, three],
            None,
        );
        let actual = compiled
            .call(&one, &one.add(&one, stream).unwrap(), stream)
            .unwrap();
        assert!(crate::array::eval_equal_values(&expected, &actual));
    }

    #[test]
    fn test_compile_with_explicit_stream_and_captures() {
        let stream = crate::test_stream();
        let one = ones::<f32>(&[20, 20], stream).unwrap();
        let two = one.add(&one, stream).unwrap();
        let three = two.add(&one, stream).unwrap();
        let four = three.add(&one, stream).unwrap();

        let args = [one.clone(), two.clone(), three.clone()];
        let mut compiled = compile_with_stream_and_captures(
            explicit_stream_multi_captured,
            vec![two.clone(), three.clone()],
            None,
        );
        let expected =
            explicit_stream_multi_captured(&args, &[two.clone(), three.clone()], stream).unwrap();
        let actual = compiled.call(&args, stream).unwrap();
        assert_eq!(actual.len(), 2);
        assert!(crate::array::eval_equal_values(&expected[0], &actual[0]));
        assert!(crate::array::eval_equal_values(&expected[1], &actual[1]));

        compiled.set_captures(vec![three.clone(), four.clone()]);
        let expected = explicit_stream_multi_captured(&args, &[three, four], stream).unwrap();
        let actual = compiled.call(&args, stream).unwrap();
        assert_eq!(actual.len(), 2);
        assert!(crate::array::eval_equal_values(&expected[0], &actual[0]));
        assert!(crate::array::eval_equal_values(&expected[1], &actual[1]));
    }

    #[test]
    fn test_compile_with_two_args() {
        let stream = crate::test_stream();
        let f = move |(x, y): (&Array, &Array)| x.multiply(y, stream).unwrap();

        let i1 = ones::<f32>(&[20, 20], stream).unwrap();
        let i2 = ones::<f32>(&[20, 20], stream).unwrap();

        // evaluate directly
        let r1 = f((&i1, &i2));

        // evaluate compiled
        let mut compiled = compile(f, None);
        let r2 = compiled((&i1, &i2)).unwrap();

        assert!(crate::array::eval_equal_values(&r1, &r2));

        let r3 = compiled((&i1, &i2)).unwrap();
        assert!(crate::array::eval_equal_values(&r1, &r3));
    }

    #[test]
    fn test_compile_with_three_args() {
        let stream = crate::test_stream();
        let f = move |(x, y, z): (&Array, &Array, &Array)| {
            x.multiply(y, stream).unwrap().multiply(z, stream).unwrap()
        };
        let mut compiled = compile(f, None);

        let i1 = ones::<f32>(&[20, 20], stream).unwrap();
        let i2 = ones::<f32>(&[20, 20], stream).unwrap();
        let i3 = ones::<f32>(&[20, 20], stream).unwrap();

        // evaluate directly
        let r1 = f((&i1, &i2, &i3));

        // evaluate compiled
        let r2 = compiled((&i1, &i2, &i3)).unwrap();

        assert!(crate::array::eval_equal_values(&r1, &r2));

        let r3 = compiled((&i1, &i2, &i3)).unwrap();
        assert!(crate::array::eval_equal_values(&r1, &r3));
    }
}
