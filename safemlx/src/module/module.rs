use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    hash::Hash,
    path::Path,
    rc::Rc,
};

use crate::{
    error::{Exception, IoError},
    nested::{NestedHashMap, NestedValue},
    Array, Stream,
};

/// Type alias for owned module parameters.
pub type ModuleParam = NestedHashMap<Rc<str>, Array>;

/// Type alias for borrowed module parameters.
pub type ModuleParamRef<'a> = NestedHashMap<Rc<str>, &'a Array>;

/// Type alias for mutably borrowed module parameters.
pub type ModuleParamMut<'a> = NestedHashMap<Rc<str>, &'a mut Array>;

/// Type alias for flattened module parameters.
pub type FlattenedModuleParam = HashMap<Rc<str>, Array>;

/// Type alias for borrowed flattened module parameters.
pub type FlattenedModuleParamRef<'a> = HashMap<Rc<str>, &'a Array>;

/// Type alias for mutably borrowed flattened module parameters.
pub type FlattenedModuleParamMut<'a> = HashMap<Rc<str>, &'a mut Array>;

/// Trait for a neural network module.
pub trait Module<Input>: ModuleParameters + std::fmt::Debug {
    /// Output type of the module.
    type Output;

    /// Error type for the module.
    type Error: std::error::Error;

    /// Forward pass of the module.
    fn forward(&mut self, input: Input, stream: &Stream) -> Result<Self::Output, Self::Error>;

    /// Set whether the module is in training mode.
    ///
    /// Training mode only applies to certain layers. For example, dropout layers applies a random
    /// mask in training mode, but is the identity in evaluation mode. Implementations of nested
    /// modules should propagate the training mode to all child modules.
    fn training_mode(&mut self, mode: bool);
}

/// Marker trait for a unary neural network module.
///
/// This trait should not be implemented directly. Instead, implement [`Module`] with `Args` as a
/// reference to the input.
pub trait UnaryModule: for<'a> Module<&'a Array, Output = Array> {}

impl<T> UnaryModule for T where T: for<'a> Module<&'a Array, Output = Array> {}

/// Trait for accessing and updating module parameters.
pub trait ModuleParameters {
    /// Get the total number of parameters in the module.
    ///
    /// Returns the total number of parameters in the module without counting
    /// the parameters iterator. `module.parameters().flatten().len()`
    fn num_parameters(&self) -> usize;

    /// Get references to the module parameters.
    fn parameters(&self) -> ModuleParamRef<'_>;

    /// Get mutable references to the module parameters.
    fn parameters_mut(&mut self) -> ModuleParamMut<'_>;

    /// Get references to the trainable parameters. A parameter is trainable if it is NOT frozen.
    fn trainable_parameters(&self) -> ModuleParamRef<'_>;

    /// Update the module parameters.
    fn update(&mut self, parameters: ModuleParam) {
        let flattened_parameters = parameters.flatten();
        update_parameters(self, flattened_parameters)
    }

    /// Update the module parameters from a flattened representation.
    fn update_flattened(&mut self, flattened_parameters: FlattenedModuleParam) {
        update_parameters(self, flattened_parameters)
    }

    /// Freeze all parameters in the module.
    fn freeze_parameters(&mut self, recursive: bool);

    /// Unfreeze all parameters in the module.
    fn unfreeze_parameters(&mut self, recursive: bool);

    /// Check if all parameters in the module are frozen. Returns `None` if there are no parameters.
    fn all_frozen(&self) -> Option<bool>;

    /// Check if any parameter in the module is frozen. Returns `None` if there are no parameters.
    fn any_frozen(&self) -> Option<bool>;
}

/// Update the module parameters from an iterator of (key, value) tuples.
pub fn update_parameters<M, I, Q>(module: &mut M, parameters: I)
where
    M: ModuleParameters + ?Sized,
    I: IntoIterator<Item = (Q, Array)>,
    Q: Hash + Eq,
    Rc<str>: Borrow<Q>,
{
    let mut flattened_self_parameters = module.parameters_mut().flatten();

    for (key, value) in parameters {
        if let Some(self_value) = flattened_self_parameters.get_mut(&key) {
            **self_value = value;
        }
    }
}

impl<T> ModuleParameters for &'_ mut T
where
    T: ModuleParameters + ?Sized,
{
    fn num_parameters(&self) -> usize {
        (**self).num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        (**self).parameters()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        (**self).parameters_mut()
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        (**self).trainable_parameters()
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        (**self).freeze_parameters(recursive);
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        (**self).unfreeze_parameters(recursive);
    }

    fn all_frozen(&self) -> Option<bool> {
        (**self).all_frozen()
    }

    fn any_frozen(&self) -> Option<bool> {
        (**self).any_frozen()
    }
}

impl<T> ModuleParameters for Box<T>
where
    T: ModuleParameters + ?Sized,
{
    fn num_parameters(&self) -> usize {
        self.as_ref().num_parameters()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        self.as_ref().parameters()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        self.as_mut().parameters_mut()
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.as_ref().trainable_parameters()
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        self.as_mut().freeze_parameters(recursive);
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.as_mut().unfreeze_parameters(recursive);
    }

    fn all_frozen(&self) -> Option<bool> {
        self.as_ref().all_frozen()
    }

    fn any_frozen(&self) -> Option<bool> {
        self.as_ref().any_frozen()
    }
}

impl<T> ModuleParameters for Vec<T>
where
    T: ModuleParameters,
{
    fn num_parameters(&self) -> usize {
        self.iter().map(|module| module.num_parameters()).sum()
    }

    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut parameters = NestedHashMap::new();
        self.iter().enumerate().for_each(|(i, module)| {
            let value = module.parameters();
            parameters.insert(Rc::from(i.to_string()), NestedValue::Map(value.entries));
        });
        parameters
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut parameters = NestedHashMap::new();
        self.iter_mut().enumerate().for_each(|(i, module)| {
            let value = module.parameters_mut();
            parameters.insert(Rc::from(i.to_string()), NestedValue::Map(value.entries));
        });
        parameters
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut parameters = NestedHashMap::new();
        self.iter().enumerate().for_each(|(i, module)| {
            let value = module.trainable_parameters();
            parameters.insert(Rc::from(i.to_string()), NestedValue::Map(value.entries));
        });
        parameters
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        self.iter_mut().for_each(|module| {
            module.freeze_parameters(recursive);
        });
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.iter_mut().for_each(|module| {
            module.unfreeze_parameters(recursive);
        });
    }

    fn all_frozen(&self) -> Option<bool> {
        let mut result = None;
        for module in self.iter() {
            match module.all_frozen() {
                Some(true) => result = Some(true),
                Some(false) => return Some(false),
                None => {}
            }
        }
        result
    }

    fn any_frozen(&self) -> Option<bool> {
        let mut result = None;
        for module in self.iter() {
            match module.any_frozen() {
                Some(true) => return Some(true),
                Some(false) => result = Some(false),
                None => {}
            }
        }
        result
    }
}

/// Extension trait for `ModuleParameters`. This is implemented for all types that implement
/// `ModuleParameters`.
pub trait ModuleParametersExt: ModuleParameters {
    /// Evaluate the module parameters.
    fn eval(&self) -> Result<(), Exception> {
        crate::transforms::eval_params(self.parameters())
    }

    /// Load module parameters from a `safetensors` file.
    fn load_safetensors(
        &mut self,
        path: impl AsRef<Path>,
        stream: impl AsRef<Stream>,
    ) -> Result<(), IoError> {
        let loaded = Array::load_safetensors(path, stream)?;

        // Load the parameters
        let mut params = self.parameters_mut().flatten();
        for (key, value) in loaded {
            if let Some(param) = params.get_mut(&*key) {
                **param = value;
            }
        }

        Ok(())
    }

    /// Copy all module parameters onto the given stream and evaluate them.
    fn copy_to_stream(&mut self, stream: impl AsRef<Stream>) -> Result<(), Exception> {
        let stream = stream.as_ref();
        let mut params = self.parameters_mut().flatten();
        for param in params.values_mut() {
            **param = param.copy(stream)?;
        }
        self.eval()
    }

    /// Copy and evaluate only the named module parameters on the given stream.
    ///
    /// This is the selective counterpart to [`Self::copy_to_stream`] for
    /// rank-aware loaders. Unknown names are rejected before any copy occurs,
    /// and parameters outside `names` are never evaluated or copied.
    fn copy_parameters_to_stream<I, S>(
        &mut self,
        names: I,
        stream: impl AsRef<Stream>,
    ) -> Result<(), Exception>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let names = names
            .into_iter()
            .map(|name| name.as_ref().to_string())
            .collect::<HashSet<_>>();
        let stream = stream.as_ref();
        {
            let params = self.parameters().flatten();
            if let Some(missing) = names
                .iter()
                .find(|name| !params.contains_key(name.as_str()))
            {
                return Err(Exception::custom(format!(
                    "unknown module parameter selected for stream copy: {missing}"
                )));
            }
        }
        {
            let mut params = self.parameters_mut().flatten();
            for name in &names {
                let param = params
                    .get_mut(name.as_str())
                    .expect("parameter selection was validated");
                **param = param.copy(stream)?;
            }
        }
        let params = self.parameters().flatten();
        crate::transforms::eval(
            names
                .iter()
                .map(|name| *params.get(name.as_str()).expect("parameter was validated")),
        )
    }

    /// Save module parameters to a file in `safetensors` format.
    fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<(), IoError> {
        let params = self.parameters().flatten();
        Array::save_safetensors(params, None, path)?;
        Ok(())
    }
}

impl<T: ModuleParameters> ModuleParametersExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{macros::ModuleParameters, module::Param, Device, DeviceType};

    #[derive(ModuleParameters)]
    #[module(root = crate)]
    struct TwoParameters {
        #[param]
        local: Param<Array>,
        #[param]
        remote: Param<Array>,
    }

    #[test]
    fn selective_stream_copy_does_not_touch_unselected_parameters() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let mut module = TwoParameters {
            local: Param::new(Array::from_slice(&[1i32, 2], &[2])),
            remote: Param::new(Array::from_slice(&[3i32, 4], &[2])),
        };
        let local_before = module.local.as_ptr().ctx;
        let remote_before = module.remote.as_ptr().ctx;
        module
            .copy_parameters_to_stream(["local"], &stream)
            .unwrap();
        assert_ne!(module.local.as_ptr().ctx, local_before);
        assert_eq!(module.remote.as_ptr().ctx, remote_before);

        let local_after = module.local.as_ptr().ctx;
        assert!(module
            .copy_parameters_to_stream(["does_not_exist"], &stream)
            .is_err());
        assert_eq!(module.local.as_ptr().ctx, local_after);
        assert_eq!(module.remote.as_ptr().ctx, remote_before);
    }
}
