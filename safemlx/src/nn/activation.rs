use std::f32::consts::PI;

use crate::module::{Module, Param};
use crate::{
    array,
    error::{Exception, Result},
    ops::{abs, erf, exp, logsumexp_axis, maximum, minimum, multiply, sqrt, tanh, which},
    Array,
};
use safemlx_internal_macros::{generate_builder, Buildable, Builder};
use safemlx_macros::ModuleParameters;

/// Applies the element-wise sigmoid logistic sigmoid.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sigmoid.html)
///
/// This is:
///
/// ```rust, ignore
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// sigmoid(x, &stream)
/// ```
pub fn sigmoid(x: impl AsRef<Array>, stream: impl AsRef<crate::Stream>) -> Result<Array> {
    crate::ops::sigmoid(x.as_ref(), stream)
}

/// Applies the Rectified Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// maximum(x, 0)
/// ```
pub fn relu(x: impl AsRef<Array>, stream: impl AsRef<crate::Stream>) -> Result<Array> {
    crate::ops::maximum(x.as_ref(), &array!(0), stream)
}

/// Applies the Leaky Rectified Linear Unit.
///
/// `neg_slope` is default to 0.01 if not provided.
///
/// This is:
///
/// ```rust, ignore
/// maximum(neg_slope * x, x)
/// ```
pub fn leaky_relu(
    x: impl AsRef<Array>,
    neg_slope: impl Into<Option<f32>>,
    stream: impl AsRef<crate::Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    let neg_slope = array!(neg_slope.into().unwrap_or(0.01));
    maximum(
        &multiply(&neg_slope, x.as_ref(), stream)?,
        x.as_ref(),
        stream,
    )
}

/// Applies the Log Softmax function.
///
/// This is:
///
/// ```rust, ignore
/// x - logsumexp_axis(x, axis, true)
/// ```
pub fn log_softmax(
    x: impl AsRef<Array>,
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<crate::Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    let x = x.as_ref();
    let axis = axis.into().unwrap_or(-1);
    x.subtract(logsumexp_axis(x, axis, true, stream)?, stream)
}

/// Applies the Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// which(x.gt(0), x, alpha * (exp(x) - 1))
/// ```
///
/// # Params
///
/// - `x`: The input array
/// - `alpha`: Default to 1.0 if not provided
pub fn elu(
    x: impl AsRef<Array>,
    alpha: impl Into<Option<f32>>,
    stream: impl AsRef<crate::Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    let x = x.as_ref();
    let alpha = array!(alpha.into().unwrap_or(1.0));
    which(
        &x.gt(&array!(0.0), stream)?,
        x,
        alpha.multiply(exp(x, stream)?.subtract(array!(1.0), stream)?, stream)?,
        stream,
    )
}

/// Applies the Rectified Linear Unit 6.
///
/// This is:
///
/// ```rust, ignore
/// minimum(maximum(x, 0), 6)
/// ```
pub fn relu6(x: impl AsRef<Array>, stream: impl AsRef<crate::Stream>) -> Result<Array> {
    let stream = stream.as_ref();
    minimum(
        maximum(x.as_ref(), &array!(0.0), stream)?,
        &array!(6.0),
        stream,
    )
}

/// Applies the Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// logaddexp(x, 0)
/// ```
pub fn softplus(x: impl AsRef<Array>, stream: impl AsRef<crate::Stream>) -> Result<Array> {
    crate::ops::logaddexp(x.as_ref(), &array!(0), stream)
}

/// Applies the Softsign function.
///
/// This is:
///
/// ```rust, ignore
/// x / (1 + abs(x))
/// ```
pub fn softsign(x: impl AsRef<Array>, stream: impl AsRef<crate::Stream>) -> Result<Array> {
    let stream = stream.as_ref();
    x.as_ref()
        .divide(array!(1.0).add(abs(x.as_ref(), stream)?, stream)?, stream)
}

/// Applies the Continuously Differentiable Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// maximum(x, 0) + alpha * (exp(minimum(x, 0) / alpha) - 1)
/// ```
pub fn celu(
    x: impl AsRef<Array>,
    alpha: impl Into<Option<f32>>,
    stream: impl AsRef<crate::Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    let x = x.as_ref();
    let alpha = array!(alpha.into().unwrap_or(1.0));
    maximum(x, &array!(0.0), stream)?.add(
        alpha.multiply(
            exp(
                &minimum(x, &array!(0.0), stream)?.divide(&alpha, stream)?,
                stream,
            )?
            .subtract(array!(1.0), stream)?,
            stream,
        )?,
        stream,
    )
}

/// Applies the Sigmoid Linear Unit. Also known as Swish.
///
/// This is:
///
/// ```rust, ignore
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// x * sigmoid(x, &stream)
/// ```
pub fn silu(x: impl AsRef<Array>, stream: impl AsRef<crate::Stream>) -> Result<Array> {
    let stream = stream.as_ref();
    x.as_ref().multiply(sigmoid(x.as_ref(), stream)?, stream)
}

/// Applies the Log Sigmoid function.
///
/// This is:
///
/// ```rust, ignore
/// -softplus(-x)
/// ```
pub fn log_sigmoid(x: impl AsRef<Array>, stream: impl AsRef<crate::Stream>) -> Result<Array> {
    let stream = stream.as_ref();
    softplus(&x.as_ref().negative(stream)?, stream)?.negative(stream)
}

/// Applies the Gaussian Error Linear Units function.
///
/// This is:
///
/// ```rust, ignore
/// x * (1 + erf(x / 2.sqrt())) / 2
/// ```
pub fn gelu(x: impl AsRef<Array>, stream: impl AsRef<crate::Stream>) -> Result<Array> {
    let stream = stream.as_ref();
    x.as_ref()
        .multiply(
            array!(1).add(
                erf(&x.as_ref().divide(array!(2f32.sqrt()), stream)?, stream)?,
                stream,
            )?,
            stream,
        )?
        .divide(array!(2.0), stream)
}

/// An approximation to Gaussian Error Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x ** 3)))
/// ```
pub fn gelu_approximate(x: impl AsRef<Array>, stream: impl AsRef<crate::Stream>) -> Result<Array> {
    let stream = stream.as_ref();
    let x = x.as_ref();
    array!(0.5).multiply(x, stream)?.multiply(
        array!(1.0).add(
            tanh(
                &sqrt(&array!(2.0 / PI), stream)?.multiply(
                    x.add(
                        array!(0.044715).multiply(x.power(&array!(3), stream)?, stream)?,
                        stream,
                    )?,
                    stream,
                )?,
                stream,
            )?,
            stream,
        )?,
        stream,
    )
}

/// A fast approximation to Gaussian Error Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// x * sigmoid(1.773 * x)
/// ```
pub fn gelu_fast_approximate(
    x: impl AsRef<Array>,
    stream: impl AsRef<crate::Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    x.as_ref().multiply(
        sigmoid(&array!(1.773).multiply(x.as_ref(), stream)?, stream)?,
        stream,
    )
}

/// Applies the gated linear unit function.
///
/// This function splits the `axis` dimension of the input into two halves
/// (`a` and `b`) and applies `a * sigmoid(b)`.
pub fn glu(
    x: impl AsRef<Array>,
    axis: impl Into<Option<i32>>,
    stream: impl AsRef<crate::Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    let split = x.as_ref().split(2, axis, stream)?;
    let (a, b) = (&split[0], &split[1]);
    a.multiply(sigmoid(b, stream)?, stream)
}

/// Applies the Step Activation Function.
///
/// This function implements a binary step activation, where the output is set
/// to 1 if the input is greater than a specified threshold, and 0 otherwise.
///
/// This is:
///
/// ```rust, ignore
/// r#where(x.gt(threshold), 1, 0)
/// ```
pub fn step(
    x: impl AsRef<Array>,
    threshold: impl Into<Option<f32>>,
    stream: impl AsRef<crate::Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    let threshold = array!(threshold.into().unwrap_or(0.0));
    crate::ops::r#where(
        &x.as_ref().gt(threshold, stream)?,
        &array!(1),
        &array!(0),
        stream,
    )
}

/// Applies the Scaled Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// elu(x, 1.67326) * 1.0507
/// ```
pub fn selu(x: impl AsRef<Array>, stream: impl AsRef<crate::Stream>) -> Result<Array> {
    let stream = stream.as_ref();
    elu(x.as_ref(), 1.67326, stream)?.multiply(array!(1.0507), stream)
}

/// Applies the element-wise parametric ReLU.
///
/// This is:
///
/// ```rust, ignore
/// maximum(0, x) + alpha * minimum(0, x)
/// ```
pub fn prelu(
    x: impl AsRef<Array>,
    alpha: impl AsRef<Array>,
    stream: impl AsRef<crate::Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    maximum(&array!(0.0), x.as_ref(), stream)?.add(
        alpha
            .as_ref()
            .multiply(minimum(&array!(0.0), x.as_ref(), stream)?, stream)?,
        stream,
    )
}

/// Applies the Mish function, element-wise.
///
/// Mish: A Self Regularized Non-Monotonic Neural Activation Function.
///
/// Reference: [https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)
///
/// This is:
///
/// ```rust, ignore
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// x * tanh(softplus(x, &stream))
/// ```
pub fn mish(x: impl AsRef<Array>, stream: impl AsRef<crate::Stream>) -> Result<Array> {
    let stream = stream.as_ref();
    x.as_ref()
        .multiply(tanh(&softplus(x.as_ref(), stream)?, stream)?, stream)
}

/// Applies the hardswish function, element-wise.
///
/// This is:
///
/// ```rust, ignore
/// x * minimum(maximum(x + 3, 0), 6) / 6
/// ```
pub fn hard_swish(x: impl AsRef<Array>, stream: impl AsRef<crate::Stream>) -> Result<Array> {
    let stream = stream.as_ref();
    let max_x_plus_3 = maximum(&x.as_ref().add(array!(3.0), stream)?, &array!(0.0), stream)?;
    x.as_ref()
        .multiply(minimum(&max_x_plus_3, &array!(6.0), stream)?, stream)?
        .divide(&array!(6.0), stream)
}

generate_builder! {
    /// Applies the gated linear unit function.
    ///
    /// This splits the `axis` dimension of the input into two halves
    /// (`a` and `b`) and applies `a * sigmoid(b)`.
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct Glu {
        /// The axis to split the input tensor. Default to [`Glu::DEFAULT_AXIS`] if not provided.
        #[builder(optional, default = Glu::DEFAULT_AXIS)]
        pub axis: i32,
    }
}

impl Glu {
    /// The default axis value.
    pub const DEFAULT_AXIS: i32 = -1;
}

impl Module<&Array> for Glu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        glu(x, self.axis, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the element-wise logistic sigmoid.
///
/// For details, please see
/// [this documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.sigmoid.html)
///
/// This is:
///
/// ```rust, ignore
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// sigmoid(x, &stream)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Sigmoid;

impl Module<&Array> for Sigmoid {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        sigmoid(x, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Mish function, element-wise.
///
/// Mish: A Self Regularized Non-Monotonic Neural Activation Function.
///
/// Reference: [https://arxiv.org/abs/1908.08681](https://arxiv.org/abs/1908.08681)
///
/// This is:
///
/// ```rust, ignore
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// x * tanh(softplus(x, &stream))
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Mish;

impl Module<&Array> for Mish {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        mish(x, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Rectified Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// maximum(x, 0)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Relu;

impl Module<&Array> for Relu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        relu(x, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

generate_builder! {
    /// Applies the Leaky Rectified Linear Unit.
    ///
    /// This is:
    ///
    /// ```rust, ignore
    /// maximum(neg_slope * x, x)
    /// ```
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct LeakyRelu {
        /// The negative slope. Default to [`LeakyReLU::DEFAULT_NEG_SLOPE`] if not provided.
        #[builder(optional, default = LeakyRelu::DEFAULT_NEG_SLOPE)]
        pub neg_slope: f32,
    }
}

impl LeakyRelu {
    /// The default negative slope value.
    pub const DEFAULT_NEG_SLOPE: f32 = 0.01;
}

impl Module<&Array> for LeakyRelu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        leaky_relu(x, self.neg_slope, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Rectified Linear Unit 6.
///
/// This is:
///
/// ```rust, ignore
/// minimum(&maximum(x, 0).unwrap(), 6).unwrap()
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Relu6;

impl Module<&Array> for Relu6 {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        relu6(x, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

generate_builder! {
    /// Applies the Softmax function.
    ///
    /// This is:
    ///
    /// ```rust, ignore
    /// softmax(&x, None, None)
    /// ```
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct Softmax {
        /// The axis to apply the softmax.
        #[builder(optional, default = Softmax::DEFAULT_AXIS)]
        pub axis: i32,
    }
}

impl Softmax {
    /// The default axis value.
    pub const DEFAULT_AXIS: i32 = -1;
}

impl Module<&Array> for Softmax {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        crate::ops::softmax_axis(x, self.axis, None, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Softplus function.
///
/// This is:
///
/// ```rust, ignore
/// logaddexp(x, 0)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Softplus;

impl Module<&Array> for Softplus {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        softplus(x, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Softsign function.
///
/// This is:
///
/// ```rust, ignore
/// x / (array!(1) + abs(x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Softsign;

impl Module<&Array> for Softsign {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        softsign(x, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

generate_builder! {
    /// Applies the Continuously Differentiable Exponential Linear Unit.
    ///
    /// This is:
    ///
    /// ```rust, ignore
    /// maximum(x, 0.0).unwrap()
    ///     + alpha * (exp(&(minimum(x, 0.0).unwrap() / alpha)) - 1)
    /// ```
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct Celu {
        /// The alpha value. Default to [`Celu::DEFAULT_ALPHA`] if not provided.
        #[builder(optional, default = Celu::DEFAULT_ALPHA)]
        pub alpha: f32,
    }
}

impl Celu {
    /// The default alpha value.
    pub const DEFAULT_ALPHA: f32 = 1.0;
}

impl Module<&Array> for Celu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        celu(x, self.alpha, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Sigmoid Linear Unit. Also known as Swish.
///
/// This is:
///
/// ```rust, ignore
/// # let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
/// x * sigmoid(x, &stream)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Silu;

impl Module<&Array> for Silu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        silu(x, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

generate_builder! {
    /// Applies the Log Softmax function.
    ///
    /// This is:
    ///
    /// ```rust, ignore
    /// x - logsumexp(x, axis, true)
    /// ```
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct LogSoftmax {
        /// The axis value. Default to [`LogSoftmax::DEFAULT_AXIS`] if not provided.
        #[builder(optional, default = LogSoftmax::DEFAULT_AXIS)]
        pub axis: i32,
    }
}

impl LogSoftmax {
    /// The default axis value.
    pub const DEFAULT_AXIS: i32 = -1;
}

impl Module<&Array> for LogSoftmax {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        log_softmax(x, self.axis, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Log Sigmoid function.
///
/// This is:
///
/// ```rust, ignore
/// -softplus(-x)
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct LogSigmoid;

impl Module<&Array> for LogSigmoid {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        log_sigmoid(x, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the element-wise parametric ReLU.
///
/// This is:
///
/// ```rust, ignore
/// maximum(0, x) + alpha * minimum(0, x)
/// ```
#[derive(Debug, Clone, ModuleParameters, Buildable)]
#[module(root = crate)]
#[buildable(root = crate)]
pub struct Prelu {
    /// The alpha value. See [`prelu`] for more details.
    #[param]
    #[builder(ignore)]
    pub weight: Param<Array>, // TODO: double check if this is trainable
}

/// The builder for the Prelu module.
#[derive(Debug, Clone, Builder)]
#[builder(
    root = crate,
    build_with = build_prelu,
    default_infallible,
    err = Exception,
)]
pub struct PreluBuilder {
    /// The count. Default to [`Prelu::DEFAULT_COUNT`] if not provided.
    #[builder(optional, default = Prelu::DEFAULT_COUNT)]
    pub count: i32,

    /// The value. Default to [`Prelu::DEFAULT_VALUE`] if not provided.
    #[builder(optional, default = Prelu::DEFAULT_VALUE)]
    pub value: f32,
}

/// Builds the Prelu module.
fn build_prelu(builder: PreluBuilder) -> Result<Prelu> {
    let count = builder.count;
    let value = builder.value;
    let values = vec![value; count as usize];
    let weight = Param::new(Array::from_slice(&values, &[count]));
    Ok(Prelu { weight })
}

impl Prelu {
    /// The default count value.
    pub const DEFAULT_COUNT: i32 = 1;

    /// The default value.
    pub const DEFAULT_VALUE: f32 = 0.25;
}

impl Module<&Array> for Prelu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        prelu(x, &self.weight, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Variants of Gaussian Error Linear Units function.
#[derive(Debug, Clone, Copy, Default)]
pub enum GeluApprox {
    /// Uses [`gelu`]
    #[default]
    None,

    /// Uses [`gelu_approximate`]
    Precise,

    /// Uses [`gelu_fast_approximate`]
    Fast,
}

generate_builder! {
    /// Applies the Gaussian Error Linear Units function.
    ///
    /// There are three variants:
    ///
    /// - `GeluApprox::None`: Uses [`gelu`]. This is the default.
    /// - `GeluApprox::Precise`: Uses [`gelu_approximate`]
    /// - `GeluApprox::Fast`: Uses [`gelu_fast_approximate`]
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct Gelu {
        /// The approximation to use. Default to `GeluApprox::None` if not provided.
        #[builder(optional, default = GeluApprox::None)]
        pub approximate: GeluApprox,
    }
}

impl Module<&Array> for Gelu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        match self.approximate {
            GeluApprox::None => gelu(x, stream),
            GeluApprox::Precise => gelu_approximate(x, stream),
            GeluApprox::Fast => gelu_fast_approximate(x, stream),
        }
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the hyperbolic tangent function
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Tanh;

impl Module<&Array> for Tanh {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        crate::ops::tanh(x, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the hardswish function, element-wise
///
/// This is:
///
/// ```rust, ignore
/// x * minimum(maximum(x + 3, 0), 6) / 6
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct HardSwish;

impl Module<&Array> for HardSwish {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        hard_swish(x, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

generate_builder! {
    /// Applies the Step Activation Function.
    ///
    /// This function implements a binary step activation, where the output is set
    /// to 1 if the input is greater than a specified threshold, and 0 otherwise.
    ///
    /// This is:
    ///
    /// ```rust, ignore
    /// r#where(x.gt(threshold), 1, 0)
    /// ```
    #[derive(Debug, Clone, ModuleParameters, Buildable)]
    #[module(root = crate)]
    #[buildable(root = crate)]
    #[builder(root = crate)]
    pub struct Step {
        /// The threshold value. Default to [`Step::DEFAULT_THRESHOLD`] if not provided.
        #[builder(optional, default = Step::DEFAULT_THRESHOLD)]
        pub threshold: f32,
    }
}

impl Step {
    /// The default threshold value.
    pub const DEFAULT_THRESHOLD: f32 = 0.0;
}

impl Module<&Array> for Step {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        step(x, self.threshold, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

/// Applies the Scaled Exponential Linear Unit.
///
/// This is:
///
/// ```rust, ignore
/// elu(x, 1.67326) * 1.0507
/// ```
#[derive(Debug, Clone, ModuleParameters)]
#[module(root = crate)]
pub struct Selu;

impl Module<&Array> for Selu {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &crate::Stream) -> Result<Array> {
        selu(x, stream)
    }

    fn training_mode(&mut self, _: bool) {}
}

// The following tests are ported from the swift binding:
// mlx-swift/Tests/MLXTests/IntegrationTests.swift
#[cfg(test)]
mod tests {
    use crate::{builder::Builder, random::uniform, Dtype};
    use float_eq::assert_float_eq;

    use super::*;

    #[test]
    fn test_glu() {
        let stream = crate::test_stream();
        let key = crate::test_key(850, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.547_252_66,
            abs <= 0.010_945_053
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            140.096_68,
            abs <= 2.801_933_5
        );
        let result = Glu::new().forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 8]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.333_276_75,
            abs <= 0.006_665_535
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            42.659_424,
            abs <= 0.853_188_46
        );
    }

    #[test]
    fn test_sigmoid() {
        let stream = crate::test_stream();
        let key = crate::test_key(589, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.529_697_9,
            abs <= 0.010_593_958
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            135.602_66,
            abs <= 2.712_053_3
        );
        let result = Sigmoid.forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.627_014,
            abs <= 0.012_540_28
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            160.515_58,
            abs <= 3.210_311_7
        );
    }

    #[test]
    fn test_mish() {
        let stream = crate::test_stream();
        let key = crate::test_key(122, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.501_719_8,
            abs <= 0.010_034_395
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            128.440_26,
            abs <= 2.568_805_2
        );
        let result = Mish.forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.395_375_73,
            abs <= 0.007_907_514
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            101.216_19,
            abs <= 2.024_323_7
        );
    }

    #[test]
    fn test_relu() {
        let stream = crate::test_stream();
        let key = crate::test_key(400, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.478_322_74,
            abs <= 0.009_566_455
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            122.450_62,
            abs <= 2.449_012_5
        );
        let result = Relu.forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.478_322_74,
            abs <= 0.009_566_455
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            122.450_62,
            abs <= 2.449_012_5
        );
    }

    #[test]
    fn test_leaky_relu() {
        let stream = crate::test_stream();
        let key = crate::test_key(93, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.499_930_68,
            abs <= 0.009_998_614
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            127.982_254,
            abs <= 2.559_645_2
        );
        let result = LeakyRelu::new().forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.499_930_68,
            abs <= 0.009_998_614
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            127.982_254,
            abs <= 2.559_645_2
        );
    }

    #[test]
    fn test_relu6() {
        let stream = crate::test_stream();
        let key = crate::test_key(379, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.493_258_66,
            abs <= 0.009_865_173
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            126.274_216,
            abs <= 2.525_484_3
        );
        let result = Relu6.forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.493_258_66,
            abs <= 0.009_865_173
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            126.274_216,
            abs <= 2.525_484_3
        );
    }

    #[test]
    fn test_softmax() {
        let stream = crate::test_stream();
        let key = crate::test_key(853, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.514_396_3,
            abs <= 0.010_287_926_5
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            131.685_46,
            abs <= 2.633_709_2
        );
        let result = Softmax::new().forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.062_499_996,
            abs <= 0.001_25
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            15.999_999,
            abs <= 0.32
        );
    }

    #[test]
    fn test_softplus() {
        let stream = crate::test_stream();
        let key = crate::test_key(118, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.498_981_42,
            abs <= 0.009_979_628
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            127.739_24,
            abs <= 2.554_784_8
        );
        let result = Softplus.forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.982_857_76,
            abs <= 0.019_657_155
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            251.611_59,
            abs <= 5.032_232
        );
    }

    #[test]
    fn test_softsign() {
        let stream = crate::test_stream();
        let key = crate::test_key(37, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.506_551_27,
            abs <= 0.010_131_026
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            129.677_12,
            abs <= 2.593_542_6
        );
        let result = Softsign.forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.314_089_83,
            abs <= 0.006_281_797
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            80.407,
            abs <= 1.608_14
        );
    }

    // The unit test below is adapted from the python binding:
    // mlx/python/tests/test_nn.py
    #[test]
    fn test_celu() {
        let stream = crate::test_stream();
        let x = array!([1.0, -1.0, 0.0]);
        let y = Celu::new().forward(&x, stream).unwrap();
        let epsilon = array!(1e-4);
        let expected_y = array!([1.0, -0.6321, 0.0]);
        assert!(y
            .subtract(&expected_y, stream)
            .unwrap()
            .abs(stream)
            .unwrap()
            .lt(&epsilon, stream)
            .unwrap()
            .all(None, stream)
            .unwrap()
            .item::<bool>(&stream));
        assert_eq!(y.shape(), &[3]);
        assert_eq!(y.dtype(), Dtype::Float32);

        let y = CeluBuilder::new()
            .alpha(1.1)
            .build()
            .unwrap()
            .forward(&x, stream)
            .unwrap();
        let expected_y = array!([1.0, -0.6568, 0.0]);
        assert!(y
            .subtract(&expected_y, stream)
            .unwrap()
            .abs(stream)
            .unwrap()
            .lt(&epsilon, stream)
            .unwrap()
            .all(None, stream)
            .unwrap()
            .item::<bool>(&stream));
        assert_eq!(y.shape(), &[3]);
        assert_eq!(y.dtype(), Dtype::Float32);
    }

    #[test]
    fn test_silu() {
        let stream = crate::test_stream();
        let key = crate::test_key(22, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.502_970_6,
            abs <= 0.010_059_412
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            128.760_47,
            abs <= 2.575_209_4
        );
        let result = Silu.forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.331_970_93,
            abs <= 0.006_639_418_7
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            84.984_56,
            abs <= 1.699_691_2
        );
    }

    #[test]
    fn test_log_softmax() {
        let stream = crate::test_stream();
        let key = crate::test_key(199, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.527_843_7,
            abs <= 0.010_556_874
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            135.127_99,
            abs <= 2.702_559_7
        );
        let result = LogSoftmax::new().forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            -2.810_954_6,
            abs <= 0.056_219_09
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            -719.604_4,
            abs <= 14.392_087
        );
    }

    #[test]
    fn test_log_sigmoid() {
        let stream = crate::test_stream();
        let key = crate::test_key(984, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.510_977_7,
            abs <= 0.010_219_553_5
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            130.810_29,
            abs <= 2.616_205_7
        );
        let result = LogSigmoid.forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            -0.479_598_55,
            abs <= 0.009_591_971
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            -122.777_23,
            abs <= 2.455_544_5
        );
    }

    #[test]
    fn test_prelu() {
        let stream = crate::test_stream();
        let key = crate::test_key(993, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.496_651_44,
            abs <= 0.009_933_028
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            127.142_77,
            abs <= 2.542_855_3
        );
        let result = Prelu::new().forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.496_651_44,
            abs <= 0.009_933_028
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            127.142_77,
            abs <= 2.542_855_3
        );
    }

    #[test]
    fn test_gelu() {
        let stream = crate::test_stream();
        let key = crate::test_key(189, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.492_950_32,
            abs <= 0.009_859_007
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            126.195_28,
            abs <= 2.523_905_8
        );
        let result = Gelu::new().forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.365_638_38,
            abs <= 0.007_312_767_7
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            93.603_424,
            abs <= 1.872_068_5
        );
    }

    #[test]
    fn test_tanh() {
        let stream = crate::test_stream();
        let key = crate::test_key(735, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.474_122_7,
            abs <= 0.009_482_454_5
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            121.375_41,
            abs <= 2.427_508_4
        );
        let result = Tanh.forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.413_079_68,
            abs <= 0.008_261_594
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            105.748_4,
            abs <= 2.114_968
        );
    }

    #[test]
    fn test_hardswish() {
        let stream = crate::test_stream();
        let key = crate::test_key(126, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.491_892_46,
            abs <= 0.009_837_849
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            125.924_47,
            abs <= 2.518_489_4
        );
        let result = HardSwish.forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.299_602_24,
            abs <= 0.005_992_044_7
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            76.698_17,
            abs <= 1.533_963_4
        );
    }

    #[test]
    fn test_step() {
        let stream = crate::test_stream();
        let key = crate::test_key(490, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.479_360_64,
            abs <= 0.009_587_212_5
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            122.716_324,
            abs <= 2.454_326_4
        );
        let result = Step::new().forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Int32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            1.0,
            abs <= 0.02
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            256.0,
            abs <= 5.12
        );
    }

    #[test]
    fn test_selu() {
        let stream = crate::test_stream();
        let key = crate::test_key(215, stream);
        let a = uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], &key, stream).unwrap();
        assert_eq!(a.shape(), &[2, 8, 16]);
        assert_eq!(a.dtype(), Dtype::Float32);
        assert_float_eq!(
            a.mean(None, stream).unwrap().item::<f32>(&stream),
            0.493_026_8,
            abs <= 0.009_860_536
        );
        assert_float_eq!(
            a.sum(None, stream).unwrap().item::<f32>(&stream),
            126.214_86,
            abs <= 2.524_297_2
        );
        let result = Selu.forward(&a, stream).unwrap();
        assert_eq!(result.shape(), &[2, 8, 16]);
        assert_eq!(result.dtype(), Dtype::Float32);
        assert_float_eq!(
            result.mean(None, stream).unwrap().item::<f32>(&stream),
            0.518_023_2,
            abs <= 0.010_360_463_5
        );
        assert_float_eq!(
            result.sum(None, stream).unwrap().item::<f32>(&stream),
            132.613_94,
            abs <= 2.652_278_7
        );
    }
}
