#![allow(dead_code)]

use safemlx::{
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    random::uniform,
    utils::IntoOption,
    Array, ArrayElement, Stream,
};

pub fn test_stream() -> &'static Stream {
    Box::leak(Box::new(safemlx::Stream::new_with_device(
        &safemlx::Device::new(safemlx::DeviceType::Cpu, 0),
    )))
}

pub fn eval_vec<T>(array: &Array) -> Vec<T>
where
    T: ArrayElement + Clone,
{
    array.evaluated().unwrap().as_slice::<T>().to_vec()
}

pub fn eval_equal_values(lhs: &Array, rhs: &Array) -> bool {
    let lhs = lhs.evaluated().unwrap();
    let rhs = rhs.evaluated().unwrap();
    lhs.equal_values(&rhs)
}

/// A helper model for testing optimizers.
///
/// This is adapted from the swift binding tests in `mlx-swift/Tests/MLXTests/OptimizerTests.swift`.
#[derive(Debug, ModuleParameters)]
pub struct LinearFunctionModel {
    #[param]
    pub m: Param<Array>,

    #[param]
    pub b: Param<Array>,
}

impl Module<&Array> for LinearFunctionModel {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Self::Error> {
        self.m.multiply(x, stream)?.add(&self.b, stream)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

impl LinearFunctionModel {
    pub fn new<'a>(
        shape: impl IntoOption<&'a [i32]>,
        stream: &Stream,
    ) -> safemlx::error::Result<Self> {
        let shape = shape.into_option();
        let m_key = safemlx::random::key(0)?;
        let b_key = safemlx::random::key(1)?;
        let m = uniform::<_, f32>(-5.0, 5.0, shape, &m_key, stream)?;
        let b = uniform::<_, f32>(-5.0, 5.0, shape, &b_key, stream)?;
        Ok(Self {
            m: Param::new(m),
            b: Param::new(b),
        })
    }
}
