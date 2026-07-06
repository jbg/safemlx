use safemlx::{
    error::Exception, macros::ModuleParameters, module::Module, nn::Linear, Array,
    ExecutionContext, Stream,
};

#[derive(Debug, ModuleParameters)]
struct M {
    #[param]
    linear: Linear,
}

impl M {
    pub fn new() -> Self {
        Self {
            linear: Linear::new(5, 5).unwrap(),
        }
    }
}

impl Module<&Array> for M {
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Array, Self::Error> {
        self.linear.forward(x, stream)
    }

    fn training_mode(&mut self, _mode: bool) {}
}

#[test]
fn test_nested_module() {
    let mut m = M::new();
    let ctx = ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let stream = ctx.stream();
    let key = safemlx::random::key(0).unwrap();
    let x = safemlx::random::uniform::<_, f32>(1.0, 2.0, &[1, 5], &key, stream).unwrap();
    let y = m.forward(&x, stream).unwrap();
    assert_ne!(y.sum(None, stream).unwrap().item::<f32>(stream), 0.0);
}
