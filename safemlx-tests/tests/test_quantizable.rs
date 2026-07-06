use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn::Linear,
    quantization::MaybeQuantized,
    Array, Stream,
};

#[derive(Debug, ModuleParameters, Quantizable)]
#[allow(dead_code)]
struct QuantizableExample {
    #[quantizable]
    pub ql: MaybeQuantized<Linear>,
}

impl Module<&Array> for QuantizableExample {
    type Output = Array;

    type Error = Exception;

    fn forward(&mut self, x: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        self.ql.forward(x, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.ql.training_mode(mode)
    }
}
