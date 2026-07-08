//! Lightweight activation inspection hooks.

use safemlx::{error::Exception, Array};

/// Receives named tensors from instrumented model forward passes.
///
/// Implementations should be selective about evaluating tensors: activations
/// can be large, and observing them does not force evaluation by itself.
pub trait ActivationObserver {
    /// Observe a named tensor.
    fn observe(&mut self, name: &str, value: &Array) -> Result<(), Exception>;
}

impl<F> ActivationObserver for F
where
    F: FnMut(&str, &Array) -> Result<(), Exception>,
{
    fn observe(&mut self, name: &str, value: &Array) -> Result<(), Exception> {
        self(name, value)
    }
}

/// Observer that ignores every tensor.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopObserver;

impl ActivationObserver for NoopObserver {
    fn observe(&mut self, _name: &str, _value: &Array) -> Result<(), Exception> {
        Ok(())
    }
}

/// A cloned activation captured by [`ActivationRecorder`].
#[derive(Debug, Clone)]
pub struct RecordedActivation {
    /// Stable path-like name of the tensor within the model forward pass.
    pub name: String,
    /// Lazy MLX array handle for the observed tensor.
    pub value: Array,
}

/// Simple observer that records cloned array handles.
#[derive(Debug, Default, Clone)]
pub struct ActivationRecorder {
    activations: Vec<RecordedActivation>,
}

impl ActivationRecorder {
    /// Creates an empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the recorded activations.
    pub fn activations(&self) -> &[RecordedActivation] {
        &self.activations
    }

    /// Consumes the recorder and returns the recorded activations.
    pub fn into_activations(self) -> Vec<RecordedActivation> {
        self.activations
    }

    /// Removes all recorded activations.
    pub fn clear(&mut self) {
        self.activations.clear();
    }
}

impl ActivationObserver for ActivationRecorder {
    fn observe(&mut self, name: &str, value: &Array) -> Result<(), Exception> {
        self.activations.push(RecordedActivation {
            name: name.to_string(),
            value: value.clone(),
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ActivationObserver, ActivationRecorder};
    use safemlx::Array;

    #[test]
    #[ignore = "requires MLX runtime execution"]
    fn recorder_clones_observed_array_handles() {
        let array = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let mut recorder = ActivationRecorder::new();

        recorder.observe("layer.output", &array).unwrap();

        let activations = recorder.activations();
        assert_eq!(activations.len(), 1);
        assert_eq!(activations[0].name, "layer.output");
        assert_eq!(activations[0].value.shape(), &[2]);
    }
}
