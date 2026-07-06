/// Asserts that two arrays are equal.
///
/// It checks that the two arrays have the same shape and that all elements are
/// sufficiently close.
#[macro_export]
macro_rules! assert_array_eq {
    ($value:expr, $expected:expr, stream = $stream:expr) => {
        assert_array_eq!($value, $expected, None, stream = $stream);
    };
    ($value:expr, $expected:expr, $atol:expr, stream = $stream:expr) => {
        assert_eq!($value.shape(), $expected.shape(), "Shapes are not equal");
        let assert = $value.all_close(&$expected, $atol, $atol, None, $stream);
        assert!(
            assert.unwrap().item::<bool>($stream),
            "Values are not sufficiently close"
        );
    };
    ($($tokens:tt)*) => {
        compile_error!("assert_array_eq! requires an explicit `stream = ...` argument");
    };
}
