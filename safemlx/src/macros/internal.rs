/// See `assertEqual` in the swift binding tests
#[allow(unused_macros)]
macro_rules! assert_array_all_close {
    ($a:tt, $b:tt, stream = $stream:expr) => {
        let _b: Array = $b.into();
        let assert = $a.all_close(&_b, None, None, None, $stream).unwrap();
        assert!(assert.item::<bool>($stream));
    };
}

#[allow(unused_macros)]
macro_rules! cfg_safetensors {
    ($($item:item)*) => {
        $(
            #[cfg(feature = "safetensors")]
            $item
        )*
    };
}
