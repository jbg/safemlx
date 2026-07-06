use safemlx::{
    array,
    error::Exception,
    exp, negative,
    transforms::compile::{compile, disable_compile, enable_compile},
    Array,
};

mod common;

#[test]
fn test_disable_compile() {
    let stream = common::test_stream();
    disable_compile().unwrap();

    let f = move |x: &Array| -> Result<Array, Exception> {
        let z = negative!(x, stream = stream)?;

        // this will crash is compile is enabled
        println!("{z:?}");

        exp!(z, stream = stream)
    };

    let x = array!(10.0);
    let mut compiled = compile(f, None);

    // This will panic if compilation is enabled
    let _result = compiled(&x).unwrap();

    // Re-enable compilation for other tests
    enable_compile().unwrap();
}
