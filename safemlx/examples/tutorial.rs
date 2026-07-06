use safemlx::transforms::grad;
use safemlx::{Array, Dtype, Stream};

fn scalar_basics() {
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    // create a scalar array
    let x: Array = 1.0.into();

    // the datatype is .float32
    let dtype = x.dtype();
    assert_eq!(dtype, Dtype::Float32);

    // scalars have a size of 1
    let size = x.size();
    assert_eq!(size, 1);

    // scalars have 0 dimensions
    let ndim = x.ndim();
    assert_eq!(ndim, 0);

    // scalar shapes are empty arrays
    let shape = x.shape();
    assert!(shape.is_empty());

    // get the value
    let s = x.item::<f32>(&stream);
    assert_eq!(s, 1.0);

    // reading the value with a different type is a fatal error
    // let i = x.item::<i32>(&stream);
}

#[allow(unused_variables)]
fn array_basics() {
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));

    // make a multidimensional array.
    let x: Array = Array::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // mlx is row-major by default so the first row of this array
    // is [1.0, 2.0] and the second row is [3.0, 4.0]

    // Make an array of shape {2, 2} filled with ones:
    let y = Array::ones::<f32>(&[2, 2], &stream).unwrap();

    // Pointwise add x and y:
    let z = x.add(&y, &stream);

    let z = z.unwrap();

    // mlx is lazy by default. At this point `z` only
    // has a shape and a type but no actual data:
    assert_eq!(z.dtype(), Dtype::Float32);
    assert_eq!(z.shape(), vec![2, 2]);

    // To actually run the computation you must evaluate `z`.
    // Under the hood, mlx records operations in a graph.
    // The variable `z` is a node in the graph which points to its operation
    // and inputs. Calling `evaluated` borrows the array, evaluates it and all
    // of its dependencies, then returns a host-readable evaluated view.
    let z = z.evaluated().unwrap();

    let z = z
        .as_array()
        .add(&y, &stream)
        .unwrap()
        .into_evaluated()
        .unwrap();
    assert_eq!(z.as_slice::<f32>(), &[3.0, 4.0, 5.0, 6.0]);

    // Scalar host readback consumes the array:
    let z = Array::ones::<f32>(&[1], &stream).unwrap();
    z.item::<f32>(&stream);

    let z = Array::ones::<f32>(&[2, 2], &stream).unwrap();
    println!("{z}");
}

fn automatic_differentiation() {
    use safemlx::error::Result;

    fn f(x: &Array, stream: &Stream) -> Result<Array> {
        x.square(stream)
    }

    fn calculate_grad(func: impl Fn(&Array) -> Result<Array>, arg: &Array) -> Result<Array> {
        grad(&func)(arg)
    }

    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let x = Array::from(1.5);

    let dfdx = calculate_grad(|x| f(x, &stream), &x).unwrap();
    assert_eq!(dfdx.item::<f32>(&stream), 2.0 * 1.5);

    let dfdx2 = calculate_grad(|args| calculate_grad(|x| f(x, &stream), args), &x).unwrap();
    assert_eq!(dfdx2.item::<f32>(&stream), 2.0);
}

fn main() {
    scalar_basics();
    array_basics();
    automatic_differentiation();
}
