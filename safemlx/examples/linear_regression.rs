use safemlx::error::Exception;
use safemlx::random::RandomState;
use safemlx::{array, ops, transforms, Array};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let mut rng = RandomState::with_seed(0)?;
    let num_features: i32 = 100;
    let num_examples: i32 = 1000;
    let num_iterations: i32 = 10000;
    let learning_rate: f32 = 0.01;

    // True weight vector
    let w_star = safemlx::normal!(
        shape = &[num_features],
        key = &rng.next_key(&stream)?,
        stream = &stream
    )?;

    // Input examples (design matrix)
    let x = safemlx::normal!(
        shape = &[num_examples, num_features],
        key = &rng.next_key(&stream)?,
        stream = &stream
    )?;

    // Noisy labels
    let eps = safemlx::normal!(
        shape = &[num_examples],
        key = &rng.next_key(&stream)?,
        stream = &stream
    )?
    .multiply(array!(1e-2f32), &stream)?;
    let y = x.matmul(&w_star, &stream)?.add(&eps, &stream)?;

    // Initialize random weights
    let w = safemlx::normal!(
        shape = &[num_features],
        key = &rng.next_key(&stream)?,
        stream = &stream
    )?
    .multiply(array!(1e-2f32), &stream)?;

    let loss_fn = |inputs: &[Array]| -> Result<Array, Exception> {
        let w = &inputs[0];
        let x = &inputs[1];
        let y = &inputs[2];

        let y_pred = x.matmul(w, &stream)?;
        let residual = y_pred.subtract(y, &stream)?;
        let mean_square = ops::mean(&ops::square(&residual, &stream)?, None, &stream)?;
        let loss = Array::from_f32(0.5).multiply(&mean_square, &stream)?;
        Ok(loss)
    };

    let mut grad_fn = transforms::grad(loss_fn);

    let now = std::time::Instant::now();
    let mut inputs = [w, x, y];

    for _ in 0..num_iterations {
        let grad = grad_fn(&inputs)?;
        let update = Array::from_f32(learning_rate).multiply(&grad, &stream)?;
        inputs[0] = inputs[0].subtract(&update, &stream)?;
        transforms::eval([&inputs[0]])?;
    }

    let elapsed = now.elapsed();

    let loss = loss_fn(&inputs)?;
    let error = inputs[0].subtract(&w_star, &stream)?;
    let error_norm = ops::sum(&ops::square(&error, &stream)?, None, &stream)?.sqrt(&stream)?;
    let throughput = num_iterations as f32 / elapsed.as_secs_f32();

    println!(
        "Loss {:.5}, L2 distance: |w-w*| = {:.5}, Throughput {:.5} (it/s)",
        loss.item::<f32>(&stream),
        error_norm.item::<f32>(&stream),
        throughput
    );

    Ok(())
}
