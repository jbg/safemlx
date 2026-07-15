#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use safemlx::{Array, Device, DeviceType, ExecutionContext};

    if !safemlx::cuda::is_available()? {
        return Err("MLX was built with CUDA, but no CUDA device is available".into());
    }

    let gpu = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]);

    let gpu_result = a.matmul(&b, &gpu)?.into_evaluated()?;
    let cpu_result = a.matmul(&b, &cpu)?.into_evaluated()?;
    let gpu_values: &[f32] = gpu_result.as_slice();
    let cpu_values: &[f32] = cpu_result.as_slice();
    if gpu_values
        .iter()
        .zip(cpu_values)
        .any(|(gpu, cpu)| (gpu - cpu).abs() > 1e-4)
    {
        return Err(format!("CUDA result {gpu_values:?} != CPU result {cpu_values:?}").into());
    }

    println!("CUDA smoke test passed: {gpu_values:?}");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("re-run with `--features cuda`");
    std::process::exit(2);
}
