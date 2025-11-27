use candle_core::{Device, Tensor, Result, DType};
use std::time::Instant;

fn benchmark_matmul(size: usize, dtype: DType, device: &Device) -> Result<f64> {
    let a = Tensor::randn(0f32, 1.0, (size, size), device)?.to_dtype(dtype)?;
    let b = Tensor::randn(0f32, 1.0, (size, size), device)?.to_dtype(dtype)?;
    
    device.synchronize()?;
    
    let iterations = 100;
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _c = a.matmul(&b)?;
    }
    
    device.synchronize()?;
    let elapsed = start.elapsed().as_secs_f64();
    
    // FLOPS calculation: 2 * N^3 operations
    let ops = 2.0 * size as f64 * size as f64 * size as f64 * iterations as f64;
    let tflops = ops / elapsed / 1e12;
    
    Ok(tflops)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("⚡ Matrix Multiplication Performance\n");
    
    let sizes = vec![1024, 2048, 4096, 8192];
    
    println!("Testing FP32 (standard precision):");
    for size in &sizes {
        let tflops = benchmark_matmul(*size, DType::F32, &device)?;
        println!("  {}x{}: {:.2} TFLOPS", size, size, tflops);
    }
    
    println!("\nTesting FP16 (half precision):");
    for size in &sizes {
        let tflops = benchmark_matmul(*size, DType::F16, &device)?;
        println!("  {}x{}: {:.2} TFLOPS", size, size, tflops);
    }
    
    println!("\n📊 GTX 1650 Mobile Theoretical Max:");
    println!("   FP32: ~3.0 TFLOPS");
    println!("   FP16: ~6.0 TFLOPS (with tensor cores if available)");
    
    Ok(())
}
