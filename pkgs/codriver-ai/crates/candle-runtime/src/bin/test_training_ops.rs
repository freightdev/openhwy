use candle_core::{Device, Tensor, Result, DType};
use std::time::Instant;

fn benchmark_matmul(device: &Device, size: usize, iterations: usize) -> Result<f64> {
    let a = Tensor::randn(0f32, 1.0, (size, size), device)?.to_dtype(DType::F16)?;
    let b = Tensor::randn(0f32, 1.0, (size, size), device)?.to_dtype(DType::F16)?;
    
    device.synchronize()?;
    let start = Instant::now();
    
    for _ in 0..iterations {
        let _c = a.matmul(&b)?;
    }
    
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() / iterations as f64)
}

fn benchmark_backward_pass(device: &Device, batch: usize, features: usize) -> Result<f64> {
    let x = Tensor::randn(0f32, 1.0, (batch, features), device)?.to_dtype(DType::F32)?;
    let w = Tensor::randn(0f32, 1.0, (features, features), device)?.to_dtype(DType::F32)?;
    
    device.synchronize()?;
    let start = Instant::now();
    
    // Simulate forward + backward
    for _ in 0..10 {
        let out = x.matmul(&w)?;
        let loss = out.sum_all()?;
        // In real training, you'd call backward here
        drop(loss);
    }
    
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() / 10.0)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("🏋️  Training Operations Benchmark\n");
    
    // Matrix multiplication (backbone of training)
    println!("1️⃣  Matrix Multiplication (FP16):");
    for size in [512, 1024, 2048, 4096] {
        let time = benchmark_matmul(&device, size, 20)?;
        let tflops = 2.0 * size.pow(3) as f64 / time / 1e12;
        println!("   {}x{}: {:.2}ms ({:.2} TFLOPS)", size, size, time * 1000.0, tflops);
    }
    
    // Simulated training step
    println!("\n2️⃣  Simulated Training Steps:");
    for (batch, features) in [(32, 768), (64, 1024), (128, 2048)] {
        let time = benchmark_backward_pass(&device, batch, features)?;
        println!("   Batch {} x Features {}: {:.2}ms/step", batch, features, time * 1000.0);
    }
    
    // Activation functions - can't use closures in array, do them individually
    println!("\n3️⃣  Activation Functions:");
    let size = (1024, 4096);
    let x = Tensor::randn(0f32, 1.0, size, &device)?.to_dtype(DType::F16)?;
    
    // ReLU
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..100 {
        let _ = x.relu()?;
    }
    device.synchronize()?;
    let relu_time = start.elapsed().as_secs_f64() / 100.0;
    println!("   ReLU: {:.3}ms", relu_time * 1000.0);
    
    // GELU
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..100 {
        let _ = x.gelu()?;
    }
    device.synchronize()?;
    let gelu_time = start.elapsed().as_secs_f64() / 100.0;
    println!("   GELU: {:.3}ms", gelu_time * 1000.0);
    
    // Tanh
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..100 {
        let _ = x.tanh()?;
    }
    device.synchronize()?;
    let tanh_time = start.elapsed().as_secs_f64() / 100.0;
    println!("   Tanh: {:.3}ms", tanh_time * 1000.0);
    
    // Reduction operations (common in loss computation)
    println!("\n4️⃣  Reduction Operations:");
    let large = Tensor::randn(0f32, 1.0, (2048, 2048), &device)?.to_dtype(DType::F32)?;
    
    // Sum
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..100 {
        let _ = large.sum_all()?;
    }
    device.synchronize()?;
    let sum_time = start.elapsed().as_secs_f64() / 100.0;
    println!("   Sum: {:.3}ms", sum_time * 1000.0);
    
    // Mean
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..100 {
        let _ = large.mean_all()?;
    }
    device.synchronize()?;
    let mean_time = start.elapsed().as_secs_f64() / 100.0;
    println!("   Mean: {:.3}ms", mean_time * 1000.0);
    
    println!("\n✅ Training ops benchmark complete");
    Ok(())
}
