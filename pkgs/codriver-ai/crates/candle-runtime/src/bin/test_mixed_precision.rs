use candle_core::{Device, Tensor, Result, DType};
use std::time::Instant;

fn benchmark_dtype(
    device: &Device,
    size: usize,
    dtype: DType,
    _name: &str,
) -> Result<(f64, f64)> {
    let a = Tensor::randn(0f32, 1.0, (size, size), device)?.to_dtype(dtype)?;
    let b = Tensor::randn(0f32, 1.0, (size, size), device)?.to_dtype(dtype)?;
    
    device.synchronize()?;
    
    // Forward pass
    let start = Instant::now();
    for _ in 0..20 {
        let _c = a.matmul(&b)?;
    }
    device.synchronize()?;
    let forward_time = start.elapsed().as_secs_f64() / 20.0;
    
    // Memory usage
    let memory_mb = size * size * dtype.size_in_bytes() * 2 / 1024 / 1024;
    
    Ok((forward_time, memory_mb as f64))
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("🔀 Mixed Precision Training Test\n");
    
    // Test 1: Precision comparison
    println!("1️⃣  Precision Comparison (2048x2048 matmul):");
    
    for (dtype, name) in [
        (DType::F32, "FP32"),
        (DType::F16, "FP16"),
        (DType::BF16, "BF16"),
    ] {
        let (time, memory) = benchmark_dtype(&device, 2048, dtype, name)?;
        let tflops = 2.0 * 2048_f64.powi(3) / time / 1e12;
        println!("   {}: {:.2}ms, {:.2} TFLOPS, {} MB", 
            name, time * 1000.0, tflops, memory as usize);
    }
    
    // Test 2: Loss scaling simulation
    println!("\n2️⃣  Loss Scaling (prevents gradient underflow):");
    let scale_factors = [1.0, 128.0, 512.0, 1024.0, 2048.0];
    
    for scale in scale_factors {
        let small_grad = 1e-7;
        let scaled_grad = small_grad * scale;
        
        // FP16 min: ~6e-5
        let representable = scaled_grad > 6e-5;
        
        println!("   Scale {}: grad {:.2e} -> {:.2e} {}",
            scale,
            small_grad,
            scaled_grad,
            if representable { "✅" } else { "❌ underflow" });
    }
    
    // Test 3: Memory savings
    println!("\n3️⃣  Memory Savings (7B model):");
    let params_7b: i64 = 7_000_000_000;
    
    println!("   FP32: {} GB", params_7b * 4 / 1024 / 1024 / 1024);
    println!("   FP16: {} GB", params_7b * 2 / 1024 / 1024 / 1024);
    println!("   BF16: {} GB", params_7b * 2 / 1024 / 1024 / 1024);

    println!("   FP32 + Adam: {} GB (params + 2 states)", params_7b * 4 * 3 / 1024 / 1024 / 1024);
    println!("   FP16 + FP32 Adam: {} GB (mixed)", 
        (params_7b * 2 + params_7b * 4 * 2) / 1024 / 1024 / 1024);
    println!("   FP16 + 8-bit Adam: {} GB ✅", 
        (params_7b * 2 + params_7b * 1 * 2) / 1024 / 1024 / 1024);    
    
    // Test 4: Automatic mixed precision strategy
    println!("\n4️⃣  AMP Strategy:");
    println!("   Forward pass: FP16 (fast, memory efficient)");
    println!("   Loss computation: FP32 (stable)");
    println!("   Backward pass: FP16 (fast)");
    println!("   Gradient accumulation: FP32 (accurate)");
    println!("   Optimizer states: FP32 or 8-bit (stable)");
    println!("   Master weights: FP32 (precision)");
    
    // Test 5: Dynamic loss scaling
    println!("\n5️⃣  Dynamic Loss Scaling:");
    println!("   Start scale: 2^16 = 65536");
    println!("   If overflow: scale /= 2");
    println!("   If no overflow for N steps: scale *= 2");
    println!("   Typical range: 512 - 32768");
    println!("   Adapts to training dynamics ✅");
    
    // Test 6: Numeric stability check
    println!("\n6️⃣  Numeric Range Comparison:");
    println!("   FP32: ±3.4e38 (8 bit exponent)");
    println!("   FP16: ±65504 (5 bit exponent) - narrow!");
    println!("   BF16: ±3.4e38 (8 bit exponent) - same as FP32");
    println!();
    println!("   💡 BF16 preferred for training (better range)");
    println!("   💡 FP16 preferred for inference (better precision)");
    
    println!("\n✅ Mixed precision tests complete");
    Ok(())
}
