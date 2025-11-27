use candle_core::{Device, Tensor, Result, DType};
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("🧪 GPU Memory Stress Test\n");
    
    // Test 1: Maximum allocation
    println!("1️⃣  Finding maximum single allocation...");
    let mut max_size = 0;
    for size in (1024..16384).step_by(512) {
        match Tensor::zeros((size, size), DType::F32, &device) {
            Ok(_) => max_size = size,
            Err(_) => break,
        }
    }
    println!("   Max single tensor: {}x{} FP32 (~{} MB)\n", 
        max_size, max_size, (max_size * max_size * 4) / 1024 / 1024);
    
    // Test 2: Fragmentation test
    println!("2️⃣  Testing memory fragmentation...");
    let mut tensors = Vec::new();
    let chunk_size = 1024;
    
    loop {
        match Tensor::zeros((chunk_size, chunk_size), DType::F32, &device) {
            Ok(t) => tensors.push(t),
            Err(_) => break,
        }
    }
    
    let total_mb = (tensors.len() * chunk_size * chunk_size * 4) / 1024 / 1024;
    println!("   Allocated {} chunks = {} MB", tensors.len(), total_mb);
    println!("   Available for training: ~{} MB\n", total_mb - 500);
    
    // Test 3: Allocation/deallocation speed
    println!("3️⃣  Memory allocation performance...");
    drop(tensors);
    
    let start = Instant::now();
    for _ in 0..100 {
        let _t = Tensor::zeros((2048, 2048), DType::F32, &device)?;
    }
    device.synchronize()?;
    let alloc_time = start.elapsed().as_secs_f64() / 100.0;
    println!("   Average allocation: {:.3}ms", alloc_time * 1000.0);
    
    // Test 4: Different dtypes
    println!("\n4️⃣  Data type memory usage:");
    let size = 4096;
    
    for (dtype, name) in [
        (DType::F32, "FP32"),
        (DType::F16, "FP16"),
        (DType::BF16, "BF16"),
        (DType::U8, "INT8"),
    ] {
        let t = Tensor::zeros((size, size), dtype, &device)?;
        let bytes = size * size * dtype.size_in_bytes();
        println!("   {} {}x{}: {} MB", name, size, size, bytes / 1024 / 1024);
        drop(t);
    }
    
    println!("\n✅ Memory stress test complete");
    Ok(())
}
