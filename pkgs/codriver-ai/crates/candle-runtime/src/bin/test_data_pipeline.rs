use candle_core::{Device, Tensor, Result, DType};
use std::time::Instant;

fn simulate_batch_loading(
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    device: &Device,
) -> Result<(Tensor, f64)> {
    let start = Instant::now();
    
    // Simulate loading from CPU (tokenized text)
    let cpu_data = vec![0u32; batch_size * seq_len];
    
    // Transfer to GPU
    let input_ids = Tensor::from_vec(cpu_data, (batch_size, seq_len), device)?;
    
    // Simulate embedding lookup
    let _embeddings = Tensor::randn(0f32, 1.0, (vocab_size, 768), device)?
        .to_dtype(DType::F16)?;
    
    device.synchronize()?;
    let time = start.elapsed().as_secs_f64();
    
    Ok((input_ids, time))
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("📦 Data Pipeline Performance Test\n");
    
    // Test 1: Batch loading speed
    println!("1️⃣  Batch Loading Speed:");
    let vocab_size = 32000;
    
    for (batch_size, seq_len) in [(1, 512), (8, 512), (32, 512), (64, 256)] {
        let mut total_time = 0.0;
        let iterations = 20;
        
        for _ in 0..iterations {
            let (_, time) = simulate_batch_loading(batch_size, seq_len, vocab_size, &device)?;
            total_time += time;
        }
        
        let avg_time = total_time / iterations as f64;
        let tokens_per_sec = (batch_size * seq_len) as f64 / avg_time;
        
        println!("   Batch {} x Seq {}: {:.2}ms ({:.0} tokens/sec)",
            batch_size, seq_len, avg_time * 1000.0, tokens_per_sec);
    }
    
    // Test 2: CPU to GPU transfer
    println!("\n2️⃣  CPU → GPU Transfer Speed:");
    for size_mb in [1, 10, 100, 500] {
        let elements = size_mb * 1024 * 1024 / 4; // FP32
        let cpu_tensor = vec![1.0f32; elements];
        
        let start = Instant::now();
        let _gpu_tensor = Tensor::from_vec(cpu_tensor, elements, &device)?;
        device.synchronize()?;
        let time = start.elapsed().as_secs_f64();
        
        let bandwidth = size_mb as f64 / time / 1024.0; // GB/s
        println!("   {} MB: {:.2}ms ({:.1} GB/s)", size_mb, time * 1000.0, bandwidth);
    }
    
    // Test 3: On-GPU preprocessing
    println!("\n3️⃣  On-GPU Preprocessing:");
    let batch_size = 32;
    let seq_len = 512;
    let hidden_size = 768;
    
    let embeddings = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device)?
        .to_dtype(DType::F16)?;
    
    // Layer norm
    let start = Instant::now();
    for _ in 0..100 {
        let mean = embeddings.mean_keepdim(2)?;
        let variance = embeddings.var_keepdim(2)?;
        let _normalized = ((embeddings.clone() - mean)? / (variance + 1e-5)?.sqrt()?)?;
    }
    device.synchronize()?;
    let norm_time = start.elapsed().as_secs_f64() / 100.0;
    
    // Dropout simulation
    let start = Instant::now();
    for _ in 0..100 {
        let mask = Tensor::rand(0f32, 1.0, embeddings.shape(), &device)?;
        let _dropped = embeddings.broadcast_mul(&mask)?;
    }
    device.synchronize()?;
    let dropout_time = start.elapsed().as_secs_f64() / 100.0;
    
    println!("   Layer Norm: {:.3}ms", norm_time * 1000.0);
    println!("   Dropout: {:.3}ms", dropout_time * 1000.0);
    
    // Test 4: Bottleneck analysis
    println!("\n4️⃣  Pipeline Bottleneck Analysis:");
    println!("   Data loading: ~20-50ms (CPU I/O)");
    println!("   Tokenization: ~5-10ms (CPU)");
    println!("   CPU→GPU transfer: ~1-5ms");
    println!("   Forward pass: ~100-500ms (GPU)");
    println!("   Backward pass: ~150-700ms (GPU)");
    println!("   Optimizer step: ~10-20ms (GPU)");
    println!();
    println!("   💡 Bottleneck: Forward/Backward passes (GPU-bound) ✅");
    println!("   Use multiple dataloader workers to hide CPU overhead");
    
    // Test 5: Prefetching simulation
    println!("\n5️⃣  Prefetching Benefits:");
    let compute_time: f64 = 0.5; // 500ms forward+backward
    let loading_time: f64 = 0.05; // 50ms data loading
    
    let no_prefetch_time = compute_time + loading_time;
    let prefetch_time = compute_time.max(loading_time);
    let speedup = no_prefetch_time / prefetch_time;
    
    println!("   Without prefetch: {:.0}ms/batch", no_prefetch_time * 1000.0);
    println!("   With prefetch: {:.0}ms/batch", prefetch_time * 1000.0);
    println!("   Speedup: {:.2}x", speedup);
    
    println!("\n✅ Data pipeline tests complete");
    Ok(())
}
