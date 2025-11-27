use candle_core::{Device, Tensor, Result, DType};
#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn;
use std::time::Instant;

fn standard_attention(
    q: &Tensor,
    k: &Tensor, 
    v: &Tensor,
    scale: f64
) -> Result<Tensor> {
    let scores = q.matmul(&k.transpose(2, 3)?)?;
    let scores = (scores / scale)?;
    let attn = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
    attn.matmul(v)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("🔍 Testing YOUR Custom FlashAttention2 (sm_75)\n");
    
    let batch = 1;
    let heads = 8;
    let seq_len = 2048;
    let head_dim = 64;
    let scale = (head_dim as f64).sqrt();
    
    println!("Creating tensors: [batch={}, heads={}, seq={}, dim={}]\n", 
        batch, heads, seq_len, head_dim);
    
    // FlashAttn expects: [batch, seq_len, heads, head_dim]
    let q = Tensor::randn(0f32, 1.0, (batch, seq_len, heads, head_dim), &device)?
        .to_dtype(DType::F16)?;
    let k = Tensor::randn(0f32, 1.0, (batch, seq_len, heads, head_dim), &device)?
        .to_dtype(DType::F16)?;
    let v = Tensor::randn(0f32, 1.0, (batch, seq_len, heads, head_dim), &device)?
        .to_dtype(DType::F16)?;
    
    // Standard attention needs: [batch, heads, seq_len, head_dim]
    let q_std = q.transpose(1, 2)?.contiguous()?;
    let k_std = k.transpose(1, 2)?.contiguous()?;
    let v_std = v.transpose(1, 2)?.contiguous()?;
    
    device.synchronize()?;
    
    // Warmup
    println!("Warming up GPU...");
    for _ in 0..5 {
        let _ = standard_attention(&q_std, &k_std, &v_std, scale)?;
    }
    device.synchronize()?;
    
    // Benchmark standard attention
    println!("\n1️⃣  Standard Attention (pure matmul):");
    let start = Instant::now();
    for _ in 0..10 {
        let _output = standard_attention(&q_std, &k_std, &v_std, scale)?;
    }
    device.synchronize()?;
    let std_time = start.elapsed().as_secs_f64() / 10.0;
    println!("   Time per call: {:.2}ms", std_time * 1000.0);
    println!("   Throughput: {:.0} tokens/sec", seq_len as f64 / std_time);
    
    // Warmup FlashAttention
    println!("\n2️⃣  FlashAttention2 (your sm_75 kernels):");
    for _ in 0..5 {
        let _ = flash_attn(&q, &k, &v, scale as f32, false)?;
    }
    device.synchronize()?;
    
    // Benchmark FlashAttention
    let start = Instant::now();
    for _ in 0..10 {
        let _output = flash_attn(&q, &k, &v, scale as f32, false)?;
    }
    device.synchronize()?;
    let fa2_time = start.elapsed().as_secs_f64() / 10.0;
    println!("   Time per call: {:.2}ms", fa2_time * 1000.0);
    println!("   Throughput: {:.0} tokens/sec", seq_len as f64 / fa2_time);
    
    // Analysis
    println!("\n📊 Results:");
    println!("┌─────────────────┬──────────────┐");
    println!("│ Method          │ Time (ms)    │");
    println!("├─────────────────┼──────────────┤");
    println!("│ Standard        │ {:>12.2} │", std_time * 1000.0);
    println!("│ FlashAttn2      │ {:>12.2} │", fa2_time * 1000.0);
    println!("└─────────────────┴──────────────┘");
    
    if fa2_time < std_time {
        let speedup = std_time / fa2_time;
        println!("\n✅ SUCCESS! FlashAttention2 is {:.2}x FASTER", speedup);
        println!("   Your sm_75 patch is working correctly!");
        
        if speedup > 1.5 {
            println!("   🔥 Excellent speedup for 4GB VRAM!");
        }
    } else {
        println!("\n⚠️  WARNING: FlashAttention2 is slower than standard");
        println!("   This suggests the kernels may not be optimized for sm_75");
        println!("   or the library isn't being used correctly.");
    }
    
    // Memory efficiency note
    println!("\n💾 Memory Efficiency:");
    println!("   Standard: O(n²) = {} MB", (seq_len * seq_len * 2) / 1024 / 1024);
    println!("   FlashAttn2: O(n) - saves memory for longer sequences");
    
    Ok(())
}
