use candle_core::{Device, Tensor, Result, DType};
use std::time::Instant;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("⚡ GPU Throughput Test\n");
    
    println!("Testing sustained performance over 30 seconds...\n");
    
    let size = 2048;
    let a = Tensor::randn(0f32, 1.0, (size, size), &device)?.to_dtype(DType::F16)?;
    let b = Tensor::randn(0f32, 1.0, (size, size), &device)?.to_dtype(DType::F16)?;
    
    device.synchronize()?;
    
    let mut iterations = 0u64;
    let mut tflops_samples = Vec::new();
    let start = Instant::now();
    let test_duration = 30.0;
    
    while start.elapsed().as_secs_f64() < test_duration {
        let iter_start = Instant::now();
        
        for _ in 0..10 {
            let _c = a.matmul(&b)?;
        }
        
        device.synchronize()?;
        iterations += 10;
        
        let iter_time = iter_start.elapsed().as_secs_f64();
        let ops = 2.0 * size.pow(3) as f64 * 10.0;
        let tflops = ops / iter_time / 1e12;
        tflops_samples.push(tflops);
        
        if iterations % 100 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let avg_tflops: f64 = tflops_samples.iter().sum::<f64>() / tflops_samples.len() as f64;
            print!("\r  {:.1}s: {:.2} TFLOPS avg, {} iterations", 
                elapsed, avg_tflops, iterations);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }
    
    println!("\n");
    
    let total_time = start.elapsed().as_secs_f64();
    let avg_tflops: f64 = tflops_samples.iter().sum::<f64>() / tflops_samples.len() as f64;
    let min_tflops = tflops_samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_tflops = tflops_samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    println!("📊 Results:");
    println!("   Total iterations: {}", iterations);
    println!("   Total time: {:.1}s", total_time);
    println!("   Average: {:.2} TFLOPS", avg_tflops);
    println!("   Min: {:.2} TFLOPS", min_tflops);
    println!("   Max: {:.2} TFLOPS", max_tflops);
    println!("   Variance: {:.2} TFLOPS", max_tflops - min_tflops);
    
    if max_tflops - min_tflops > 0.5 {
        println!("\n⚠️  High variance detected - possible thermal throttling");
    } else {
        println!("\n✅ Stable performance - no throttling");
    }
    
    Ok(())
}
