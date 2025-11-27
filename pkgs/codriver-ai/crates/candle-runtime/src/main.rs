use candle_core::{Device, Tensor, Result};
use std::time::Instant;
use std::thread;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("🔥 Making GTX 1650 SCREAM - Maximum Load\n");
    println!("Run 'watch -n 0.5 nvidia-smi' in another terminal NOW!\n");
    
    thread::sleep(std::time::Duration::from_secs(3));
    
    // Fill VRAM with multiple large tensors
    println!("Allocating tensors to fill VRAM...");
    let size = 8192; // 8192x8192 FP32 = 256MB each
    let mut tensors = Vec::new();
    
    // Allocate ~3GB of tensors
    for i in 0..12 {
        match Tensor::randn(0f32, 1.0, (size, size), &device) {
            Ok(t) => {
                tensors.push(t);
                println!("  Allocated tensor {}: 256MB (total: {} GB)", i+1, (i+1) as f64 * 0.256);
            }
            Err(e) => {
                println!("  Stopped at {} tensors: {:?}", i, e);
                break;
            }
        }
    }
    
    println!("\n🔥 MAXIMUM COMPUTE TORTURE - 60 seconds");
    println!("GPU should hit 100% utilization and thermal limits\n");
    
    let start = Instant::now();
    let mut iterations = 0u64;
    
    while start.elapsed().as_secs() < 60 {
        // Compute on all tensor pairs
        for i in 0..tensors.len()-1 {
            let _c = tensors[i].matmul(&tensors[i+1])?;
        }
        iterations += 1;
        
        if iterations % 10 == 0 {
            device.synchronize()?;
            let elapsed = start.elapsed().as_secs_f64();
            println!("  {} iterations, {:.1}s elapsed, GPU SCREAMING", iterations, elapsed);
        }
    }
    
    device.synchronize()?;
    let total_time = start.elapsed();
    
    let ops_per_iter = (tensors.len() - 1) as f64 * 2.0 * size as f64 * size as f64 * size as f64;
    let total_tflops = (iterations as f64 * ops_per_iter) / total_time.as_secs_f64() / 1e12;
    
    println!("\n📊 RESULTS:");
    println!("  Iterations: {}", iterations);
    println!("  VRAM used: ~{} GB", tensors.len() as f64 * 0.256);
    println!("  Average TFLOPS: {:.2}", total_tflops);
    println!("  GPU should have hit thermal throttling");
    
    Ok(())
}
