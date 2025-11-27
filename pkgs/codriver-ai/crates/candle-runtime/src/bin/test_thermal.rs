use candle_core::{Device, Tensor, Result, DType};
use std::time::Instant;
use std::thread;
use std::time::Duration;

fn sustained_workload(device: &Device, duration_secs: u64) -> Result<Vec<f64>> {
    let size = 2048;
    let a = Tensor::randn(0f32, 1.0, (size, size), device)?.to_dtype(DType::F16)?;
    let b = Tensor::randn(0f32, 1.0, (size, size), device)?.to_dtype(DType::F16)?;
    
    let mut tflops_history = Vec::new();
    let start = Instant::now();
    
    while start.elapsed().as_secs() < duration_secs {
        let iter_start = Instant::now();
        
        for _ in 0..10 {
            let _c = a.matmul(&b)?;
        }
        
        device.synchronize()?;
        
        let iter_time = iter_start.elapsed().as_secs_f64();
        let ops = 2.0 * size.pow(3) as f64 * 10.0;
        let tflops = ops / iter_time / 1e12;
        
        tflops_history.push(tflops);
        
        // Small delay to allow thermal reading
        thread::sleep(Duration::from_millis(100));
    }
    
    Ok(tflops_history)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("🌡️  Thermal & Stability Test\n");
    println!("⚠️  This will run your GPU at 100%% for 5 minutes!");
    println!("    Make sure your laptop has adequate cooling.\n");
    println!("    Press Ctrl+C to stop early if temps get too high.\n");
    
    thread::sleep(Duration::from_secs(3));
    
    // Test 1: Cold start performance
    println!("1️⃣  Cold Start (GPU at idle temp):");
    let cold_start = sustained_workload(&device, 10)?;
    let cold_avg: f64 = cold_start.iter().sum::<f64>() / cold_start.len() as f64;
    println!("   Average: {:.2} TFLOPS", cold_avg);
    println!("   Samples: {}", cold_start.len());
    
    // Test 2: Sustained load (thermal throttling test)
    println!("\n2️⃣  Sustained Load (5 minutes - thermal test):");
    println!("   Running... (watch nvidia-smi in another terminal)\n");
    
    let sustained = sustained_workload(&device, 300)?;
    
    // Analyze performance over time
    let chunk_size = sustained.len() / 10;
    println!("   Performance over time:");
    
    for (i, chunk) in sustained.chunks(chunk_size).enumerate() {
        let avg: f64 = chunk.iter().sum::<f64>() / chunk.len() as f64;
        let min = chunk.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = chunk.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        println!("   Minute {}: {:.2} TFLOPS (min: {:.2}, max: {:.2})",
            i + 1, avg, min, max);
    }
    
    // Overall statistics
    let sustained_avg: f64 = sustained.iter().sum::<f64>() / sustained.len() as f64;
    let sustained_min = sustained.iter().cloned().fold(f64::INFINITY, f64::min);
    let sustained_max = sustained.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    println!("\n3️⃣  Thermal Analysis:");
    println!("   Cold start: {:.2} TFLOPS", cold_avg);
    println!("   Sustained avg: {:.2} TFLOPS", sustained_avg);
    println!("   Sustained min: {:.2} TFLOPS", sustained_min);
    println!("   Sustained max: {:.2} TFLOPS", sustained_max);
    
    let performance_drop = 100.0 * (1.0 - sustained_avg / cold_avg);
    let variance = sustained_max - sustained_min;
    
    println!("\n   Performance drop: {:.1}%", performance_drop);
    println!("   Variance: {:.2} TFLOPS", variance);
    
    if performance_drop < 5.0 {
        println!("\n   ✅ Excellent: Minimal thermal throttling");
        println!("      Your cooling is adequate for sustained training");
    } else if performance_drop < 15.0 {
        println!("\n   ⚠️  Moderate: Some thermal throttling");
        println!("      Consider improving airflow or reducing ambient temp");
        println!("      Training will work but may be slower over time");
    } else {
        println!("\n   ❌ Significant: Heavy thermal throttling");
        println!("      Cooling is insufficient for sustained workloads");
        println!("      Recommendations:");
        println!("        - Use laptop cooling pad");
        println!("        - Clean dust from vents");
        println!("        - Reduce ambient temperature");
        println!("        - Consider undervolting GPU");
    }
    
    // Test 3: Recovery time
    println!("\n4️⃣  Thermal Recovery:");
    println!("   Waiting 60 seconds for cooldown...");
    thread::sleep(Duration::from_secs(60));
    
    let recovery = sustained_workload(&device, 10)?;
    let recovery_avg: f64 = recovery.iter().sum::<f64>() / recovery.len() as f64;
    
    println!("   After cooldown: {:.2} TFLOPS", recovery_avg);
    let recovery_pct = 100.0 * (recovery_avg / cold_avg);
    println!("   Recovery: {:.0}% of cold start performance", recovery_pct);
    
    if recovery_pct > 95.0 {
        println!("   ✅ Fast recovery - good thermal design");
    } else {
        println!("   ⚠️  Slow recovery - heat soak issue");
    }
    
    // Test 4: Recommendations
    println!("\n5️⃣  Training Recommendations:");
    
    if performance_drop < 10.0 {
        println!("   ✅ Can train continuously");
        println!("   ✅ No special precautions needed");
    } else {
        println!("   ⚠️  Consider training in intervals:");
        println!("      - Train for 30-60 min");
        println!("      - Let GPU cool for 10-15 min");
        println!("      - Resume training");
        println!("   Or:");
        println!("      - Reduce batch size slightly");
        println!("      - Accept slower but stable performance");
    }
    
    println!("\n   💡 Optimal training temperature: 60-75°C");
    println!("   ⚠️  Throttling typically starts: 80-85°C");
    println!("   🔥 Thermal shutdown: 95-100°C");
    
    println!("\n✅ Thermal tests complete");
    println!("   Check nvidia-smi output for actual temperatures");
    
    Ok(())
}
