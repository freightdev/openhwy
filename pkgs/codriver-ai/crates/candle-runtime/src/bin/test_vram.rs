use candle_core::{Device, Tensor, Result};

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("🧪 VRAM Capacity Test - How much can we fit?\n");
    
    let mut total_mb = 0;
    let chunk_mb = 128; // 128MB chunks
    let size = ((chunk_mb * 1024 * 1024) as f64 / 4.0).sqrt() as usize;
    
    println!("Allocating {}MB chunks ({} x {} FP32 matrices)...\n", chunk_mb, size, size);
    
    let mut tensors = Vec::new();
    
    loop {
        match Tensor::randn(0f32, 1.0, (size, size), &device) {
            Ok(t) => {
                tensors.push(t);
                total_mb += chunk_mb;
                print!("  ✓ {}MB allocated\r", total_mb);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
            Err(_) => {
                println!("\n\n🎯 Maximum VRAM capacity: ~{}MB", total_mb);
                println!("   Available for models: ~{}MB", total_mb - 200); // reserve overhead
                break;
            }
        }
    }
    
    println!("\n📊 Model capacity estimates (with {}MB available):", total_mb - 200);
    println!("   Whisper Small (FP16):  ~1000MB  {}", 
        if total_mb > 1200 { "✅ Will fit" } else { "❌ Too large" });
    println!("   Whisper Base (FP16):   ~1500MB  {}", 
        if total_mb > 1700 { "✅ Will fit" } else { "❌ Too large" });
    println!("   Whisper Medium (FP16): ~2500MB  {}", 
        if total_mb > 2700 { "✅ Will fit" } else { "❌ Too large" });
    println!("   SD 1.5 (FP16):         ~2000MB  {}", 
        if total_mb > 2200 { "✅ Will fit" } else { "❌ Too large" });
    
    Ok(())
}
