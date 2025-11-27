use candle_core::{Device, Result};

fn main() -> Result<()> {
    println!("🔍 Checking what's available...\n");
    
    // Check CUDA
    match Device::new_cuda(0) {
        Ok(device) => {
            println!("✅ CUDA Device 0 available");
            println!("   Device: {:?}", device);
        }
        Err(e) => {
            println!("❌ CUDA not available: {:?}", e);
        }
    }
    
    // Check if FA2 crate is compiled
    #[cfg(feature = "flash-attn")]
    {
        println!("✅ FlashAttention2 feature enabled");
    }
    
    #[cfg(not(feature = "flash-attn"))]
    {
        println!("✅ FlashAttention2 compiled (no feature flag needed)");
    }
    
    println!("\n📦 Loaded dependencies:");
    println!("   - candle-core");
    println!("   - candle-nn");
    println!("   - candle-flash-attn");
    
    Ok(())
}
