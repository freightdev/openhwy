use candle_core::{Device, Tensor, Result, DType};
use std::time::Instant;

// Simulate optimizer updates
fn adam_update(
    param: &Tensor,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    step: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
) -> Result<Tensor> {
    // m = beta1 * m + (1 - beta1) * grad
    *m = ((m.clone() * beta1 as f64)? + (grad * (1.0 - beta1) as f64)?)?;
    
    // v = beta2 * v + (1 - beta2) * grad^2
    let grad_sq = grad.sqr()?;
    *v = ((v.clone() * beta2 as f64)? + (grad_sq * (1.0 - beta2) as f64)?)?;
    
    // Bias correction
    let m_hat = (m.clone() / (1.0 - beta1.powi(step as i32)) as f64)?;
    let v_hat = (v.clone() / (1.0 - beta2.powi(step as i32)) as f64)?;
    
    // Update: param = param - lr * m_hat / (sqrt(v_hat) + eps)
    let denom = (v_hat.sqrt()? + eps as f64)?;
    let update = (m_hat / denom)? * lr as f64;
    Ok((param - update)?)
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("🔧 Optimizer Performance Test\n");
    
    let param_sizes = [
        ("Small layer", 512, 512),
        ("Medium layer", 2048, 2048),
        ("Large layer", 4096, 4096),
    ];
    
    for (name, rows, cols) in param_sizes {
        println!("{}:", name);
        
        // Initialize parameters and optimizer states
        let params = Tensor::randn(0f32, 0.01, (rows, cols), &device)?
            .to_dtype(DType::F32)?;
        let grad = Tensor::randn(0f32, 0.001, (rows, cols), &device)?
            .to_dtype(DType::F32)?;
        
        let mut m = Tensor::zeros((rows, cols), DType::F32, &device)?;
        let mut v = Tensor::zeros((rows, cols), DType::F32, &device)?;
        
        // Adam hyperparameters
        let lr = 0.001;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        
        device.synchronize()?;
        let start = Instant::now();
        
        // Simulate 100 optimizer steps
        let mut current_params = params;
        for step in 1..=100 {
            current_params = adam_update(
                &current_params,
                &grad,
                &mut m,
                &mut v,
                step,
                lr,
                beta1,
                beta2,
                eps,
            )?;
        }
        
        device.synchronize()?;
        let time = start.elapsed().as_secs_f64() / 100.0;
        
        let param_count = rows * cols;
        let memory_mb = param_count * 4 * 3 / 1024 / 1024; // params + m + v
        
        println!("   Params: {}", param_count);
        println!("   Memory: {} MB (param + 2 states)", memory_mb);
        println!("   Update time: {:.3}ms/step", time * 1000.0);
        println!();
    }
    
    // Learning rate scheduling
    println!("📊 Learning Rate Schedules:");
    let initial_lr = 0.001;
    let warmup_steps = 1000;
    let total_steps = 10000;
    
    println!("   Initial LR: {}", initial_lr);
    println!("   Warmup steps: {}", warmup_steps);
    println!("   Total steps: {}", total_steps);
    
    for step in [100, 500, 1000, 5000, 10000] {
        let lr = if step < warmup_steps {
            initial_lr * (step as f32 / warmup_steps as f32)
        } else {
            let progress = (step - warmup_steps) as f32 / (total_steps - warmup_steps) as f32;
            initial_lr * (1.0 - progress)
        };
        println!("   Step {}: LR = {:.6}", step, lr);
    }
    
    // Optimizer comparison
    println!("\n⚖️  Optimizer Memory Requirements (1B params):");
    println!("   SGD: 4 GB (params only)");
    println!("   SGD + Momentum: 8 GB (params + momentum)");
    println!("   Adam: 12 GB (params + m + v)");
    println!("   AdamW: 12 GB (same as Adam)");
    println!("   8-bit Adam: 6 GB (quantized states) ✅ Best for 4GB VRAM");
    
    println!("\n✅ Optimizer tests complete");
    Ok(())
}
