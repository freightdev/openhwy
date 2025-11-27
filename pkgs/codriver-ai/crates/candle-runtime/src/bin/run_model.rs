// src/bin/run_model.rs

use candle_core::{Device, Tensor, DType, Result, quantized::gguf_file};
use candle_transformers::models::quantized_llama as model;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;
use std::path::PathBuf;

const TEMPERATURE: f64 = 0.7;
const TOP_P: f64 = 0.9;
const SEED: u64 = 42;
const SAMPLE_LEN: usize = 256;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    
    // Get model path from args or default
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("models/model.gguf")
    };
    
    let prompt = if args.len() > 2 {
        args[2..].join(" ")
    } else {
        "Hello".to_string()
    };
    
    println!("Loading model: {:?}", model_path);
    println!("Device: CUDA");
    println!("Prompt: {}\n", prompt);
    
    // Load GGUF file
    let mut file = std::fs::File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    
    // Load model from GGUF content
    let mut model = model::ModelWeights::from_gguf(content, &mut file, &device)?;
    
    // Load tokenizer (assumes tokenizer.json in same dir as model)
    let tokenizer_path = model_path.parent()
        .unwrap()
        .join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(|e| candle_core::Error::Msg(format!("Tokenizer error: {}", e)))?;
    
    // Encode prompt
    let tokens = tokenizer
        .encode(prompt.clone(), true)
        .map_err(|e| candle_core::Error::Msg(format!("Encode error: {}", e)))?;
    let tokens = tokens.get_ids();
    let mut tokens = tokens.to_vec();
    
    // Setup generation
    let mut logits_processor = LogitsProcessor::new(SEED, Some(TEMPERATURE), Some(TOP_P));
    
    print!("{}", prompt);
    std::io::Write::flush(&mut std::io::stdout()).ok();
    
    // Generate
    for _ in 0..SAMPLE_LEN {
        let input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
        
        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        
        if let Some(text) = tokenizer.decode(&[next_token], false).ok() {
            print!("{}", text);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
        
        if next_token == tokenizer.token_to_id("<|endoftext|>").unwrap_or(0) {
            break;
        }
    }
    
    println!("\n");
    Ok(())
}
