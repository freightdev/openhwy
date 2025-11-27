#!/bin/bash

echo "🧪 Running Complete ML Test Suite"
echo "=================================="
echo ""

# Test All
echo "Begin Candle Tests:"
cargo run --release --bin test_burn_integration
cargo run --release --bin test_checkpointing
cargo run --release --bin test_data_pipeline
# cargo run --release --bin test_fa2             # (if complied)
cargo run --release --bin test_gradients
cargo run --release --bin test_inference
cargo run --release --bin test_lib_check
cargo run --release --bin test_lora
cargo run --release --bin test_matmul
cargo run --release --bin test_memory_stress
cargo run --release --bin test_mixed_precision
cargo run --release --bin test_optimizers
cargo run --release --bin test_quantization
# cargo run --release --bin test_thermal         # time: 5 minutes
# cargo run --release --bin test_throughput      # time: 30 seconds
cargo run --release --bin test_training_loop
cargo run --release --bin test_training_ops
cargo run --release --bin test_transformer_ops
cargo run --release --bin test_vram
# cargo run --release --bin test_workload        # time: Variable

echo ""
echo "✅ All tests complete!"
echo ""
