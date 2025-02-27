# LLM-Powered Trajectory Prediction & Compression for Robotic Arm Path Planning 
This project explores the use of Large Language Models (LLMs) for trajectory prediction and compression in robotic arm path planning. By fine-tuning an LLM (e.g., Phi-2) on trajectory sequence data, the model learns to predict future trajectory points and optimize path execution with GPU-accelerated PCA (via CuPy/RAPIDS cuML).

### Advanced Features:

## Adaptive Compression: Context-aware rank selection using LLM attention patterns
Temporal Coherence: Add contrastive loss for smooth trajectory transitions
Hardware-Aware: CUDA kernel fusion for PCA+prediction pipeline
Failure Recovery: Compressed trajectory error correction via learned codebooks
Meta-Learning: Few-shot adaptation to new manipulator configurations

## Critical Optimization Strategies for T4 GPU:
Use gradient accumulation (4+ steps) with microbatching
Implement FP16 mixed precision with NVIDIA Apex
Freeze first 6 layers of Phi-2 during fine-tuning
Use PyTorch's optim.lr_scheduler.CyclicLR for stable training
Pre-compile custom CUDA ops for PCA using Numba

## Validation Metrics:
Compression: Bits-per-parameter (BPP) <0.15 at 10:1 ratio
Prediction: End-point error <1.5cm RMS for 2s lookahead
Temporal Alignment: Dynamic Time Warping score >0.92
Runtime: <50ms inference time per 100-step trajectory
