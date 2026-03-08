# KVSlimmer

> **KVSlimmer: Theoretical Insights and Practical Optimizations for Asymmetric KV Merging**

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2603.00907)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()

**KVSlimmer**, a theoretically grounded and efficient asymmetric KV cache merging framework for long-context LLM inference.

This repository is built upon the open-source **AsymKV** codebase, and extends it with improved theoretical foundations, a more principled merging formulation, and practical optimizations for efficient long-context generation. We sincerely thank the authors of **AsymKV** for making their code publicly available.

---

## Updates

- **[2026-03]** Paper released on arXiv.
- **[2026-03]** Initial repository created.
- More code, scripts, and reproducibility materials will be released soon.

---

## Overview
<p align="center">
  <img src="assets/method.png" width="85%">
</p>

The growing computational and memory demands of the Key‑Value (KV) cache significantly limit the ability of Large Language Models (LLMs). While KV merging has emerged as a promising solution, existing methods that rely on empirical observations of KV asymmetry and gradient-based Hessian approximations lack a theoretical foundation and incur suboptimal compression and inference overhead.
To bridge these gaps, we establish a theoretical framework that characterizes this asymmetry through the spectral energy distribution of projection weights, demonstrating that concentrated spectra in Query/Key weights induce feature homogeneity, whereas dispersed spectra in Value weights preserve heterogeneity.
Then, we introduce KVSlimmer, an efficient algorithm that captures exact Hessian information through a mathematically exact formulation, and derives a closed-form solution utilizing only forward-pass variables, resulting in a gradient-free approach that is both memory- and time-efficient.
Extensive experiments across various models and benchmarks demonstrate that KVSlimmer consistently outperforms SOTA methods. For instance, on Llama3.1-8B-Instruct, it improves the LongBench average score by 0.92 while reducing memory costs and latency by 29\% and 28\%, respectively.

# Installation
## Environment Setup
## Environment Setup
```bash
conda create -yn kvslimmer python=3.9
conda activate kvslimmer

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

git clone https://github.com/the-scale-lab/Asymkv.git
cd Asymkv
pip install -e .

cd ..
git clone <YOUR_KVSLIMMER_REPO_URL> KVSlimmer
cd KVSlimmer
pip install -e .
```

## Model Requirements
```json
{
    "Llama-3-8B-Instruct" : "transformers==4.33.0",
    "Llama-3.1-8B-Instruct" : "transformers==4.44.2",
    "Mistral-7B-Instruct-v0.3": "transformers==4.44.2",
    "Qwen2-7B-Instruct" : "transformers==4.44.2",
}
```
# Experiments

## LongBench
Run the following command to evaluate KVSlimmer on LongBench:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port 6656 pred.py --model Llama-3.1-8B-Instruct
```
# Acknowledgements
This repo is built upon the following projects:
- [LongBench](https://github.com/THUDM/LongBench/)
- [Streamingllm](https://github.com/mit-han-lab/streaming-llm)
- [Pyramidkv](https://github.com/Zefan-Cai/KVCache-Factory/)
- [AsymKV](https://github.com/the-scale-lab/Asymkv/)

# TODO
- [ ] Implement Triton-based FlashAttention kernel with AsymKV support
