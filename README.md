# KVSlimmer

> **KVSlimmer: Theoretical Insights and Practical Optimizations for Asymmetric KV Merging**

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2603.00907)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()

Official implementation of **KVSlimmer**, a theoretically grounded and efficient asymmetric KV cache merging framework for long-context LLM inference.

KVSlimmer is developed on top of the **AsymKV** codebase and extends it with improved theoretical foundations, a more principled merging formulation, and practical optimizations for efficient long-context generation.

---

## Updates

- **[2026-03]** Paper released on arXiv.
- **[2026-03]** Initial repository created.
- More code, scripts, and reproducibility materials will be released soon.

---

## Overview

Large Language Models (LLMs) suffer from substantial memory and latency overhead when serving long-context inference, largely due to the growth of the Key-Value (KV) cache.

**AsymKV** revealed an important asymmetry between keys and values:
- **keys** tend to exhibit stronger local homogeneity,
- **values** tend to preserve higher heterogeneity.

Built upon this observation, **KVSlimmer** further improves asymmetric KV merging from both **theoretical** and **practical** perspectives.

### KVSlimmer provides:
- a stronger theoretical understanding of asymmetric KV merging,
- a more principled and efficient merging objective,
- practical optimizations for long-context inference,
- improved performance-memory-latency trade-offs.

---

## Key Contributions

- **Theoretical insights into KV asymmetry**  
  KVSlimmer provides a stronger theoretical interpretation of why asymmetric merging works.

- **Practical optimization for KV merging**  
  KVSlimmer introduces a more efficient merging strategy tailored for long-context LLM inference.

- **Efficient long-context serving**  
  KVSlimmer reduces KV cache overhead while maintaining strong downstream performance.

- **Improved empirical performance**  
  KVSlimmer is designed to outperform prior AsymKV-style methods under comparable budgets.

---

## Relation to AsymKV

This repository builds upon and extends the open-source implementation of **AsymKV**.

Compared with AsymKV, KVSlimmer focuses on:
- stronger theoretical grounding,
- improved asymmetric KV merging formulation,
- more efficient practical implementation,
- better long-context inference trade-offs.

Some parts of the training / evaluation / inference pipeline may be adapted from the AsymKV repository, while the core KVSlimmer algorithm and related improvements are introduced in this work.

---

## Method Illustration

> You can place your method figure here, for example `assets/framework.png`.

<p align="center">
  <img src="assets/method.png" width="85%">
</p>

**Figure:** Overview of KVSlimmer.  
KVSlimmer performs asymmetric KV cache merging for long-context inference by leveraging the distinct structural properties of keys and values.

---

## Repository Structure

```text
KVSlimmer/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── assets/
│   ├── framework.png
│   └── results.png
├── kvslimmer/
│   ├── __init__.py
│   ├── cache.py
│   ├── merge.py
│   ├── attention_patch.py
│   └── utils.py
├── scripts/
│   ├── run_longbench.sh
│   ├── eval_longbench.py
│   └── benchmark_latency.py
├── examples/
│   ├── quick_start.py
│   └── demo_qwen2.py
├── configs/
│   ├── llama31_8b.yaml
│   └── qwen2_7b.yaml
├── docs/
│   └── reproduction.md
└── outputs/
