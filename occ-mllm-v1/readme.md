# OCC-MLLM-V1: Chain-of-Thoughts Guided Multimodal Learning for Occluded Object Understanding

This repository contains the official implementation of our paper: "Chain-of-Thoughts Guided Multimodal Learning for Occluded Object Understanding"

## Overview

Comprehending occluded objects remains a significant challenge for existing large visual-language multimodal models. Current state-of-the-art multimodal large models struggle to provide satisfactory results when understanding occluded objects through universal visual encoders and supervised learning strategies. 

Inspired by the effectiveness of step-by-step Chain-of-Thoughts (CoTs) reasoning in large language models, we propose a novel end-to-end visual-language multimodal framework with self-generated step-by-step CoTs guidance for understanding occluded objects.

## Directory Structure

```
occ-mllm-v1/
├── inference.py                            # Main inference script
├── step1/
│   ├── 4b_q1.sh                           # Training script for step 1
│   └── train_Q1_10140.jsonl               # Training data for step 1
├── step2/
│   ├── 4b_q1_q6.sh                        # Training script for step 2
│   └── train_Q1_Q6_10140.jsonl            # Training data for step 2
├── step3/
│   ├── 4b_q1_q6_MPO.sh                    # MPO training script
│   ├── 4b_q1_q6-MPO-q1.sh                 # MPO-Q1 training script
│   ├── train_Q1_10140.jsonl               # Training data
│   └── train_Q1_Q6_balanced_dpo_10140.jsonl # Balanced DPO training data
└── step4/
    ├── 4b_q1_q6-MPO-q1-sc.sh              # SC training script
    ├── 4b_q1_q6-MPO-q1-sc-q1.sh           # SC-Q1 training script
    ├── 4b_q1_q6-MPO-q1-sc-q1-cot.sh       # SC-Q1-CoT training script
    ├── internvl_chat_finetune_sc.py       # InternVL chat fine-tuning script
    ├── sc_trainer.py                       # SC trainer implementation
    ├── train_Q1_101.jsonl                  # Small training sample
    ├── train_Q1_10140.jsonl                # Full training data for Q1
    └── train_Q1_Q6_Q0_4_balanced_10140.jsonl # Balanced training data
```

## Training Pipeline

Our training process consists of four sequential steps:

### Step 1: Initial Training
```bash
cd step1
bash 4b_q1.sh
```
This step initializes the model with basic occluded object understanding capabilities using the `train_Q1_10140.jsonl` dataset.

### Step 2: Multimodal Query Enhancement
```bash
cd ../step2
bash 4b_q1_q6.sh
```
This step enhances the model's ability to handle multiple types of queries using the `train_Q1_Q6_10140.jsonl` dataset.

### Step 3: Minimum Preference Optimization
```bash
cd ../step3
bash 4b_q1_q6_MPO.sh
# or
bash 4b_q1_q6-MPO-q1.sh
```
This step applies preference optimization techniques to improve the model's reasoning capabilities.

### Step 4: Self-Consistency Training with Chain-of-Thoughts
```bash
cd ../step4
bash 4b_q1_q6-MPO-q1-sc.sh
# or for Chain-of-Thoughts specific training
bash 4b_q1_q6-MPO-q1-sc-q1-cot.sh
```
The final step incorporates self-consistency and Chain-of-Thoughts reasoning to further enhance the model's performance.

## Inference

To run inference with the trained model:

```bash
python inference.py
```

## Dataset

Our dataset contains 110k samples of occluded objects held in hand, specially designed for training multimodal models to understand occluded objects through Chain-of-Thoughts reasoning. The training data is provided in JSONL format in the respective step directories.

---

## 🧩 Inference Process Visualization

The following figure showcases the inference steps on sample inputs. It demonstrates how the model analyzes the original RGB image, performs self-assessment on visibility clarity, optionally applies 3D reconstruction if necessary, and ultimately identifies the object held in the hand.

<p align="center">
  <img src="./sampleImg/visualresults.jpg" alt="Inference Process Visualization" width="85%">
</p>

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@misc{wang2025occmllmcotalphamultistageocclusionrecognition,
      title={OCC-MLLM-CoT-Alpha: Towards Multi-stage Occlusion Recognition Based on Large Language Models via 3D-Aware Supervision and Chain-of-Thoughts Guidance}, 
      author={Chaoyi Wang and Baoqing Li and Xinhan Di},
      year={2025},
      eprint={2504.04781},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.04781}, 
}
```

## License

MIT License

Copyright (c) 2025 Chaoyi Wang, Baoqing Li, Xinhan Di

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

