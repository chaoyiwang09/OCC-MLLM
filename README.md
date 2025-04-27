# OCC-MLLM-COT (Multi-stage OCClusion reasoning with MLLM via 3D-aware supervision and Chain-of-Thoughts Reasoning)

## Overview

MORE-MLLM is a cutting-edge framework for multi-modal large language model (MLLM) tasks, with a primary focus on 3D object reconstruction. This framework is built upon two foundational environments: [InternVL](https://github.com/OpenGVLab/InternVL) and [MOHO](https://github.com/ZhangCYG/MOHO). 

This repository provides instructions to:
- Set up the required environments.
- Generate 3D reconstructed images.
- Fine-tune the model using your dataset.
- Perform inference with the fine-tuned model.

Additionally, our best model, **OCC-MLLM-COT**, is publicly available for download.
We provide the trained model checkpoints via Baidu Netdisk.

    Download link: https://pan.baidu.com/s/1bZ4NztX8WlFkHoTsjHHSVg?pwd=cprj

## 📦 Model Overview

| Model Name | Parameters | File Name |
|:----------:|:----------:|:---------:|
|Qwen2-1B Model | ~1 Billion | outputdir-1b-Q1-Q6-MPO-SC-067.tar.gz |
|Internlm2-2B Model | ~2 Billion | outputdir-2b-Q1-Q6-MPO-SC-067.tar.gz |
|Phi3-4B Model | ~4 Billion | outputdir-4b-Q1-Q6-MPO-SC-069.tar.gz |
|Internlm2.5-8B Model | ~8 Billion | outputdir-8b-Q1-Q6-MPO-SC-075.tar.gz |

> 🔗 Download link: [Baidu Netdisk Link](https://pan.baidu.com/s/1bZ4NztX8WlFkHoTsjHHSVg?pwd=cprj) (extraction code: `cprj`)


---

## Installation and Environment Setup

### Step 1: Clone this repository
```bash
git clone https://github.com/OpenGVLab/InternVL.git
```

### Step 2: Set up the `InternVL` environment
1. Create and activate a new conda environment:
   ```bash
   conda create -n internvl python=3.9
   conda activate internvl
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install `flash-attn==2.3.6` for training chat models:
   ```bash
   pip install flash-attn==2.3.6 --no-build-isolation
   ```

### Step 3: Set up the `MOHO` environment
1. Clone the [MOHO repository](https://github.com/ZhangCYG/MOHO).
2. Activate the MOHO conda environment:
   ```bash
   conda activate moho
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3D Image Reconstruction

To generate 3D reconstructed images:
1. Ensure that `obman_view_test_all.txt` is available in your working directory.
2. Run the reconstruction script:
   ```bash
   python recontruct_3Dimage.py
   ```
3. The 3D reconstructed images are uploaded to:
   ```
   https://obmandataset.s3.us-east-2.amazonaws.com/3d_views_fine_test_all/{imageID}_0_obman_test_rgb_{imageID}.jpg.png
   ```

---

## Repository Structure

The repository is organized by checkpoint stages, with each stage containing the following components:

### Example SFT2 (and other checkpoint folders)

```
SFT2/
├── generate_train_json_Q1Q6.py     # Script to generate training data
├── eccv_train_convert_sft2.jsonl   # Training dataset
├── finetune-sft2.sh               # Fine-tuning script
├── inferenceImageAcc.py           # Inference scripts
├── internvl_chat_finetune.py      # Core training script
└── result/                        # Inference results for current checkpoint
```

## Usage Instructions

### Data Preparation
1. Run `generate_train_json_Q1Q6.py` to prepare your training data:
```bash
python generate_train_json_Q1Q6.py
```
This will generate the training dataset in JSONL format.

### Training
1. Place `internvl_chat_finetune.py` in the corresponding checkpoint directory.
2. Execute the fine-tuning script:
```bash
bash finetune-sft2.sh
```

### Inference
1. Use the scripts `inferenceImageAcc.py` to run inference with the trained model.
2. Results saved in the `result` directory.

## File Descriptions

- **generate_train_json_Q1Q6.py**: Script for generating the training dataset in the required format
- **eccv_train_convert_sft2.jsonl**: Training dataset file containing prepared data
- **finetune-sft2.sh**: Shell script containing all parameters and commands for fine-tuning
- **inferenceImageAcc.py**: inference-related script
- **internvl_chat_finetune.py**: Main training script (must be placed in the corresponding checkpoint directory)
- **result/**: Directory containing inference results for the current checkpoint
---

## Best Model

Our best-performing model, **OCC-MLLM-COT**, is publicly available at:
[Model Download Link](#)
> Replace `#` with the URL where your model is hosted.

---



## License

This project is licensed under the [MIT License](LICENSE). Please refer to the `LICENSE` file for details.

---

## Acknowledgements

We extend our gratitude to:
- [InternVL](https://github.com/OpenGVLab/InternVL): The foundational multi-modal large language model framework.
- [MOHO](https://github.com/ZhangCYG/MOHO): The base model for 3D object reconstruction.

---

For questions or issues, feel free to open an issue or reach out chaoyiwang@mail.sim.ac.cn.



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

