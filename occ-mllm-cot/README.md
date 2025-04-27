# OCC-MLLM-COT (Multi-stage OCClusion reasoning with MLLM via 3D-aware supervision and Chain-of-Thoughts Reasoning)

## Overview

MORE-MLLM is a cutting-edge framework for multi-modal large language model (MLLM) tasks, with a primary focus on 3D object reconstruction. This framework is built upon two foundational environments: [InternVL](https://github.com/OpenGVLab/InternVL) and [MOHO](https://github.com/ZhangCYG/MOHO). 

This repository provides instructions to:
- Set up the required environments.
- Generate 3D reconstructed images.
- Fine-tune the model using your dataset.
- Perform inference with the fine-tuned model.

Additionally, our best model, **OCC-MLLM-COT**, is publicly available for download.

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
