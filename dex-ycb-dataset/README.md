# OCC-MLLM DexYCB-based Dataset

This folder contains the OCC-MLLM dataset (Dataset 2) built on DexYCB. It includes ~34K samples across 20 categories, designed for occlusion and multi-object scenes in multimodal/vision learning and evaluation.

## Download
- Link: `https://pan.baidu.com/s/1olDfMPFIDlmsR9NKDWZymQ?pwd=k56i`
- Access code: `k56i`
- Archive parts (4 splits, each ~4.47G):
  - `dex_ycb_occ_mllm_dataset.tar.gz.part00`
  - `dex_ycb_occ_mllm_dataset.tar.gz.part01`
  - `dex_ycb_occ_mllm_dataset.tar.gz.part02`
  - `dex_ycb_occ_mllm_dataset.tar.gz.part03`

## Merge and Extract
- Linux/macOS:
  - Merge: `cat dex_ycb_occ_mllm_dataset.tar.gz.part0* > dex_ycb_occ_mllm_dataset.tar.gz`
  - Extract: `tar -zxvf dex_ycb_occ_mllm_dataset.tar.gz`
- Notes:
  - Ensure all four parts are in the same directory before merging.
  - `-z` handles gzip; `-xvf` extracts with verbose output.

## Contents After Extraction
- `train_origin`: original training RGB images
- `train_rendered`: 3D reconstructed/rendered training images
- `test_origin`: original test RGB images
- `test_rendered`: 3D reconstructed/rendered test images

## Category Distribution (20 classes; percentages)
| Category            | %    |
|---------------------|------|
| Pitcher             | 5.61 |
| Cracker Box         | 5.43 |
| Mustard Bottle      | 5.33 |
| Wood Block          | 5.19 |
| Tomato Soup Can     | 5.17 |
| Master Chef Can     | 5.15 |
| Gelatin Box         | 5.15 |
| Bleach Cleanser     | 5.14 |
| Pudding Box         | 5.11 |
| Sugar Box           | 5.06 |
| Power Drill         | 5.03 |
| Bowl                | 5.03 |
| Mug                 | 5.03 |
| Tuna Fish Can       | 4.93 |
| Potted Meat Can     | 4.91 |
| Banana              | 4.78 |
| Large Clamp         | 4.72 |
| Foam Brick          | 4.65 |
| Scissors            | 4.32 |
| Marker              | 4.24 |

## Repository Index Files
- `dexycb_train_dual_images.jsonl`: training set index
- `dexycb_test_dual_images.jsonl`: test set index
- `dexycb_dual_image_clarity.json` and `dexycb_dual_image_clarity_balanced.json`: auxiliary indices for image clarity

> Use the indices above to locate images/annotations in the extracted folders.

## License and Acknowledgments
- Built upon DexYCB objects and scene settings. Please cite and acknowledge the original DexYCB project in academic use.
 - For research use only; see the repository license for details.

## Citation
If you use this dataset in a paper or project, please include: dataset name (OCC-MLLM DexYCB-based Dataset), version info (split archive; 34K samples; 20 classes). Also cite:

DexYCB:
```
@INPROCEEDINGS{chao:cvpr2021,
  author    = {Yu-Wei Chao and Wei Yang and Yu Xiang and Pavlo Molchanov and Ankur Handa and Jonathan Tremblay and Yashraj S. Narang and Karl Van Wyk and Umar Iqbal and Stan Birchfield and Jan Kautz and Dieter Fox},
  title     = {{DexYCB}: A Benchmark for Capturing Hand Grasping of Objects},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2021},
}
```

OCC-MLLM:
```
@article{wang5702186occ,
  title={OCC-MLLM-V1: Commonsense-Guided Multi-Modal LLM Based Agent for Occlusion Reasoning With Internal Chain-of-Thoughts (CoTs) Guidance},
  author={Wang, Chaoyi and He, Qingdong and Pei, Jun and Xia, Lijie and Liu, Jianpo and Li, Baoqing and Di, Xinhan},
  journal={Available at SSRN 5702186}
}
```

