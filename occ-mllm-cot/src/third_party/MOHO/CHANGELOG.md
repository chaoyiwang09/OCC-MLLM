# Changelog

### Changed
- Modified reference implementation in `obman_dataset_ddp_dino.py`
  - Updated reference to use custom implementation
  - Original reference has been replaced with self-implementation

### Added
- New dataset file placement instructions:
  - Test dataset: Place `obman_view_test_all.txt` in the `cache/` directory
  - Training dataset: Place `obman_view_train.txt` in the `cache/` directory

### Updated
- Configuration changes:
  - Updated dataset configuration in conf files to use ObMan dataset
  - Modified `moo_wmask_dp_hand_test.conf` to support ObMan dataset parameters

### Usage
To run the reconstruction:
```bash
python recontruct_3Dimage.py \
    --conf ./confs/moo_wmask_dp_hand_test.conf \
    --case finetuning \
    --mode test_image \
    --gpu_num 3 \
    --device cuda
```

### Prerequisites
- Ensure correct placement of dataset files:
  - For testing: `cache/obman_view_test_all.txt`
  - For training: `cache/obman_view_train.txt`
- Configure dataset parameters in conf files to match ObMan requirements

### Notes
- GPU configuration set to use 3 GPUs with CUDA support
- Configuration file path: `./confs/moo_wmask_dp_hand_test.conf`
- Supports both testing and training modes through respective dataset files