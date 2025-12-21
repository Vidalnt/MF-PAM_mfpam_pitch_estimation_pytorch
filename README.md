# MF-PAM_pytorch

![overall_architecture](https://github.com/Woo-jin-Chung/mfpam-pitch-estimation-pytorch/assets/76720656/9771d5ca-9993-4e84-ae13-d6d7481abf0f)

This repo is the official Pytorch implementation of ["MF-PAM: Accurate Pitch Estimation through Periodicity Analysis and Multi-level Feature Fusion"](https://arxiv.org/abs/2306.09640) accepted in INTERSPEECH 2023.

In the paper, we predicted the quantized F0 with **BCELoss**.  
However, you can also directly estimate the F0 value with **L1 Loss**, which provides more accurate VAD performance. You may use `train_direct.py` or `model_direct.py` for direct F0 estimation.

## Dependencies
```bash
pip install -r requirements.txt
```

## Dataset Preparation
This implementation reads directly from a data directory. You no longer need to generate JSON files.

### Folder Structure
Your dataset should be organized as follows:
```text
data_path/
├── train/
│   ├── sample1.wav
│   ├── sample1.pv
│   ├── sample2.wav
│   └── sample2.pv
└── test/
    ├── test_sample1.wav
    └── test_sample1.pv
```

*   **`.wav` files**: Clean mono audio (16kHz recommended).
*   **`.pv` files**: Text files containing the ground truth pitch in **Hz** (one value per frame).
*   **Augmentation**: Noise and reverberation are added **on-the-fly** during training. You do not need to provide pre-mixed noisy files.

## Training
To start training, simply provide the path to the root data directory.

### 1. Standard Quantized F0 (Paper Version)
Uses 360 frequency bins and BCELoss.
```bash
python train.py --checkpoint_path /path/to/save/checkpoint --data_path /path/to/dataset/root --rir_dir /path/to/rir_list/
```

### 2. Direct F0 Estimation
Preferred for better VAD performance using L1 Loss.
```bash
python train_direct.py --checkpoint_path /path/to/save/checkpoint --data_path /path/to/dataset/root --rir_dir /path/to/rir_list/
```

### Arguments:
*   `--data_path`: Path to the root folder containing `train/` and `test/` subfolders.
*   `--rir_dir`: (Optional) Path to a directory containing Room Impulse Responses for reverberation augmentation.

## Publications
```bibtex
@inproceedings{chung23_interspeech,
  author={Woo-Jin Chung and Doyeon Kim and Soo-Whan Chung and Hong-Goo Kang},
  title={{MF-PAM: Accurate Pitch Estimation through Periodicity Analysis and Multi-level Feature Fusion}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={4499--4503},
  doi={10.21437/Interspeech.2023-2487}
}
```