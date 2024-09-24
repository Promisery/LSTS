# LSTS: Periodicity Learning via Long Short-term Temporal Shift for Remote Physiological Measurement

This repository is the official implementation of the paper **LSTS: Periodicity Learning via Long Short-term Temporal Shift for Remote Physiological Measurement**. Currently, a demo is provided. The full code will be released after the acceptance of the paper.

## Dependencies
```
conda create -n lsts python=3.9
conda activate lsts
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pandas scipy einops matplotlib seaborn notebook -c conda-forge
pip install timm entmax
```

## Model & Data

The source code of the proposed LSTS model is provided in `models/lsts.py`. The Periodic Channel Shift mechanism is implemented in `PeriodicShift` class. The MPOS and TSAug techniques are implemented in the `preprocess` function of `LSTS` class.

We provide a model trained on the RLAP-rPPG dataset in `weights.pt`, and a sample data in `data.pt`. The data is a publicly available sample data in the RLAP-rPPG dataset, which can be obtained [here](https://github.com/KegangWangCCNU/PhysRecorder/tree/main/Example/v01). Note that the sample data itself is not part of the RLAP dataset.

Scripts for model training and validation will be released after the acceptance of the paper.

## Visualization

Please run `main.ipynb` to check the visualized results on the sample data.