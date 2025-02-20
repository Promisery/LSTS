# LSTS: Periodicity Learning via Long Short-term Temporal Shift for Remote Physiological Measurement

This repository is the official implementation of the paper [**LSTS: Periodicity Learning via Long Short-term Temporal Shift for Remote Physiological Measurement**](https://ieeexplore.ieee.org/abstract/document/10870326/).

## Dependencies
```
conda create -n rppg python=3.9
conda activate rppg
pip install -r requirements.txt
# optional
pip install notebook
```

## Train & Validation

1. Change **Path/to/XXXX/dataset** and **Path/to/cache/directory** to the actual paths in *preprocess.py*
2. Run `python preprocess.py`
3. Change **Path/to/XXXX/dataset** and **Path/to/cache/directory** to the actual paths in the config files in *./configs/*
4. Run`python ./train.py --config ./configs/lsts_xxxx.yaml --split_idx idx` where `xxxx` is the name of the dataset and `idx` is the split index ranging from 0 to 4.
5. The training logs are managed using [Weights & Biases](https://wandb.ai/). Visit the website to check the results.

## Citation

```
@article{lsts,
    author={Jiang, Titong and Ma, Yuan and Li, Jiaqi and Dong, Qing and Ji, Xuewu and Liu, Yahui},
    journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
    title={LSTS: Periodicity Learning via Long Short-term Temporal Shift for Remote Physiological Measurement}, 
    year={2025},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TCSVT.2025.3538474}
}
```


## Credit

This project is heavily dependent on the following projects. If you find them useful, please give them a star.

[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)
[TPS](https://github.com/MartinXM/TPS)
[Heartpy](https://github.com/paulvangentcom/heartrate_analysis_python)
[NeuroKit2](https://github.com/neuropsychology/NeuroKit)
[PhysBench](https://github.com/KegangWangCCNU/PhysBench/)
[minGPT](https://github.com/karpathy/minGPT)