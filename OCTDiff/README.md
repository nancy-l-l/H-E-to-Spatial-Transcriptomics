# **OCTDiff: Bridged Diffusion Model for Portable OCT Super-Resolution and Enhancement**
https://neurips.cc/virtual/2025/loc/san-diego/poster/117693

<img width="3240" height="1030" alt="pipeline (1)" src="https://github.com/user-attachments/assets/feac6643-bdc0-4959-a9ac-18753b90f85e" />

## **Requirements**
```bash
conda env create -f environment.yml
```

## **Training Script**
For running the training / inference script, see [template](https://github.com/AI4VSLab/OCTDiff/blob/main/Template-shell.sh)

## **Dataset Curation**
The train/val/test set should by default in this format:

```text
/Dataset_Root
├── /Train
│   ├── /LowRes
│   └── /HiRes
├── /Val
│   ├── /LowRes
│   └── /HiRes
└── /Test
    ├── /LowRes
    └── /HiRes
```

The [data_splitter](https://github.com/AI4VSLab/OCTDiff/blob/main/dataset_splitter.py)  is useful to curate such path structure. Please consider modifying the customized dataloader otherwise. 
To implement loss function with weights, a .csv file is needed.

## **Model Training**
To switch on / off the ANA module, parse False here:
```
    params:
      ana_on: True
```
Multiscale Cross Attention is inherently integrated, based on [CrossFusion](https://github.com/RustinS/CrossFusion) and [x-transformer](https://pypi.org/project/x-transformers/).

## **Acknowledgements**
Our code is based on [BBDM](https://github.com/xuekt98/BBDM?tab=readme-ov-file) and OpenAI [Guided Diffusion](https://github.com/openai/guided-diffusion/tree/0ba878e517b276c45d1195eb29f6f5f72659a05b), [Improved Diffusion](https://github.com/openai/improved-diffusion/tree/main).


## **Citation**
```
@inproceedings{tian2025octdiff,
  title={OCTDiff: Bridged Diffusion Model for Portable OCT Super-Resolution and Enhancement},
  author={Tian, Ye and McCarthy, Angela and Gomide, Gabriel and Liddle, Nancy and Golebka, Jedrzej and Chen, Royce and Liebmann, Jeff and Thakoor, Kaveri A},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```
