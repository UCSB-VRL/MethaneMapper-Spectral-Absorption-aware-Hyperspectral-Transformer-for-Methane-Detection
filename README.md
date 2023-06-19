# MethaneMapper: Spectral Absorption aware Hyperspectral Transformer for Methane Detection #

MethaneMapper is a fast and efficient deep learning based solution for methane detection from airborne hyperspectral imagery. MethaneMapper introduces a novel end-to-end spectral absorption wavelength aware transformer network to detect the emissions. We also introduce the largest public dataset called Methane HotSpot dataset (MHS) for methane detection. This repository contains source code for MethaneMapper, scripts to download training dataset and visualize ground truth.

### [**MethaneMapper: Spectral Absorption aware Hyperspectral Transformer for Methane Detection**](https://openaccess.thecvf.com/content/CVPR2023/papers/Kumar_MethaneMapper_Spectral_Absorption_Aware_Hyperspectral_Transformer_for_Methane_Detection_CVPR_2023_paper.pdf)
[Satish Kumar*](https://www.linkedin.com/in/satish-kumar-81912540/), [Ivan Arevalo](), [A S M Iftekhar](), [B S Manjunath](https://vision.ece.ucsb.edu/people/bs-manjunath).

Official repository of our [**CVPR 2023 (Highlights)**](https://openaccess.thecvf.com/content/CVPR2023/papers/Kumar_MethaneMapper_Spectral_Absorption_Aware_Hyperspectral_Transformer_for_Methane_Detection_CVPR_2023_paper.pdf) paper.


![Alt text](./docs/architecture.png)


This repository includes:
* Source code of MethaneMapper.
* Pre-trained weights for methane plume bounding box detector and segmentation mask
* Scripts to download MHS dataset
* Online tool to visualize MHS dataset
* Code for custom data preparation for training/testing
* Code for mapping ground truth masks from CarbonMapper to AVIRIS-NG flightline
* Annotation generator to read-convert mask annotation into json.


![supported versions](https://img.shields.io/badge/python-(3.8--3.10)-brightgreen/?style=flat&logo=python&color=green)
![Library](https://img.shields.io/badge/Library-Pytorch-blue)
![GitHub license](https://img.shields.io/cocoapods/l/AFNetworking)


## Usage

### Data preparation

```bash
# PUT CODE HERE
```

### Training

```bash
# PUT CODE HERE
```

### Evaluation

```bash
# PUT CODE HERE
```

## Methane Hot Spots (MHS)

In addition, we introduce Methane Hot Spots (MHS), a large-scale dataset of methane
plume segmentation masks for over 1200 AVIRIS-NG flight
lines collected by JPL from 2015-2022. It contains over 4000 methane plume
sites covering terrain from 6 states: California, Nevada, New Mexico, Colorado,
Midland Texas, and Virginia. MHS dataset is available for browsing, visualization, and download in [BisQue]().

## For developers
Pre-Commit
In order to provide some degree of uniformity in style, we can use the pre-commit tool to clean up source files prior to being committed. Pre-Commit runs a number of plugins defined in .pre-commit-config.yaml. These plugins enforce coding style guidelines.

Install pre-commit by following the instructions here: https://pre-commit.com/#install

Linux:
```
pip install pre-commit
```

Once pre-commit is installed, install the git hooks by typing:
```
# In git repo root dir
pre-commit install
```
Now, whenver you commit code, pre-commit will clean it up before it is committed. You can then add the cleaned-up code and commit it. This enforces coding standards and consistency across developers.

## Citation
Please cite our work if used in your research.
```bash
# PLACE BIBTEX CITATION
```

## License
MethaneMapper is released under the MIT license. Please see the [LICENSE](./LICENSE) file for more information.
