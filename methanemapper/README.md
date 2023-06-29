### MethaneMapper Training code
MethaneMapper repository structure is similar to DETR repository from META. The training policy involves first training for bounding [box detection](https://github.com/UCSB-VRL/MethaneMapper-Spectral-Absorption-aware-Hyperspectral-Transformer-for-Methane-Detection/blob/main/methanemapper/plume_box_run.sh) of methane plumes, followed by fine-tuning for methane plume [segmentation mask](https://github.com/UCSB-VRL/MethaneMapper-Spectral-Absorption-aware-Hyperspectral-Transformer-for-Methane-Detection/blob/main/methanemapper/plume_mask_run.sh).

There are no extra compiled components in MethaneMapper and package dependencies are minimal mentioned in the [requirements.txt](https://github.com/UCSB-VRL/MethaneMapper-Spectral-Absorption-aware-Hyperspectral-Transformer-for-Methane-Detection/blob/main/requirements.txt). Codebase is very simple to use. 

#### Steps:
1.  Clone the repository
```
git clone https://github.com/UCSB-VRL/MethaneMapper-Spectral-Absorption-aware-Hyperspectral-Transformer-for-Methane-Detection.git
```
2.  Install dependencies
```
pip install -r requirements.txt
```
3.  That's all! It should be good to train and evaluate MethaneMapper

#### Data preparation
Download MHS dataset by following these steps. [DATA_DOWNLOAD](https://github.com/UCSB-VRL/MethaneMapper-Spectral-Absorption-aware-Hyperspectral-Transformer-for-Methane-Detection/tree/main/mhs_dataset)

Directory structure from download :
```

```

