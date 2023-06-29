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
./training_dataset2020 OR ./training_dataset16171819
|_ mf_output
|_ mf_output_norm
|_ mf_tiles #matched filter tiles
|_ mf_tiles_no_norm
|_ rdata_tiles #raw 90 bands from hyperspectral image
|_ rgb_tiles #reconstructed RGB image from hyperspectral image
```
No create soft links to the "./data/train_dataset" folder as follows to avoud duplication of dataset (~size 10 TB):
1. matched_filter output tiles to mf_tiles folder
```
ln -s ./training_dataset2020/mf_tiles methanemapper/data/train_dataset/train/mf_tiles
ln -s ./training_dataset2020/mf_tiles methanemapper/data/train_dataset/val/mf_tiles
```
2. Link the raw 90 bands tiles to rgb_tiles folder
```
ln -s ./training_dataset2020/rgb_tiles methanemapper/data/train_dataset/train/rgb_tiles
ln -s ./training_dataset2020/rgb_tiles methanemapper/data/train_dataset/val/rgb_tiles
```
3. Link the reconstructed RGB image tiles to rdata_tiles folder
```
ln -s ./training_dataset2020/rdata_tiles methanemapper/data/train_dataset/train/rdata_tiles
ln -s ./training_dataset2020/rdata_tiles methanemapper/data/train_dataset/val/rdata_tiles
```

#### Training methane plume bounding box detector 
```
./plume_box_run.sh
```
#### Fine-tuning plume segmentation mask detector 
```
./plume_mask_run.sh
```
