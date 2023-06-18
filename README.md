# MethaneMapper: Spectral Absorption aware Hyperspectral Transformer for Methane Detection #

Official repository of [MethaneMapper: Spectral Absorption aware Hyperspectral Transformer for Methane Detection]().

We tackle the problem of detecting and localizing methane plumes from hyperspectral imaging data.
Our approach builds upon the DETR model, exploiting the spectral and spatial correlations in the images
to generate a map of potential methane plumes and remove confusers (materials in image background with similar
spectral absorption properties as methane). Methane mapper is a light-weight end-to-end single-stage
CH4 detector which introduces two novel modules: a Spectral Feature Generator and a Query Refiner.
The former generates spectral features from a linear filter that maximizes the CH4-to-noise ratio in the presence of
additive background noise, while the latter integrates these features for decoding.

![Alt text](./docs/architecture.png)

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
