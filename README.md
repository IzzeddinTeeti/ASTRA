# ASTRA: A Scene-aware TRAnsformer-based model for trajectory prediction
This is the official PyTorch implmentation of the paper **ASTRA: A Scene-aware TRAnsformer-based model for trajectory prediction**.

## Model Architecture 🏗️

<p align="center"><img width="100%" src="src/Architecture.png"/></p>

## Table of Contents 📋
- [ASTRA: A Scene-aware TRAnsformer-based model for trajectory prediction](#astra-a-scene-aware-transformer-based-model-for-trajectory-prediction)
  - [Model Architecture 🏗️](#model-architecture-️)
  - [Table of Contents 📋](#table-of-contents-)
  - [Getting Started 🚀](#getting-started-)
    - [Environment](#environment)
    - [Installation](#installation)
  - [Repository Structure 📂](#repository-structure-)
  - [Download \& Process Datasets](#download--process-datasets)
  - [Download Pretrained Models](#download-pretrained-models)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Visualizations 📊](#visualizations-)
  - [Results 📈](#results-)
  - [Citation 📖](#citation-)

## Getting Started 🚀
### Environment
- Tested OS: Linux
- Python >= 3.9
- PyTorch == 2.0
### Installation
  1. Clone the repository to your local machine.
  2. Navigate to the project directory: `cd ASTRA`
  3. Create the environment and install the requirements using `source scripts/build_env_data_process.sh`

## Repository Structure 📂
The repository is structured as follows:

```
📦 ASTRA
 ┣ 📂 configs
 ┃ ┣ 📜 eth.yaml
 ┃ ┗ 📜 pie.yaml
 ┣ 📂 data
 ┣ 📂 datasets
 ┃ ┣ 📂 eth_ucy
 ┃ ┗ 📂 PIE
 ┣ 📂 models
 ┃ ┣ 📜astra_model.py
 ┃ ┗ 📜keypoint_model.py
 ┣ 📂 scripts
 ┣ 📂 utils
 ┣ 📂 visualization
 ┃ ┣ 📜 gradcam_visualizer_ETH.ipynb
 ┃ ┣ 📜 gradcam_visualizer_PIE.ipynb
 ┃ ┣ 📜 traj_visualizer_ETH.ipynb
 ┃ ┗ 📜 traj_visualizer_PIE.ipynb
 ┣ 📜 main.py
 ┣ 📜 train_ETH.py
 ┣ 📜 train_PIE.py 
 ┗ 📜 README.md (You are here!)
```

## Download & Process Datasets
> **ETH-UCY Dataset (Bird's Eye View (BEV))**
* Download ETH dataset, videos and annotations, and process them using:
```bash
bash ./scripts/down_process_eth.bash
```

> **PIE Dataset (Ego Vehicle View (EVV))**
* Download PIE dataset, videos and annotations, and process them using:
```bash
bash ./scripts/down_process_PIE.bash
```

* The datasets will be downloaded so that its structure is like the one shown above.

## Download Pretrained Models
* Download pretrained U-Net Keypoint Embedding model using:
```bash
bash ./scripts/down_pretrained_unet_models.bash
``` 
*(downloads pretrained unet weights in folder: `./pretrained_unet_weights/`. By default, these pretrained weights are used in training the ASTRA model)*

* Download pretrained ASTRA models using:
```bash
bash ./scripts/down_pretrained_astra_models.bash
```
*(downloads pretrained ASTRA model weights in folder: `./pretrained_astra_weights/`)*

## Training 
> **ETH-UCY Dataset (BEV)**

To train the model on ETH dataset, run the following command:
```
python main.py --config_file ./configs/eth.yaml
```
*(NOTE: The above command will train the model on **eth** subset. To train on other subsets, please change the config file accordingly.)*

**OPTIONAL**: Pretraining U-Net based Keypoint Embedding model 
*By default, the U-Net based Keypoint Embedding model loads the pretrained embedding weights from `'./pretrained_unet_weights/eth_unet_model_best.pt` but if you want to pretrain the U-Net based Keypoint Embedding model, run the following command:*
```
python ./scripts/pretrain_unet_eth.py
```


> **PIE Dataset (EVV)**

To train the model on PIE dataset, run the following command:
```
python main.py --config_file ./configs/pie.yaml
```

**OPTIONAL**: Pretraining U-Net based Keypoint Embedding model 
*By default, the U-Net based Keypoint Embedding model takes the pretrained embedding weights from `'./pretrained_unet_weights/pie_unet_model_best.pt` but if you want to pretrain the U-Net based Keypoint Embedding model, run the following command:*
```
python ./scripts/pretrain_unet_pie.py
```

## Evaluation
To ensure reproducibility, we have provided the pretrained models in `./pretrained_astra_weights/` folder. 
> **ETH-UCY Dataset (BEV)**
* To evaluate the downloaded pretrained models on ETH dataset, run the following command:
```python
python test_ETH.py --config_file ./configs/eth.yaml
```

> **PIE Dataset (EVV)**
* To evaluate the downloaded pretrained models on PIE dataset, run the following command:
```python
python test_PIE.py --config_file ./configs/pie.yaml
```

## Visualizations 📊
<p align="center">
    <img width="100%" src="visualization/gradcam.png"/>
</p>
<table align="center" style="width:100%; table-layout:fixed;">
    <tr>
        <td><img style="width: 100%; height: auto;" src="visualization/eth.png"/></td>
        <td><img style="width: 100%; height: auto;" src="visualization/eth_quali.png"/></td>
    </tr>
</table>

* Grad-CAM Visualizers are available at:
  * `./visualization/gradcam_visualizer_ETH.ipynb`
  * `./visualization/gradcam_visualizer_PIE.ipynb`
* Trajectory Visualizers are available at:
  * `./visualization/traj_visualizer_ETH.ipynb`
  * `./visualization/traj_visualizer_PIE.ipynb`

## Results 📈
| Dataset | ADE   | FDE   | CADE | CFDE  | ARB   | FRB   |
|---------|-------|-------|------|-------|-------|-------|
| ETH     | 0.47  | 0.82  | N/A  | N/A   | N/A   | N/A   |
| HOTEL   | 0.29  | 0.56  | N/A  | N/A   | N/A   | N/A   |
| UNIV    | 0.55  | 1.00  | N/A  | N/A   | N/A   | N/A   |
| ZARA1   | 0.34  | 0.71  | N/A  | N/A   | N/A   | N/A   |
| ZARA2   | 0.24  | 0.41  | N/A  | N/A   | N/A   | N/A   |
| PIE     | N/A   | N/A   | 9.91 | 22.42 | 18.32 | 17.07 |


## Citation 📖

If you find this repository useful for your research, please consider giving a star ⭐ and a citation

```
@inproceedings{,
    title={},
    author={},
    booktitle={},
    year={}
}
```
