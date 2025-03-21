# CryoNeRF

CryoNeRF is a computational tool for homogeneous and heterogeneous (conformational and compositional) cryo-EM reconstruction in Euclidean 3D space.

Copyright (C) 2025 Huaizhi Qu, Xiao Wang, Yuanyuan Zhang, Sheng Wang, William Stafford Noble and Tianlong Chen.

License: GPL v3. (If you are interested in a different license, for example, for commercial use, please contact us.)

Contact: Tianlong Chen (tianlong@cs.unc.edu)

For technical problems or questions, please reach to Huaizhi Qu (huaizhiq@cs.unc.edu).

### Citation

Huaizhi Qu, Xiao Wang, Yuanyuan Zhang, Sheng Wang, William Stafford Noble & Tianlong Chen. CryoNeRF: reconstruction of homogeneous and heterogeneous cryo-EM structures using neural radiance field. Biorxiv, 2025. Paper

```

@misc{qu_cryonerf:_2025,
	title = {{CryoNeRF}: reconstruction of homogeneous and heterogeneous cryo-{EM} structures using neural radiance field},
	shorttitle = {{CryoNeRF}},
	url = {https://www.biorxiv.org/content/10.1101/2025.01.10.632460v1},
	doi = {10.1101/2025.01.10.632460},
	language = {en},
	urldate = {2025-02-04},
	publisher = {bioRxiv},
	author = {Qu, Huaizhi and Wang, Xiao and Zhang, Yuanyuan and Wang, Sheng and Noble, William Stafford and Chen, Tianlong},
	month = jan,
	year = {2025},
}

```

### Checkpoints & Files

The checkpoints for all experiments in our paper and the reconstructions can be found at https://doi.org/10.5281/zenodo.14602456.

### Installation

#### 1. Clone the repository to your computer

```bash
git clone https://github.com/UNITES-Lab/CryoNeRF.git && cd CryoNeRF
```

#### 2. Configure Python environment for CryoNeRF

1. Install conda at https://conda-forge.org/

2. Set up the Python environment via yml file

   ```bash
   conda env create -f environment.yml
   ```

   and activate the environmentt

   ```bash
   conda activate CryoNeRF
   ```

   To deactivate

   ```bash
   conda deactivate
   ```

3. After setting up `CryoNeRF` environment, install `tiny-cuda-nn`

   ```bash
   pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
   ```

### Data Preparation

#### Preparation for New Datasets

CryoNeRF can be easily applied to new datasets not used in our paper. When applying CryoNeRF to these new datasets, unprocessed datasets, we follow similar processes to cryoDRGN, which contains:

1. Consensus reconstruction [using cryoSPARC.](https://ez-lab.gitbook.io/cryodrgn/cryodrgn-empiar-10076-tutorial#id-2-consensus-reconstruction-optional)
2. [Preprocess inputs with cryoDRGN](https://ez-lab.gitbook.io/cryodrgn/cryodrgn-empiar-10076-tutorial#id-3-preprocess-inputs) to extract the CTF and pose file from the previous step.
3. Perform reconstruction with the extracted CTF and pose using CryoNeRF.

After processing, please put 

- `particles.mrcs` that contains all the particle images in a single file
- `ctf.pkl` that contains all ctf parameters for particle images
- `poses.pkl` that contains poses of all images for the dataset

into the same folder and use `--dataset-dir` to specify the directory of the dataset.

#### Dataset Downloading

[EMPIAR-10028](https://www.ebi.ac.uk/empiar/EMPIAR-10028/), [EMPIAR-10049](https://www.ebi.ac.uk/empiar/EMPIAR-10049/), [EMPIAR-10180](https://www.ebi.ac.uk/empiar/EMPIAR-10180/), [EMPIAR-10076](https://www.ebi.ac.uk/empiar/EMPIAR-10176/) can be downloaded from the [EMPIAR website](https://www.ebi.ac.uk/empiar/). [IgG-1D](https://zenodo.org/records/11629428/files/IgG-1D.zip?download=1) and [Ribosembly](https://zenodo.org/records/12528292/files/Ribosembly.zip?download=1) can be downloaded by clicking the link.

### Usage

The commands for CryoNeRF are:

```bash
Arguments of CryoNeRF:

  -h, --help
      Show this help message and exit.
  --dataset-dir STR
      Root directory for datasets. It should be the parent folder of the dataset you want to reconstruct. (required)
  --dataset {empiar-10028,empiar-10076,empiar-10049,empiar-10180,IgG-1D,Ribosembly}
      Specify which dataset to use. (default: "")
  --size INT
      Size of the volume and particle images. (default: 256)
  --batch-size INT
      Batch size for training. (default: 1)
  --ray-num INT
      Number of rays to query in a batch. (default: 8192)
  --nerf-hid-dim INT
      Hidden dimension of NeRF. (default: 128)
  --nerf-hid-layer-num INT
      Number of hidden layers besides the input and output layer. (default: 2)
  --hetero-encoder-type {resnet18,resnet34,resnet50,convnext_small,convnext_base}
      Encoder type for deformation latent variable. (default: resnet18)
  --hetero-latent-dim INT
      Latent variable dimension for deformation encoder. (default: 16)
  --save-dir STR
      Directory to save visualization and checkpoint. (default: experiments/test)
  --log-vis-step INT
      Number of steps between logging visualization. (default: 1000)
  --log-density-step INT
      Number of steps between logging density maps. (default: 10000)
  --print-step INT
      Number of steps between printing logs. (default: 100)
  --sign {1,-1}
      Sign of the particle images. For datasets used in the paper, this will be automatically set. (default: -1)
  --seed INT
      Random seed. Defaults to not setting one. (default: -1)
  --load-ckpt {None}|STR
      Path to the checkpoint to load. (default: None)
  --epochs INT
      Number of training epochs. (default: 1)
  --hetero, --no-hetero
      Enable or disable heterogeneous reconstruction. (default: False)
  --val-only, --no-val-only
      Run validation only. (default: False)
  --first-half, --no-first-half
      Use the first half of the data for GSFSC computation. (default: False)
  --second-half, --no-second-half
      Use the second half of the data for GSFSC computation. (default: False)
  --precision STR
      Numerical precision for computations. Recommended to use "16-mixed". (default: 16-mixed)
  --max-steps INT
      Maximum number of training steps. If set, this overrides the number of epochs. (default: -1)
  --log-time, --no-log-time
      Log the training time. (default: False)
  --hartley, --no-hartley
      Encode the particle image in Hartley space for improved heterogeneous reconstruction. (default: True)
```

### Training

To launch training, an example command would be like:
```bash
python main.py --size 128 --save-dir /PATH/TO/SAVE --dataset-dir /PATH/TO/FOLDER --dataset empiar-10076  \
	--batch-size 2 --epochs 60 --nerf-hid-dim 128 --nerf-hid-layer-num 3 \
	--hetero --hetero-latent-dim 32 --hetero-encoder-type resnet34
```
And `--dataset` could be either one of `empiar-10049`, `empiar-10028`, `IgG-1D`, `Ribosembly`, `empiar-10180`, `empiar-10076`.

### Evaluation

To run evaluation using a checkpoint, an example command is:
```bash
python main.py --size 128 --save-dir /PATH/TO/SAVE --dataset-dir /PATH/TO/FOLDER --dataset empiar-10076  \
	--batch-size 2 --epochs 60 --nerf-hid-dim 128 --nerf-hid-layer-num 3 \
	--hetero --hetero-latent-dim 32 --hetero-encoder-type resnet34 --val-only --load-ckpt /PATH/TO/CKPT
```
This will run evaluation to generate the particle embeddings of all the particle images, embed the particle embeddings using UMAP, divide UMAP embeddings into six clusters and produce one reconstruction for the center of each cluster.
