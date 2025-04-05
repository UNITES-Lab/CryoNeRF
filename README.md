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
   conda activate cryonerf
   ```

   To deactivate

   ```bash
   conda deactivate
   ```

3. After setting up `cryonerf` environment, install `tiny-cuda-nn`

   ```bash
   pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
   ```

### Data Preparation

#### Preparation for New Datasets

CryoNeRF can be easily applied to new datasets not used in our paper. When applying CryoNeRF to these new datasets, unprocessed datasets, we follow similar processes to cryoDRGN, which contains:

1. Consensus reconstruction [using cryoSPARC.](https://ez-lab.gitbook.io/cryodrgn/cryodrgn-empiar-10076-tutorial#id-2-consensus-reconstruction-optional)

2. [Preprocess inputs with cryoDRGN](https://ez-lab.gitbook.io/cryodrgn/cryodrgn-empiar-10076-tutorial#id-3-preprocess-inputs) to extract the CTF and pose file from the previous step.

   1. To extract poses of particles as poses.pkl, you can use the following command

      ```bash
      cryodrgn parse_pose_csparc /PATH/TO/YOUR/CS/FILE -D IMAGE_RESOLUTION -o poses.pkl
      ```

      and replace `/PATH/TO/YOUR/CS/FILE` with the path to your cs file, and `IMAGE_RESOLUTION` with your image resolution (e. g., 128 for 128*128 particle images)

   2. To extract ctf of particle images, you can use the following command

      ```bash
      cryodrgn parse_ctf_csparc /PATH/TO/YOUR/CS/FILE -o ctf.pkl
      ```

3. Perform reconstruction with the extracted CTF and pose using CryoNeRF.

After processing, please put 

- `particles.mrcs` that contains all the particle images in a single file
- `ctf.pkl` that contains all ctf parameters for particle images
- `poses.pkl` that contains poses of all images for the dataset

into the same folder and use `--dataset-dir` to specify the directory of the dataset.

#### Dataset Downloading

[EMPIAR-10028](https://www.ebi.ac.uk/empiar/EMPIAR-10028/), [EMPIAR-10049](https://www.ebi.ac.uk/empiar/EMPIAR-10049/), [EMPIAR-10180](https://www.ebi.ac.uk/empiar/EMPIAR-10180/), [EMPIAR-10076](https://www.ebi.ac.uk/empiar/EMPIAR-10076/) can be downloaded from the [EMPIAR website](https://www.ebi.ac.uk/empiar/). [IgG-1D](https://zenodo.org/records/11629428/files/IgG-1D.zip?download=1) and [Ribosembly](https://zenodo.org/records/12528292/files/Ribosembly.zip?download=1) can be downloaded by clicking the link.

### Usage

The commands for CryoNeRF are:

```bash
-h, –help
show this help message and exit

–dataset-dir STR
Root dir for datasets. It should be the parent folder of the dataset you want to reconstruct. (default: ‘’)

–dataset {empiar-10028, empiar-10076, empiar-10049, empiar-10180, IgG-1D, Ribosembly, uniform, cooperative, noncontiguous,}
Which dataset to use. Default as “” for new datasets. (default: ‘’)

–particles {None}|STR|{[STR [STR …]]}
particle support path(s) to mrcs files, the input could be XXX,YYY,ZZZ or XXX. Will use these particle files if specified. (default: None)

–poses {None}|STR|{[STR [STR …]]}
pose support path(s) to pose files, the input could be XXX,YYY,ZZZ or XXX. Will use these pose files if specified. (default: None)

–ctf {None}|STR|{[STR [STR …]]}
ctf support path(s) to ctf files, the input could be XXX,YYY,ZZZ or XXX. Will use these ctf files if specified. (default: None)

–size INT
Size of the volume and particle images. (default: 256)

–batch-size INT
Batch size for training. (default: 1)

–ray-num INT
Number of rays to query in a batch. (default: 8192)

–nerf-hid-dim INT
Hidden dim of NeRF. (default: 128)

–nerf-hid-layer-num INT
Number of hidden layers besides the input and output layer. (default: 2)

–hetero-encoder-type {resnet18, resnet34, resnet50, convnext_small, convnext_base,}
Encoder for deformation latent variable. (default: resnet34)

–hetero-latent-dim INT
Latent variable dim for deformation encoder. (default: 16)

–save-dir STR
Dir to save visualization and checkpoint. (default: experiments/test)

–log-vis-step INT
Number of steps to log visualization. (default: 1000)

–log-density-step INT
Number of steps to log a density map. (default: 10000)

–ckpt-save-step INT
Number of steps to save a checkpoint. (default: 20000)

–print-step INT
Number of steps to print once. (default: 100)

–sign {1,-1}
Sign of the particle images. For datasets used in the paper, this will be automatically set. (default: -1)

–load-to-mem, –no-load-to-mem
Whether to load the full dataset to memory. This can cost a large amount of memory. (default: False)

–seed INT
Whether to set a random seed. Default to not. (default: -1)

–load-ckpt {None}|STR
The checkpoint to load. (default: None)

–epochs INT
Number of epochs for training. (default: 1)

–hetero, –no-hetero
Whether to enable heterogeneous reconstruction. (default: False)

–val-only, –no-val-only
Only val. (default: False)

–first-half, –no-first-half
Whether to use the first half of the data to train for GSFSC computation. (default: False)

–second-half, –no-second-half
Whether to use the second half of the data to train for GSFSC computation. (default: False)

–precision STR
The numerical precision for all the computation. Recommended to set as default at 16-mixed. (default: 16-mixed)

–max-steps INT
The number of training steps. If set, this will supersede num_epochs. (default: -1)

–log-time, –no-log-time
Whether to log the training time. (default: False)

–hartley, –no-hartley
Whether to encode the particle image in hartley space. This will improve heterogeneous reconstruction. (default: True)

–embedding {2d,1d}
Whether to use scalar embeddings for particle images. (default: 2d)
```

### GPU Selection

To select GPUs to use when using the following commands for training and evaluation, please add `CUDA_VISIBLE_DEVICES=XXX` before `python main.py`.

For example, if you only want to use GPU 0 on your server, you can add `CUDA_VISIBLE_DEVICES=0`, or if you want to use GPU 0, 2, 3 for a paralleled training, please add `CUDA_VISIBLE_DEVICES=0,2,3`. If `CUDA_VISIBLE_DEVICES` is not added, the following training and evaluation commands will automatically use all the GPUs available.

### Training

Please refer to [Preparation for New Datasets Section](https://github.com/UNITES-Lab/CryoNeRF?tab=readme-ov-file#preparation-for-new-datasets).  To launch training, an example command would be like:
```bash
python main.py --size 128 --save-dir /PATH/TO/SAVE --dataset-dir /PATH/TO/FOLDER \
	--batch-size 2 --epochs 60 --nerf-hid-dim 128 --nerf-hid-layer-num 3 \
	--hetero --hetero-latent-dim 32 --hetero-encoder-type resnet34
```
If you are using one of `empiar-10049`, `empiar-10028`, `IgG-1D`, `Ribosembly`, `empiar-10180`, `empiar-10076`, `--dataset` option can be set accordingly to set the sign (-1 or 1) of the dataset. Or if you are doing training on new datasets, this option is not needed and the sign of the dataset can be set by `--sign 1` or `--sign 1`.

Another way to perform training is to use a similar format of cryoDRGN command:

```bash
python main.py --size 128 --save-dir /PATH/TO/SAVE \
	--particles /PATH/TO/PARTICLE/FILES --poses /PATH/TO/POSE/FILES --ctf /PATH/TO/CTF/FILES
	--batch-size 2 --epochs 60 --nerf-hid-dim 128 --nerf-hid-layer-num 3 \
	--hetero --hetero-latent-dim 32 --hetero-encoder-type resnet34
```

In this way, `--particles` option accepts (1) a path to the particle file XXX, (2) comma separated paths, e. g., XXX,YYY,ZZZ, or (3) a txt file containing the paths to all particle files, e. g., XXX.txt

### Evaluation

To run evaluation using a checkpoint, you only need to add `--val-only` and `--load-ckpt /PATH/TO/YOUR/CKPT` after your training command.
This will run evaluation to generate the particle embeddings of all the particle images, embed the particle embeddings using UMAP, divide UMAP embeddings into six clusters and produce one reconstruction for the center of each cluster.
