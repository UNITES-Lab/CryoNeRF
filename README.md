# CryoNeRF

### Training

To launch training, an example command would be like:
`python main.py --size 128 --save-dir /PATH/TO/SAVE --dataset empiar-10076 --enable-dfom --dfom-latent-dim 32 --batch-size 2 --epochs 60 --nerf-hid-dim 128 --nerf-hid-layer-num 3 --simple --dfom-encoder-type resnet34 --hartley`
And `--dataset` could be either one of `empiar-10049`, `empiar-10028`, `IgG-1D`, `Ribosembly`, `empiar-10180`, `empiar-10076`.

### Evaluation

To run evaluation using a checkpoint, an example command is:
`python main.py --size 128 --save-dir /PATH/TO/SAVE --dataset empiar-10076 --enable-dfom --dfom-latent-dim 32 --batch-size 2 --epochs 60 --nerf-hid-dim 128 --nerf-hid-layer-num 3 --simple --dfom-encoder-type resnet34 --hartley --val-only --load-ckpt /PATH/TO/CKPT`.
This will run evaluation to generate the particle embeddings of all the particle images, embed the particle embeddings using UMAP, divide UMAP embeddings into six clusters and produce one reconstruction for the center of each cluster.
