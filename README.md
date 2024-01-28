# DiffInfinite 

We present DiffInfinite, a hierarchical diffusion model that generates arbitrarily large histological images while preserving long-range correlation structural information. Our approach first generates synthetic segmentation masks, subsequently used as conditions for the high-fidelity generative diffusion process. The proposed sampling method can be scaled up to any desired image size while only requiring small patches for fast training. Moreover, it can be parallelized more efficiently than previous large-content generation methods while avoiding tiling artefacts. The training leverages classifier-free guidance to augment a small, sparsely annotated dataset with unlabelled data. Our method alleviates unique challenges in histopathological imaging practice: large-scale information, costly manual annotation, and protective data handling. The biological plausibility of DiffInfinite data is validated in a survey by ten experienced pathologists as well as a downstream segmentation task. Furthermore, the model scores strongly on anti-copying metrics which is beneficial for the protection of patient data.

Check out some examples on the [Project Website](https://marcoaversa.github.io/diffinfinite/).

![Example Image](images/examples/synth_examples.png)

Click on the link below to run the Jupyter Notebook on Google Colab (set ```colab=True``` in the first cell):

<a target="_blank" href="https://colab.research.google.com/github/diffinfinite/diffinfinite/blob/master/main.ipynb">

  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>

</a>


## Run Locally

Create a conda environment using the requirements file.

```
conda env create -n env_name -f environment.yaml
conda activate env_name
```

Download and unzip the models (```n_classes``` can be 5 or 10):

```
python download.py --n_classes=5
```

Usage example in [Jupyter Notebook](main.ipynb). 


## Synthetic data visualisation

In ```./results```, we share some synthetic data generated with the model. 

In ```./results/large``` we show 2048x2048 images for different ω.

In ```./results/patches``` we show 512x512 images for different ω.

## Evaluation Metrics

Save your generated data and your real data as ```.png``` in ```generated_dir``` and ```real_dir``` respectively.
To calculate the [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500) and the [Inception Score](https://arxiv.org/pdf/1606.03498.pdf) use the [torch-fidelity](https://github.com/toshas/torch-fidelity) package and run

```
fidelity --gpu 0 --isc --input1 generated_dir
fidelity --gpu 0 --fid --input1 generated_dir --input2 real_dir
```

For [Improved Precision and Improved Recall](https://arxiv.org/abs/1904.06991), clone the repository [improved-precision-and-recall-metric-pytorch](https://github.com/blandocs/improved-precision-and-recall-metric-pytorch) and run

```
python main.py --cal_type precision_and_recall --generated_dir generated_dir --real_dir real_dir
```

To evaluate your generated data wrt. the [Authenticity](https://arxiv.org/abs/2102.08921) and the $C_{T}$ [score](https://arxiv.org/abs/2004.05675), clone the repository [fls](https://github.com/marcojira/fls) by

```
cd diffinfinite
git clone https://github.com/marcojira/fls.git
pip install git+https://github.com/openai/CLIP.git
```

and run 

```
python eval_privacy.py --train_dir train_dir --test_dir test_dir --generated_dir generated_dir 
```

## Citations

```
@inproceedings{
aversa2023diffinfinite,
title={DiffInfinite: Large Mask-Image Synthesis via Parallel Random Patch Diffusion in Histopathology},
author={Marco Aversa and Gabriel Nobis and Miriam H{\"a}gele and Kai Standvoss and Mihaela Chirica and Roderick Murray-Smith and Ahmed Alaa and Lukas Ruff and Daniela Ivanova and Wojciech Samek and Frederick Klauschen and Bruno Sanguinetti and Luis Oala},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2023},
url={https://openreview.net/forum?id=QXTjde8evS}
}
```
