# Paper reproduction: Tighter Variational Bounds are Not Necessarily Better
Part of the [Advanced Topics in Machine Learning course of 2021](https://www.cs.ox.ac.uk/teaching/courses/2020-2021/advml/) as reproducibility challenge

Authors alphabetically: [Amine M'Charrak](https://github.com/mcharrak), [Vít Růžička](https://github.com/previtus), [Sangyun Shin](https://github.com/yunshin), [Madhu Vankadari](https://github.com/madhubabuv/) @ cs.ox.ac.uk

<p align="center">
<a href="https://github.com/madhubabuv/TightIWAE/blob/main/_report/ATML_poster_final.pdf"><img src="https://raw.githubusercontent.com/madhubabuv/TightIWAE/master/_report/ATML_poster_final.gif" width="700"></a>
</p>


## Description:

Reproduction of the paper [1] which implements PIWAE (partially importance weighted auto-encoder), MIWAE (multiply importance weighted auto-encoder) and the CIWAE (combination importance weighted autoencoder) as an extension to the IWAE (importance weighted auto-encoder) model of [2].

We additionally extend the experiments with a **larger tested model** (with roughly doubled number of parameters), with **evaluation on the Omniglot dataset** (and also check cross-dataset generalization between Omniglot and MNIST) and finally propose an edited version of the **CIWAE model with learnable parameter Beta** (which results in a more robust model in terms of the Beta parameter initialization.

More details can be seen in our [report](https://github.com/madhubabuv/TightIWAE/blob/main/_report/ATML_report_final.pdf) or in our [poster](https://github.com/madhubabuv/TightIWAE/blob/main/_report/ATML_poster_final.pdf).

## Run instructions:

### Training:

Start with inspecting:

`python miwae_simplified.py --help`

To **reproduce the P/W/C-IWAE** methods on MNIST run:

```
# PIWAE (M=8,k=8)
python miwae_simplified.py --piwae --k 8 --M 8 --dataset_name mnist

# MIWAE (M=8,k=8)
python miwae_simplified.py --miwae --k 8 --M 8 --dataset_name mnist

# IWAE (M=1,k=64)
python miwae_simplified.py --miwae --k 64 --M 1 --dataset_name mnist

# CIWAE (Beta=0.5)
python miwae_simplified.py --ciwae --beta 0.5 --dataset_name mnist

# VAE (M=1,k=1)
python miwae_simplified.py --miwae --k 1 --M 1 --dataset_name mnist
```

To use our proposed learnable Beta CIWAE (with initial Beta 0.5) run:
```
python miwae_simplified.py --ciwae-beta --beta 0.5 --dataset_name mnist
```

To load an older experiment (both model and history of losses) add `--cont` to the line. Furthermore if you need to run several repetitions of the same method settings, you can use `--repetition N`, which will mark the newly saved logs and models with this N.

All runs save their history into `logs/*.h5` (such as `logs/log_MIVAE_M8_k8.h5` for MIWAE(8,8)) and full models into `logs/*.pt` (such as `logs/model_CIVAE_beta0.5_repeat10.pt` for CIWAE(Beta=0.5) with repetition set to 10).

### Plotting and evaluation:

To plot the saved history files please inspect the functions at `plotting_utils/plot_1_reference.py` and replace the paths to your experiment histories. Alternatively to generate reconstruction images use the script `python visualize_models.py --dataset_name mnist` while adjusting paths to your desired loaded models. With this code we also calculate the SSIM metric between the input images and the reconstructions (average across the whole test set). Note that the selected evaluation dataset with `--dataset_name` can differ from the one used to train the model (we argue that this can be used as a generalization ability proxy for tested methods across different datasets - namely those as similar as MNIST and Omniglot).

## Results

**Figure 5. reconstruction from the original paper [1].** Shows the training progress of the newly tested methods:

<div align="center">
  <table>
    <tr>
      <td><img src="https://raw.githubusercontent.com/madhubabuv/TightIWAE/master/plotting_utils/1_reference__iwae64.png" width="400"></td>
      <td><img src="https://raw.githubusercontent.com/madhubabuv/TightIWAE/master/plotting_utils/1_reference__iwae5000.png" width="400"></td>
    </tr>
    <tr>
      <td align="center">(a) IWAE-64</td>
      <td align="center">(b) log p̂(x) ( = IWAE-5000 )</td>
    </tr>
    <tr>
      <td colspan=2>Figure 1: The metrics of IWAE-64 and log p̂(x) of referential models trained on the MNIST dataset. We used rolling window of 10 for better clarity of the IWAE-64 plot, also the figures are shared between (a) and (b).}</td>
    </tr>

  </table>
</div>

**Table 1. reproduction compared with our results:**

| Reproduction|  IWAE  | PIWAE (8,8)  | MIWAE (8,8)  | CIWAE β=0.5      |VAE     |
| :---        | :---:  | :---:        | :---:        | :---:            |  :---: |
| IWAE-64     | –84.64 | –84.72       | –84.98       | –87.05           | –87.26 |
| log p̂(x)    | –84.64 | –84.90       | –85.04       | –87.00           | –87.66 | 
| −KL(Q\|\|P) | 0.00   | 0.19         | 0.06         | –0.05            | 0.40   |
| **Original paper**  |  **IWAE**      | **PIWAE (8,8)**       | **MIWAE (8,8)**   | **CIWAE β=0.5**    |**VAE** |
| IWAE-64     | –86.11 ± 0.10 | –85.74 ± 0.07  | –85.69 ± 0.04 | –86.08 ± 0.08  | –86.69 ± 0.08 |
| log p̂(x)    |–84.52 ± 0.02  | –84.46 ± 0.06  | –84.97 ± 0.10 | –85.24 ± 0.08  |–86.21 ± 0.19  |
| −KL(Q\|\|P) |–1.59 ± 0.10   | –1.28 ± 0.09   | –0.72 ± 0.11  | –0.84 ± 0.11   |–0.47 ± 0.20   |
| **Larger model**     |  **IWAE**         | **PIWAE (8,8)**      | **MIWAE (8,8)**   | **CIWAE β=0.5**    |**VAE** |
| IWAE-64     | –83.85 | –83.86| –83.47| –84.92    | –85.33     |
| log p̂(x)    | –83.92 | –83.84| –83.66| –84.98    | –85.31     |
| −KL(Q\|\|P) | 0.06   | –0.02 | 0.19  | 0.06      | –0.02      |

_Note: Due to compututional resources limitations, we run these experiments for the whole 3280 epochs, however with only one repetition (the original paper shows performance averaged over 4 runs)._

## Reconstructions with qualitative and quantitative evaluations:

On the **MNIST** dataset:

<p align="center">
<img src="https://raw.githubusercontent.com/madhubabuv/TightIWAE/master/_illustrations/reconstructions_mnist.png" width="660">
</p>

On the **Omniglot** dataset:

<p align="center">
<img src="https://raw.githubusercontent.com/madhubabuv/TightIWAE/master/_illustrations/reconstructions_omniglot.png" width="660">
</p>

And finally a **generalization-ability evaluation** between these datasets:

<p align="center">
<img src="https://raw.githubusercontent.com/madhubabuv/TightIWAE/master/_illustrations/reconstructions_cross_generalization.png" width="600">
</p>

# Environment:

Tested with CUDA 10.0 and the following libraries (note that other versions are also likely to work, but the presented model loading might be dependant on the same used versions): 

```
pip install torch==1.4.0 torchvision==0.5.0
pip install h5py
pip install matplotlib
pip install scipy==1.3.1 numpy==1.17.0
```

# Acknowledgements:

Implementation of the code present in this repository is based on the [basic Pytorch VAE demo](https://github.com/pytorch/examples/blob/master/vae/main.py) and the unofficial [MIWAE implementation by yoonholee](https://github.com/yoonholee/pytorch-vae). We have however edited the code significantly, added new methods (PIWAE, CIWAE including a new learnable Beta version of CIWAE), new experiments and evaluation metrics (SSIM from [here](https://github.com/Po-Hsun-Su/pytorch-ssim)). Data loaders have also been inspired by existing repositories ([this one](https://github.com/yoonholee/pytorch-vae/blob/master/data_loader/stoch_mnist.py) and [IWAE official](https://github.com/yburda/iwae/blob/master/datasets.py)) to maintain reproducibility of the results between implementations.


# References:

[1] Rainforth, Tom, et al. "_Tighter variational bounds are not necessarily better._" ICML. PMLR, 2018. [arXiv:[1802.04537](https://arxiv.org/abs/1802.04537)]

[2] Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov. "_Importance weighted autoencoders_." (2015). [arXiv:[1509.00519](https://arxiv.org/abs/1509.00519)]
