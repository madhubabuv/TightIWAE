# Paper reproduction: Tighter Variational Bounds are Not Necessarily Better
Part of the [Advanced Topics in Machine Learning course of 2021](https://www.cs.ox.ac.uk/teaching/courses/2020-2021/advml/) as reproducibility challenge

Authors alphabetically: [Amine M'Charrak](https://github.com/mcharrak), [Vít Růžička](https://github.com/previtus), [Sangyun Shin](https://github.com/yunshin), [Madhu Vankadari](https://github.com/madhubabuv/) @ cs.ox.ac.uk

## Description:

Reproduction of the paper [1] which implements PIWAE (partially importance weighted auto-encoder), MIWAE (multiply importance weighted auto-encoder) and the CIWAE (combination importance weighted autoencoder) as an extension to the IWAE (importance weighted auto-encoder) model of [2].

We additionally extend the experiments with a larger tested model (with roughly doubled number of parameters), with evaluation on the Omniglot dataset (and also check cross-dataset generalization between Omniglot and MNIST) and finally propose an edited version of the CIWAE model with learnable parameter Beta (which results in a more robust model in terms of the Beta parameter initialization.

More details can be seen in our report or in our poster (links to both to be added).

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

Figure 5. reconstruction from the original paper [1]. Shows the training progress of the newly tested methods:

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



# References:

[1] Rainforth, Tom, et al. "_Tighter variational bounds are not necessarily better._" ICML. PMLR, 2018. [arXiv:[1802.04537](https://arxiv.org/abs/1802.04537)]

[2] Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov. "_Importance weighted autoencoders_." (2015). [arXiv:[1509.00519](https://arxiv.org/abs/1509.00519)]
