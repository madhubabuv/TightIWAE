# Paper reproduction: Tighter Variational Bounds are Not Necessarily Better
Part of the [Advanced Topics in Machine Learning course of 2021](https://www.cs.ox.ac.uk/teaching/courses/2020-2021/advml/) as reproducibility challenge

Authors alphabetically: [Amine M'Charrak](https://github.com/mcharrak), [Vít Růžička](https://github.com/previtus), [Sangyun Shin](https://github.com/yunshin), [Madhu Vankadari](https://github.com/madhubabuv/) @ cs.ox.ac.uk

## Description:

Reproduction of the paper [1] which implements PIWAE (partially importance weighted auto-encoder), MIWAE (multiply importance weighted auto-encoder) and the CIWAE (combination importance weighted autoencoder) as an extension to the IWAE (importance weighted auto-encoder) model of [2].

We additionally extend the experiments with a larger tested model (with roughly doubled number of parameters), with evaluation on the Omniglot dataset (and also check cross-dataset generalization between Omniglot and MNIST) and finally propose an edited version of the CIWAE model with learnable parameter Beta (which results in a more robust model in terms of the Beta parameter initialization.

More details can be seen in our report or in our poster (links to both to be added).

## Run instructions:

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

To load an older experiment (both model and history of losses) add `--cont` to the line.


# References:

[1] Rainforth, Tom, et al. "_Tighter variational bounds are not necessarily better._" ICML. PMLR, 2018.

[2] Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov. "_Importance weighted autoencoders_." arXiv:1509.00519 (2015).
