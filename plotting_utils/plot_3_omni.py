from pylab import plt
from plot_utils import *

# all at ~ 500 now after 10 hrs ... will wait for tmr

experiment_name = "Model replication, different datasets"
plot_name = "3_omni_"
folder = "3_other_datasets/logs/"
piwaes = [folder+"log_PIWAE_M8_k8_omniglot.h5"]
miwaes = [folder+"log_MIVAE_M8_k8_omniglot.h5"]
iwaes = [folder+"log_MIVAE_M1_k64_omniglot.h5"]
vaes = [folder+"log_MIVAE_M1_k1_omniglot.h5"]
ciwaes = [folder+"log_CIVAE_beta0.5_omniglot.h5" ]


### bellow is unchanged ###
linewidth = 1

def plot_all(select_func, rolling_mean, plot_name, ylabel, legend=True):
    all_available = min_epoch([piwaes, miwaes, iwaes, vaes, ciwaes], select_func)
    print("from all logs we have data at least in epoch", all_available)

    # MIVAE M8, k8
    p = plot_h5_file_k64_average(piwaes, "PIWAE M=8, k=8", color='#c44e52', at=all_available, select_func=select_func,
                                 rolling_mean=rolling_mean, linewidth=linewidth)
    m = plot_h5_file_k64_average(miwaes, "MIWAE M=8, k=8", color='#8172b2', at=all_available, select_func=select_func,
                                 rolling_mean=rolling_mean, linewidth=linewidth)
    i = plot_h5_file_k64_average(iwaes, "IWAE k=64", color='#4c72b0', at=all_available, select_func=select_func,
                                 rolling_mean=rolling_mean, linewidth=linewidth)
    v = plot_h5_file_k64_average(vaes, "VAE", color='#ccb974', at=all_available, select_func=select_func,
                                 rolling_mean=rolling_mean, linewidth=linewidth)
    c = plot_h5_file_k64_average(ciwaes, "CIWAE beta=0.5", color='#55a868', at=all_available, select_func=select_func,
                                 rolling_mean=rolling_mean, linewidth=linewidth)

    # plt.ylim(-92,-83)
    plt.ylim(-125.0, -105.0)
    #plt.ylim(-94, -82.5)

    # plt.xlim(0,2300)
    plt.xlim(0, all_available)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    if legend:
        plt.legend(loc='lower right')
    plt.tight_layout()
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    plt.savefig(plot_name+".png", dpi=120)
    plt.savefig(plot_name+".pdf", dpi=120)

    plt.show()
    plt.close()

    print(ylabel,":")
    print("============(average from last 20 epochs)======================")
    print("PIWAE M=8, k=8       ", p)
    print("MIWAE M=8, k=8       ", m)
    print("IWAE k=64       ", i)
    print("VAE       ", v)
    print("CIWAE beta=0.5       ", c)


### Figure 5 reconstruction

select_iwae_64 = lambda x: x[1]
#plt.figure(figsize=(8,4))
plt.figure(figsize=(4,3.4))
#plt.title("Figure 5.a replication, IWAE64; "+experiment_name)
select_func = select_iwae_64
rolling_mean = 5
plot_all(select_func,rolling_mean,plot_name+"_iwae64", ylabel="IWAE-64")


select_iwae_5000 = lambda x: x[2]
#plt.figure(figsize=(8,4))
plt.figure(figsize=(4,3.4))
#plt.title("Figure 5.b replication, log p̂(x) (= IWAE5000); "+experiment_name)
select_func = select_iwae_5000
rolling_mean = None
plot_all(select_func,rolling_mean,plot_name+"_iwae5000", ylabel="log p̂(x)", legend=False)
