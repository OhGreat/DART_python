import numpy as np
from os import listdir
import matplotlib.pyplot as plt

def plot_results(res_dir, labels=None, tick_labels=None,
                title=None, use_log=False, fig_size=(6,4),
                xlabel=None, x_rotate=None, ylabel=None, ylim=50,
                save_name=None):
    """ Creates a plot of the experiments, given the file structure:
            - res_path/
                -- exp_0/
                    --- alg_0.npy
                    --- alg_1.npy
                --exp_1/
                    --- alg_0.npy
                    --- alg_1.npy
                ...
    """
    #check path
    if res_dir[-1] != "/":
        res_dir += "/"

    family_subdirs = sorted(listdir(res_dir))
    print("\nConsidered subdirectories:",family_subdirs)
    alg_names = sorted(listdir(res_dir+family_subdirs[0]))
    n_algs = len(sorted(listdir(res_dir+family_subdirs[0])))
    
    # define figure
    plt.figure(figsize=fig_size)
    
    counter = 0
    # iterate over each algorithm
    for i in range(n_algs):
        found = False
        for label in labels:
            if label[0] in alg_names[i]:
                found = True
        if not found: continue
        curr_alg_errs = []
        # iterate over each experiment
        for curr_dir in family_subdirs:
            curr_dir = res_dir + curr_dir + "/"
            dir_files = sorted(listdir(curr_dir))
            # take only files relevant to current algorithm
            for filename in dir_files:
                if alg_names[i] not in filename:
                    continue
                if '.npy' in filename:
                    curr_alg_errs.append(np.load(curr_dir+filename))
        # define values to plot
        means = np.mean(curr_alg_errs, axis=0)
        maxs = np.max(curr_alg_errs, axis=0)
        mins = np.min(curr_alg_errs, axis=0)
        if use_log:
            means = np.log(means)
            maxs = np.log(maxs)
            mins = np.log(mins)
        # plotting styles
        if "DART" in alg_names[i]:
            line = "solid"
            if "sirt" in alg_names[i]:
                line = "--"
        elif "DART" not in alg_names[i] and "SART" in alg_names[i] or "SIRT" in alg_names[i]:
            line = "dotted"
        else: line = "dashdot"
        if "FBP" in alg_names[i] or "fbp" in alg_names[i]:
            line = "dashdot"
        # choose label and color
        lab = None
        curr_color = None
        for label in labels:
            if label[0] in alg_names[i]:
                lab = label[0]
                curr_color = label[1]
                # fix label names
                if "RBF" in lab:
                    lab = "FBP"
                if "rbf" in lab:
                    lab = "DART_fbp"
        # plot main mean trend of algorithm
        plt.plot(means, label=lab, 
                linestyle=line,
                linewidth=5, c=curr_color)
        # fill values between min/max
        plt.fill_between(range(len(means)), mins, maxs, alpha=0.2)
        # update control params
        curr_alg_errs = []
        counter += 1

    plt.legend(fontsize=12,frameon=True, fancybox=True, framealpha=1)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xticks(ticks=range(len(means)) ,
                labels=tick_labels, rotation=x_rotate, 
                fontsize=14)
    plt.yticks( fontsize=14)
    if ylim:
        plt.ylim([0, ylim])
    plt.margins(x=0, y=0)
    if save_name is not None:
        plt.savefig(save_name)
