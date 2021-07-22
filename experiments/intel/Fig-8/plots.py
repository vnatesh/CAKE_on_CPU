import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys

def plot_small_mat(x, label=""):
    plt_title = f"{label} Relative Throughput for M = {'' if x < 2 else x}N"
    plt.rcParams.update({"font.size": 12})
    plt.rcParams.update({"figure.figsize": (5,5)})
    colors = ["#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]  # colorbrewer red

    M = [i * 500 for i in range(2, 17, 2)]
    K = [
        1,
        10,
        50,
        100,
        500,
        1000,
        1500,
        2000,
        2500,
        3000,
        3500,
        4000,
        4500,
        5000,
        5500,
        6000,
        6500,
        7000,
        7500,
        8000,
    ]
    vals = [1.0, 1.25, 1.5, 2.0]
    thresh = [[] for i in range(len(vals))]
    #
    for m in M:
        # df_file = "results_full" if x < 4 else f"results_{x}N"
        df_file = "results_full"
        df1 = pandas.read_csv(df_file)

        diffs = []
        for k in K:
            a = df1[
                (df1["algo"] == "cake")
                & (df1["M"] == m)
                & (df1["K"] == k)
                & (df1["N"] == m / x)
            ]["time"]
            lat_cake = a.mean()
            b = df1[
                (df1["algo"] == "mkl")
                & (df1["M"] == m)
                & (df1["K"] == k)
                & (df1["N"] == m / x)
            ]["time"]
            lat_mkl = b.mean()
            diffs.append(lat_mkl / lat_cake)

        for v in range(len(vals)):
            thresh[v].append(8000)
            for i in range(len(diffs)):
                if diffs[i] <= vals[v]:
                    thresh[v][-1] = K[i]
                    break

    for i in range(len(vals)):
        plt.plot(M, thresh[i], label="≥ %.2fx" % vals[i], color=colors[i])

    plt.xticks(list(range(0, 8001, 1000)))
    plt.yticks(list(range(0, 8001, 1000)))
    plt.title(plt_title)
    plt.axis("scaled")
    plt.xlim([1000, 8000])
    plt.ylim([1000, 8000])

    plt.xlabel(f"M = {'' if x < 2 else x}N", labelpad=4, fontsize=16)
    plt.ylabel("K", fontsize=16, labelpad=12).set_rotation(0)
    # plt.legend(loc="upper right", prop={"size": 12})

    leg = plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(10.0)

    plt.fill_between(M, 0, thresh[3], color=colors[3])
    plt.fill_between(M, thresh[3], thresh[2], color=colors[2])
    plt.fill_between(M, thresh[2], thresh[1], color=colors[1])
    plt.fill_between(M, thresh[1], thresh[0], color=colors[0])
    plt.tight_layout()
    plt.savefig(f"contour_{x}N.pdf", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.clf()
    plt.close("all")


plot_small_mat(1, label="(a)")
plot_small_mat(2, label="(b)")
plot_small_mat(4, label="(c)")
plot_small_mat(8, label="(d)")
