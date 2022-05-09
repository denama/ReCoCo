import matplotlib.pyplot as plt
import itertools
import numpy as np

def plot_cdf(data, x_label, figsize=(10,5), color="teal", legend=None):

    # figsize = (16, 9)
    labelspacing = 0.4
    legend_fontsize = 14
    fontsize = 12


    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='both', which='minor',
                   length=0,
                   direction='in'
                  )
    ax.tick_params(axis='both', which='major',
                   #length=0,
                   direction='in'
                  )


    cdfx = np.sort(data)
    cdfy = np.linspace(1 / len(data), 1.0, len(data))

    p = plt.plot(cdfx, cdfy, color=color,)
    # print(p[0].get_color())

    # plt.xticks(np.arange(0,26,2))
    plt.yticks(np.arange(0,1.1,0.1))
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize, labelpad=10)
#     plt.xlim(-0.1,24)
    plt.ylabel('ECDF', fontsize=fontsize, labelpad=10)
    plt.grid()

    plt.tight_layout()


def plot_cdf_multiple(list_cdf_data, x_label, figsize=(10,5), legend=None):

    # figsize = (16, 9)
    labelspacing = 0.4
    legend_fontsize = 14
    fontsize = 12

    # colors_dataset = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    linestyles = ["solid", "dotted", "dashed", "dashdot", (0, (3, 1, 1, 1))]
    ls = itertools.cycle(linestyles)
    colors_dataset = ["indianred", "teal", "lightgreen"]
    cd = itertools.cycle(colors_dataset)

    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='both', which='minor',
                   length=0,
                   direction='in'
                  )
    ax.tick_params(axis='both', which='major',
                   #length=0,
                   direction='in'
                  )

    for i in range(len(list_cdf_data)):
        data = list_cdf_data[i]
        cdfx = np.sort(data)
        cdfy = np.linspace(1 / len(data), 1.0, len(data))

        p = plt.plot(cdfx, cdfy,
                     linestyle=next(ls),
    #                  color = next(cd),
                    )
        print(p[0].get_color())

    # plt.xticks(np.arange(0,26,2))
    plt.yticks(np.arange(0,1.1,0.1))
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    plt.xlabel(x_label, fontsize=fontsize, labelpad=10)
#     plt.xlim(-0.1,24)
    plt.ylabel('ECDF', fontsize=fontsize, labelpad=10)
    plt.grid()

    plt.legend(legend, prop={'size': legend_fontsize}, labelspacing=labelspacing)
    plt.tight_layout()