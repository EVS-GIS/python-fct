import matplotlib.pyplot as plt
import matplotlib as mpl
from Plotting import MapFigureSizer

def plot_swath(x, swath, relative=False, title=None, filename=None):

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
    ax = fig.add_subplot(gs[25:100,10:95])

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    if relative:
        ax.set_ylabel("Relative Elevation (m)")
    else:
        ax.set_ylabel("Elevation (m NGF)")
    ax.set_xlabel("Distance from reference axis (m)")
    ax.tick_params(axis='both', width=1, pad = 2)
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(2)
    ax.grid(which='both', axis='both', alpha=0.5)

    ax.fill_between(x, swath[:, 0], swath[:, 4], facecolor='#b9d8e6', alpha = 0.2, interpolate=True)
    ax.plot(x, swath[:, 0], "gray", x, swath[:, 4], "gray", linewidth = 0.5, linestyle='--')
    ax.fill_between(x, swath[:, 1], swath[:, 3], facecolor='#48638a', alpha = 0.5, interpolate=True)
    ax.plot(x, swath[:, 2], "#48638a", linewidth = 1)

    fig_size_inches = 6.25
    aspect_ratio = 2
    cbar_L = "None"
    [fig_size_inches,map_axes,cbar_axes] = MapFigureSizer(fig_size_inches, aspect_ratio, cbar_loc=cbar_L, title=True)

    plt.title(title)
    fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
    ax.set_position(map_axes)

    if filename is None:
        fig.show()
    elif filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', dpi=600)
        plt.clf()
    else:
        plt.savefig(filename, format='png', dpi=300)
        plt.clf()
