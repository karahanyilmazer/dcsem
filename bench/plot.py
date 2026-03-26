import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = [5, 3]


def plot_conf_mat(conf_mat, param_names, title=None):
    """
    Plots a heatmap of the given confusion matrix using seaborn.

    :param conf_mat: numpy array or 2D list
        The confusion matrix to visualize. Should have shape (len(param_names), len(param_names)).
        The values in the matrix should be between 0 and 1, representing normalized frequencies or probabilities.

    :param param_names: list of str
        The names of the parameters or classes corresponding to the rows and columns of the confusion matrix.
        The order should match the rows/columns of the conf_mat.

    :param title: str, optional
        The title to display above the heatmap. If not provided, no title will be shown.

    :return: matplotlib.axes.Axes
        The Axes object containing the heatmap plot.
    """

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(conf_mat.T, annot=True, fmt="2.2f", vmin=0, vmax=1,
                     annot_kws={'size': 8})

    ax.set_xticklabels(labels=param_names, rotation=45, fontdict={'size': 8})
    ax.set_yticklabels(labels=param_names, rotation=45, fontdict={'size': 8})
    ax.set_xlabel('Actual change', fontdict={'size': 12})
    ax.set_ylabel('Inferred Change', fontdict={'size': 12})
    if title is not None:

        plt.title(title)

    plt.tight_layout()
    return ax


def bar_plot(measures, names, bvals, summary_type='dt'):
    """
     Plots a series of bar charts to visualize summary measurements for different conditions or groups.

     :param measures: list of numpy arrays
         A list of arrays, each representing a set of measurements to be plotted.
         Each array should have the same length and the length should correspond to the number of unique b-values.
         The array contains the actual measurements for the bars.

     :param names: list of str
         A list of names, each corresponding to a set of measurements in the `measures` list.
         These names will be used as the labels in the legend.

     :param bvals: list or numpy array
         A list of unique b-values (not including b0) that correspond to the x-axis locations for each group of bars.

     :param summary_type: str, optional (default='dt')
         Determines the type of summary measurements being plotted.
         'dt': Diffusion tensor measurements. Will label the y-axis with 'Mean Diffusivity' and 'Fractional Anisotropy'.
         'sh': Spherical harmonics measurements. Will label the y-axis with 'Mean signal' and 'L2 norm'.
         Any other value: General measurements. Will label the y-axis with 'measure 1' and 'measure 2'.

     :return: matplotlib.figure.Figure
         The Figure object containing the bar plots.
     """

    if summary_type == 'dt':
        summary_names = ['Mean Diffusivity', 'Fractional Anisotropy']
    elif summary_type == 'sh':
        summary_names = ['Mean signal', 'L2 norm']
    else:
        summary_names = ['measure 1', 'measure 2']

    n_meas = len(measures)
    m1_idx = 1 + 2 * np.arange(0, len(bvals))
    m2_idx = 2 + 2 * np.arange(0, len(bvals))

    bar_width = min(np.diff(bvals)) / n_meas * 0.8
    colors = sns.color_palette("hls", n_meas)

    fig = plt.figure(figsize=[7, 4], dpi=300)
    for j, (m, idx) in enumerate(zip(summary_names, [m1_idx, m2_idx])):
        plt.subplot(1, 2, j + 1)
        for i, meas in enumerate(measures):
            plt.bar(x=bvals + bar_width * (i - n_meas / 2) + bar_width / 2, height=np.squeeze(meas)[idx],
                    width=bar_width, label=names[i], color=colors[i])

        plt.xticks(bvals)
        plt.xlabel('b-value')
        plt.ylabel(m)
        if j == 0:
            plt.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 8})

    plt.tight_layout()
    return fig


def arrow_plot(baseline, changes, names, bvals, covs=None, arrowprops=dict(), ax=None, offset=0.005, summary_type='dt'):
    """
        Plots arrows showing changes from a baseline value. Arrows start from the baseline and end at the changed value.

        :param baseline: numpy array
            Base values from which the arrows will originate.

        :param changes: list of numpy arrays
            A list of change arrays, each representing changes from the baseline. The number of changes
            should correspond to the length of names minus one (since the first name is for the baseline).

        :param names: list of str
            A list of names corresponding to the baseline and each change. The first name is for the baseline,
            and the subsequent names are for the changes.

        :param bvals: list or numpy array
            A list of b-values used for subplots.

        :param covs: list of numpy arrays, optional
            List of covariance matrices corresponding to each change. If provided, Gaussian uncertainty ellipses
            will be plotted around the arrow endpoints. Default is None.

        :param arrowprops: dict, optional
            Dictionary with properties for the arrows. Default properties are `arrowstyle='<-'` and `color='black'`.

        :param ax: matplotlib.axes.Axes, optional
            Axes on which to draw the plot. If not provided, a new figure with axes will be created.

        :param offset: float, optional
            Offset for the text annotations to avoid overlapping with data points. Default is 0.005.

        :param summary_type: str, optional (default='dt')
            Determines the type of summary measurements being plotted.
            'dt': Diffusion tensor measurements.
            'sh': Spherical harmonics measurements.
            Any other value: General measurements.

        :return: matplotlib.axes.Axes
            The Axes object containing the plot.
        """

    if summary_type == 'dt':
        summary_names = ['Mean Diffusivity', 'Fractional Anisotropy']
    elif summary_type == 'sh':
        summary_names = ['Mean signal', 'L2 norm']
    else:
        summary_names = ['measure 1', 'measure 2']

    base = np.squeeze(baseline)
    default_arrowprops = dict(arrowstyle='<-', color='black')
    default_arrowprops.update(arrowprops)
    max_name_length = max([len(name) for name in names])
    offset_x = np.maximum(.01 * max_name_length, 0.005)
    offset_y = 0.1
    if ax is None:
        fig, ax = plt.subplots(1, len(bvals), figsize=[4 * len(bvals), 4], dpi=300)

    for b_i, b in enumerate(bvals):
        idx1, idx2 = 1 + b_i * 2, 2 + b_i * 2
        ax[b_i].scatter(base[idx1], base[idx2], c='grey')

        ax[b_i].text(base[idx1] + offset, base[idx2], names[0], ha='left', va='bottom')
        all_x = [*ax[b_i].get_xlim(), base[idx1]]
        all_y = [*ax[b_i].get_ylim(), base[idx2]]
        colors = sns.color_palette("tab10", len(names))

        for i, ch in enumerate(changes):
            ch = np.squeeze(ch)

            ax[b_i].annotate('', xy=(base[idx1], base[idx2]),
                             xytext=(base[idx1] + ch[idx1], base[idx2] + ch[idx2]),
                             arrowprops=default_arrowprops)
            if covs is not None:
                plot_gaussian_2d(ch[[idx1, idx2]] + base[[idx1, idx2]],
                                 covs[i][[idx1, idx2], :][:, [idx1, idx2]],
                                 ax=ax[b_i], color=colors[i], xlim=0.2, ylim=0.2)

            ax[b_i].scatter(ch[idx1] + base[idx1], ch[idx2] + base[idx2], color=colors[i])
            ax[b_i].text(base[idx1] + ch[idx1] + offset, base[idx2] + ch[idx2] + offset, names[i+1], ha='left', va='top',
                         size=12, fontdict=None)
            all_x.extend([ch[idx1] + base[idx1]])
            all_y.extend([ch[idx2] + base[idx2]])

        ax[b_i].axis(xmin=min(all_x) - offset_x, xmax=max(all_x) + offset_x)
        ax[b_i].axis(ymin=min(all_y) - offset_y, ymax=max(all_y) + offset_y)

        ax[b_i].set_xlabel(summary_names[0])
        ax[b_i].set_ylabel(summary_names[1])
        ax[b_i].set_title(f'b={b}')
    plt.tight_layout()
    return ax


def plot_gaussian_2d(mean, cov, ax, color, xlim=0.1, ylim=0.1):
    """
    Plots a 2D Gaussian as contour lines on a given axis.

    :param mean: numpy array
        2D mean of the Gaussian distribution.

    :param cov: numpy array
        2x2 covariance matrix of the Gaussian distribution.

    :param ax: matplotlib.axes.Axes
        Axes on which to draw the Gaussian.

    :param color: matplotlib color
        Color for the Gaussian contour fill.

    :param xlim: float, optional
        Limits for the x-axis around the mean to generate the grid for plotting. Default is 0.1.

    :param ylim: float, optional
        Limits for the y-axis around the mean to generate the grid for plotting. Default is 0.1.
    """

    # Generate 2D grid
    x = mean[0] + np.arange(-xlim, xlim, 0.001)
    y = mean[1] + np.arange(-ylim, ylim, 0.001)
    X, Y = np.meshgrid(x, y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    inv_cov = np.linalg.inv(cov)
    diff = pos - mean
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=-1)
    Z = np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    Z = Z / Z.max()
    ax.plot(mean[0], mean[1], '.', color='grey')

    cs = ax.contourf(X, Y, Z, colors=color, alpha=.5, levels=np.logspace(-3, 0, 4))
    ax.contour(X, Y, Z, levels=cs.levels, colors='black', linewidths=0.5)
    # ax.clabel(cs, fontsize=5, colors='black')


def dummy_legends(ax, attr, vals, names):
    """
    Adds dummy legend entries to a given axis based on specified attributes, values, and names.
    The function is useful for adding legends with arbitrary styles, e.g., line styles or marker styles.

    :param ax: matplotlib.axes.Axes
        The Axes object on which to add the legends.

    :param attr: str
        The attribute which varies among the legends, e.g., 'linestyle', 'marker', 'color', etc.

    :param vals: list
        List of values for the attribute. Each value corresponds to a legend entry.

    :param names: list of str
        Names to be shown for each legend entry. Should be the same length as `vals`.

    :return: None
        Modifies the given Axes object to include the legend.
    """
    legend_elements = []
    for v, n in zip(vals, names):
        props = {'color': 'k', 'ls': '-', 'lw': 2}
        props.update({attr: v, 'label': n})
        legend_elements.extend(ax.plot([0], [0], **props))

        ax.legend(handles=legend_elements, loc='upper left')

