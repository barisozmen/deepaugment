__date__ = "10 Nov 2018"
__author__ = "Baris Ozmen"


"""Collects plotting functions to be used
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

import sys

sys.path.append("../")

from lib.decorators import Reporter

timer = Reporter.timer
counter = Reporter.counter

plt.interactive(False)


class PlotOp:
    """Plotting Operations class. Keeps static plot functions.

    Each function is conforming as much as possible to "Functional Programming Standards" (https://en.wikipedia.org/wiki/Functional_programming).
    Each function, hence, do not affected by any variable not given to it as argument (with exception of influences from
    appealed external libraries, such as matplotlib, altair, and bokeh) .
    """

    @staticmethod
    @timer
    @counter
    def plot_heatmap(
        value_matrix,
        xlabels,
        ylabels,
        min_value=-1,
        max_value=1,
        labels_fontsize=24,
        xlabels_fontsize=None,
        ylabels_fontsize=None,
        colorbar_labels_fontsize=None,
        sg_length=None,
        figsize=None,
        save_fig_to=None,
    ):

        """Plots heatmap of a given 2d array

            Args:
                value_matrix (numpy.array): matrix to plot its heatmap
                xlabels (list):
                ylabels (list):
                min_value (float):
                max_value (float):
                labels_fontsize (int): determines fontsize of text in the plot.

                xlabels_fontsize (int): determines fontsize of x axis labels. If not given,
                                        value of `labels_fontsize` argument is used.

                ylabels_fontsize (int): determines fontsize of y axis labels. If not given,
                                        value of `labels_fontsize` argument is used.

                colorbar_labels_fontsize (int): determines fontsize of colorbar labels. If
                                                not given, value of `labels_fontsize` argument is used.

                sg_length (int): slopper over grid line lenght. If not given, it is
                                 taken as 10% of length of x or y axes whichever is smaller.

                figsize (tuple): determines size of the plot
            """

        if xlabels_fontsize is None:
            xlabels_fontsize = labels_fontsize

        if ylabels_fontsize is None:
            ylabels_fontsize = labels_fontsize

        if colorbar_labels_fontsize is None:
            colorbar_labels_fontsize = labels_fontsize - 2

        if figsize is None:
            figsize = (int(len(xlabels) / 2), int(len(ylabels) / 2))
            print("figsize is {}".format(figsize))

        fig = plt.figure(figsize=figsize)

        ################
        # Plot heatmap #
        ################
        im = plt.imshow(
            value_matrix,
            cmap="bwr",
            interpolation="nearest",
            vmin=min_value,
            vmax=max_value,
        )

        ax = plt.gca()

        ax.xaxis.set_ticks_position("top")
        ax.yaxis.set_ticks_position("left")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=90, fontsize=xlabels_fontsize)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, rotation=0, fontsize=ylabels_fontsize)

        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xlength = xlim[1] - xlim[0]
        ylength = ylim[1] - ylim[0]

        if sg_length is None:
            sg_length = min(xlength, ylength) * 0.1

        x_micro_grid = np.arange(0, len(xlabels) + 1, 1) + 0.5
        y_micro_grid = np.arange(0, len(ylabels) + 1, 1) + 0.5

        x_macro_grid = np.arange(0, len(xlabels) + 1, 4) - 0.5
        y_macro_grid = np.arange(0, len(ylabels) + 1, 4) - 0.5

        for x in x_micro_grid:
            plt.plot([x, x], ylim, "k-", lw=0.5)

        for y in y_micro_grid:
            plt.plot(xlim, [y, y], "k-", lw=0.5)

        for x in x_macro_grid:
            lines = plt.plot([x, x], [ylim[0], ylim[1] + sg_length], "k-", lw=2)
            lines[0].set_clip_on(False)

        for y in y_macro_grid:
            lines = plt.plot([xlim[0] + sg_length, xlim[1]], [y, y], "k-", lw=2)
            lines[0].set_clip_on(False)

        plt.xlim(xlim)
        plt.ylim(ylim)

        colorbar_axes = fig.add_axes([1, 0.13, 0.02, 0.8])

        colorbar_axes.tick_params(labelsize=labels_fontsize - 2)

        plt.colorbar(im, cax=colorbar_axes)

        if save_fig_to is not None:
            plt.savefig(save_fig_to)

        plt.show()

    @staticmethod
    @timer
    @counter
    def plot_heatmap_of_uppertriangle(
        value_matrix,
        xlabels,
        ylabels,
        min_value=-1,
        max_value=1,
        labels_fontsize=18,
        xlabels_fontsize=None,
        ylabels_fontsize=None,
        colorbar_labels_fontsize=None,
        sg_length=None,
        figsize=None,
        save_fig_to=None,
    ):

        """Plots heatmap of a given 2d array

        Args:
            value_matrix (numpy.array): matrix to plot its heatmap
            xlabels (list):
            ylabels (list):
            min_value (float):
            max_value (float):
            labels_fontsize (int): determines fontsize of text in the plot.

            xlabels_fontsize (int): determines fontsize of x axis labels. If not given,
            value of `labels_fontsize` argument is used.

            ylabels_fontsize (int): determines fontsize of y axis labels. If not given,
            value of `labels_fontsize` argument is used.

            colorbar_labels_fontsize (int): determines fontsize of colorbar labels. If
            not given, value of `labels_fontsize` argument is used.

            sg_length (int): slopper over grid line lenght. If not given, it is
            taken as 10% of length of x or y axes whichever is smaller.

            figsize (tuple): determines size of the plot
        """

        if xlabels_fontsize is None:
            xlabels_fontsize = labels_fontsize

        if ylabels_fontsize is None:
            ylabels_fontsize = labels_fontsize

        if colorbar_labels_fontsize is None:
            colorbar_labels_fontsize = labels_fontsize - 2

        if figsize is None:
            figsize = (int(len(xlabels) / 2), int(len(ylabels) / 2))
            print("figsize is {}".format(figsize))

        # get upper-triangle
        value_matrix = np.triu(value_matrix, k=1)

        fig = plt.figure(figsize=figsize)

        ################
        # Plot heatmap #
        ################
        im = plt.imshow(
            value_matrix,
            cmap="bwr",
            interpolation="nearest",
            vmin=min_value,
            vmax=max_value,
        )

        ax = plt.gca()

        ax.xaxis.set_ticks_position("top")
        ax.yaxis.set_ticks_position("right")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=90, fontsize=xlabels_fontsize)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, rotation=0, fontsize=ylabels_fontsize)

        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xlength = xlim[1] - xlim[0]
        ylength = ylim[1] - ylim[0]

        if sg_length is None:
            sg_length = min(xlength, ylength) * 0.1

        x_micro_grid = np.arange(0, len(xlabels) + 1, 1) + 0.5
        y_micro_grid = np.arange(0, len(ylabels) + 1, 1) + 0.5

        x_macro_grid = np.arange(0, len(xlabels) + 1, 4) - 0.5
        y_macro_grid = np.arange(0, len(ylabels) + 1, 4) - 0.5

        for x in x_micro_grid:
            plt.plot([x, x], [x, ylim[1]], "k-", lw=0.5)

        for y in y_micro_grid:
            plt.plot([y, xlim[1]], [y, y], "k-", lw=0.5)

        for x in x_macro_grid:
            lines = plt.plot([x, x], [x, ylim[1] + sg_length], "k-", lw=2)
            lines[0].set_clip_on(False)

        for y in y_macro_grid:
            lines = plt.plot([y, xlim[1] - sg_length], [y, y], "k-", lw=2)
            lines[0].set_clip_on(False)

        plt.xlim(xlim)
        plt.ylim(ylim)

        colorbar_axes = fig.add_axes([0, 0.13, 0.02, 0.6])
        colorbar_axes.tick_params(labelsize=labels_fontsize - 2)

        plt.colorbar(im, cax=colorbar_axes)

        if save_fig_to is not None:
            plt.savefig(save_fig_to + ".png")

        plt.show()

    @staticmethod
    @timer
    @counter
    def plot_heatmap_with_altair(
        df,
        xcol="event1",
        ycol="event2",
        valuecol="phi",
        minval=-1,
        maxval=1,
        tooltip=None,
        savefigto=None,
    ):
        """

        Args:
            df (pandas.DataFrame): dataframe to be drawn
            xcol (str): `df` column for x-axis
            ycol (str): `df` column for
            valuecol (str): `df` column for value (colored)
            tooltip (list): `df` columns to be shown when blocks are hovered by mouse
            savefigto (str): address that figure will be saved
        """
        # Argument operations
        assert xcol in df.columns
        assert ycol in df.columns
        assert savefigto is not None

        if tooltip is None:
            tooltip = [xcol, ycol]
        assert all(item in df.columns for item in tooltip)

        import altair as alt

        color_scheme = "redblue"

        selection_by_x = alt.selection_multi(fields=[xcol], nearest=False)

        selection_by_y = alt.selection_multi(fields=[ycol], nearest=False)

        heatmap_base = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=xcol,
                y=ycol,
                color=alt.Color(
                    "{}:Q".format(valuecol),
                    scale=alt.Scale(scheme=color_scheme, domain=[maxval, minval]),
                ),
                tooltip=tooltip,
            )
        )

        heatmap1 = heatmap_base.encode(
            opacity=alt.condition(selection_by_x, alt.value(1), alt.value(0.2))
        ).add_selection(selection_by_x)

        heatmap2 = heatmap_base.encode(
            opacity=alt.condition(selection_by_y, alt.value(1), alt.value(0.2))
        ).add_selection(selection_by_y)

        whole_chart = alt.layer(heatmap1, heatmap2).configure_view(strokeWidth=0)

        # whole_chart.save(savefigto + '.png')

        whole_chart.save(savefigto + ".html")

        return whole_chart

    @staticmethod
    @timer
    @counter
    def plot_mutation_percentage(df):
        """Plots percentage of patients having misssense or nonsense mutation, as a vertical barplot.

            Args:
                dataframe:
                    columns: hgnc_symbol, percentage_missense, percentage_nonsense
            """

        # data to plot
        n_groups = df["hgnc_symbol"].unique().__len__()
        mis_percentages = df["percentage_missense"].values
        non_percentages = df["percentage_nonsense"].values

        # create plot
        fig, ax = plt.subplots(figsize=(5, 30))

        index = np.arange(n_groups)
        bar_width = 0.25
        opacity = 1

        rects1 = plt.barh(
            index,
            non_percentages,
            bar_width,
            alpha=opacity,
            color="salmon",
            label="Nonsense",
            zorder=2,
        )

        rects2 = plt.barh(
            index + bar_width,
            mis_percentages,
            bar_width,
            alpha=opacity,
            color="c",
            label="Missense",
            zorder=2,
        )

        plt.title(
            "LUAD Driver Genes by Proportion of Samples Mutated \n (ordered alphabetically)"
        )

        plt.xticks(np.arange(0, 35, 5), [str(i) + "%" for i in np.arange(0, 35, 5)])
        plt.yticks(index + bar_width / 2, df["hgnc_symbol"].unique())
        ax.set_xticks(np.arange(0, 35, 1), minor=True)
        plt.legend()

        # ax.yaxis.grid()
        ax.xaxis.grid(True, "minor", linestyle="--", lw=0.2, color="black", zorder=0)
        ax.xaxis.grid(True, "major", linestyle="-", lw=0.3, color="black", zorder=0)

        x0, x1 = plt.xlim()
        y0, y1 = plt.ylim()

        for ylevel in np.arange(5, y1, 10):

            plt.plot(
                [x0, x1 + 1],
                [ylevel + 0.5, ylevel + 0.5],
                color="black",
                alpha=0.3,
                lw=1.2,
                ls="--",
            )

            for i in np.arange(5, 35, 5):
                plt.text(i - 0.8, ylevel + 0.6, str(i) + "%", alpha=0.4)

        plt.xlim((x0, x1 - 1))

        plt.ylim(y0 + 2, y1 - 1.5)

        plt.tight_layout()

    @staticmethod
    @timer
    @counter
    def plot_volcano(
        df,
        xcol,
        ycol,
        xcolrange=[-1, 1],
        xtitle=None,
        ytitle=None,
        ythreshold=2,
        clampthreshold=400,
        tooltip=[],
        topk=5,
        figsize=(600, 600),
        titlefontsize=18,
        titlefontweight="normal",
        labelfontsize=14,
        savefigto="volcano_plot",
        show=False,
    ):
        """Plots volcano figure and saves it to given address

        Args:
            df (pandas.DataFrame): data
            xcol (str): column for x-axis
            ycol (str):
            xcolrange (tuple):
            xtitle (str):
            ytitle (str):
            ythreshold int:
            tooltip (list):
            topk int:
            figsize (tuple):
            titlefontsize (int):
            titlefontweight (int or str):
            labelfontsize (int):
            savefigto (str):
        """
        assert xcol in df.columns
        assert ycol in df.columns
        assert all(item in df.columns for item in tooltip)

        # CLAMP fisher-exact values
        if np.sum(df[ycol] > clampthreshold) > 0:
            df["fisher_exact"] = np.where(
                df["fisher_exact"] > clampthreshold, clampthreshold, df["fisher_exact"]
            )
            print(
                "Datapoints' fisher-exact values clamped at {}!".format(clampthreshold)
            )

        xAxis = alt.Axis(title=xtitle)
        yAxis = alt.Axis(title=ytitle)

        X = alt.X(xcol, scale=alt.Scale(domain=xcolrange), axis=xAxis)
        Y = alt.Y(ycol, scale=alt.Scale(domain=[0, df[ycol].max()]), axis=yAxis)

        points_below = (
            alt.Chart(df[df[ycol] < ythreshold])
            .encode(x=X, y=Y, tooltip=tooltip)
            .mark_circle(size=300, opacity=0.5, color="blue")
            .interactive()
        )

        points_above = (
            alt.Chart(df[df[ycol] >= ythreshold])
            .encode(x=X, y=Y, tooltip=tooltip)
            .mark_circle(size=300, opacity=0.5, color="red")
            .interactive()
        )

        top_k_points = (
            alt.Chart(df.nlargest(topk, ycol))
            .encode(x=X, y=Y)
            .mark_circle(size=0, opacity=0)
            .mark_text(align="right", baseline="bottom", dx=35, dy=-17)
            .encode(text="pairs")
        )

        rule1_data = pd.DataFrame([{"fisher_exact": ythreshold}])
        rule1 = (
            alt.Chart(rule1_data)
            .mark_rule(color="black", opacity=1.0, strokeDash=[1, 1])
            .encode(y=ycol)
        )

        rule2_data = pd.DataFrame([{xcol: 0}])
        rule2 = (
            alt.Chart(rule2_data).mark_rule(color="black", opacity=1.0).encode(x=xcol)
        )

        # Superimpose all charts
        whole_chart = rule1 + rule2 + points_below + points_above + top_k_points

        # adjust aesthetics
        whole_chart = whole_chart.properties(
            width=figsize[0], height=figsize[1]
        ).configure_axis(
            titleFontSize=titlefontsize,
            titleFontWeight=titlefontweight,
            labelFontSize=labelfontsize,
        )

        # save as html
        if savefigto is not None:
            whole_chart.save(savefigto + ".html")

        if show == True:
            alt.renderers.enable("notebook")
            return whole_chart

    def pvalue_qq_plot(obs_pvalues, label_fontsize=20, ax=None, param_dict=None):
        """ Draws a p-value QQ-plot given observed p-values
            
            Args:
            obs_pvalues (list): observed p-values
            label_fontsize (int): fontsize of x and y axis labels
            ax (matplotlib.axes._subplots.AxesSubplot): (optional) AxesSubplot object for plot to be drawn on.
            """

        if ax is None:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.gca()

        obs_pvalues = np.array(obs_pvalues)
        if obs_pvalues.ndim == 1:
            xs = np.arange(1, len(obs_pvalues) + 1) / len(obs_pvalues)
            ax.plot(-np.log10(xs), -np.log10(sorted(obs_pvalues)), ".", color="black")
        else:
            for pvalues in obs_pvalues:
                print(len(pvalues))
                xs = np.arange(1, len(pvalues) + 1) / len(pvalues)
                ax.plot(-np.log10(xs), -np.log10(sorted(pvalues)), ".")

        # see **kwargs parameter in matplotlib
        # documentation (https://matplotlib.org/api/pyplot_api.html)
        # for using param_dict more efficiently in the future
        if "fillstyle" not in param_dict:
            param_dict["fillstyle"] = "--"
        if "color" not in param_dict:
            param_dict["color"] = "gray"

        ax.plot(-np.log10(xs), -np.log10(xs), **param_dict)
        ax.set_xlabel("\n-log10 (Expected P-value)", fontsize=label_fontsize)
        ax.set_ylabel("-log10 (Observed P-value)\n", fontsize=label_fontsize)
