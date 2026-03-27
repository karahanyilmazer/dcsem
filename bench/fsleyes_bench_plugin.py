#!/usr/bin/env python

import glob
import os.path as op

import fsl.utils.settings as fslsettings
import matplotlib.backends.backend_wxagg as wxagg
import matplotlib.pyplot as plt
import numpy as np
import wx
from fsl.data.image import Image
from fsleyes.controls.controlpanel import ControlPanel
from fsleyes.views.orthopanel import OrthoPanel


class BenchPanel(ControlPanel):
    @staticmethod
    def title():
        return "BENCH"

    @staticmethod
    def supportedViews():
        return [OrthoPanel]

    def __init__(self, parent, overlayList, displayCtx, viewPanel):
        super().__init__(parent, overlayList, displayCtx, viewPanel)

        self.param_names = []
        self.probmaps = []
        self.amoutmaps = []
        self.loadButton = wx.Button(
            self, label="Load Results Directory", size=(150, 50)
        )
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.figure, self.axis = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            gridspec_kw={"hspace": 0.5},
            subplot_kw=dict(frameon=False),
            figsize=(2, 3),
        )

        self.canvas = wxagg.FigureCanvasWxAgg(self, -1, self.figure)
        self.sizer.Add(self.loadButton, flag=wx.CENTER | wx.TOP, border=5, proportion=0)
        self.sizer.Add(self.canvas, flag=wx.EXPAND, proportion=1)

        for ax in self.axis:
            ax.clear()
            ax.axis("off")
        self.canvas.draw()

        self.Bind(wx.EVT_SIZE, self.on_size)

        self.SetSizer(self.sizer)
        self.loadButton.Bind(wx.EVT_BUTTON, self.onLoad)

        displayCtx.listen("location", self.name, self.locationChanged)
        self.onLoad(None)

    def on_size(self, event):
        self.figure.tight_layout()
        self.canvas.draw()
        event.Skip()

    def destroy(self):
        self.loadButton.Unbind(wx.EVT_BUTTON)
        self.Unbind(wx.EVT_SIZE)

        if self.figure:
            plt.close(self.figure)
            self.figure = None

        if self.displayCtx is not None:
            self.displayCtx.removeListener("location", self.name)

        self.canvas.Destroy()
        self.loadButton.Destroy()

        super().destroy()

    def onLoad(self, event):
        fromdir = fslsettings.read("fsleyes.bench.dir")
        if fromdir is None:
            overlay = self.displayCtx.getSelectedOverlay()
            if overlay is not None:
                fromdir = op.dirname(overlay.dataSource)
            else:
                fromdir = op.expanduser("~")

        dlg = wx.DirDialog(
            wx.GetApp().GetTopWindow(),
            message="Load BENCH results directory",
            defaultPath=fromdir,
            style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
        )

        # user cancelled dialog
        if dlg.ShowModal() != wx.ID_OK:
            return

        benchdir = dlg.GetPath()

        with open(f"{benchdir}/model_names.txt", "r") as file:
            lines = file.readlines()
            self.param_names = [s.split(":")[1][:-1] for s in lines]

        self.probmaps = [
            Image(f"{benchdir}/{m}_probability.nii.gz") for m in self.param_names
        ]
        amountmaps = [
            Image(f"{benchdir}/{m}_amount.nii.gz") for m in self.param_names[1:]
        ]
        amountmaps.insert(0, amountmaps[0].data * 0)
        self.amoutmaps = amountmaps

        fslsettings.write("fsleyes.bench.dir", benchdir)
        self.overlayList.append(
            Image(f"{benchdir}/inferred_change.nii.gz"), overlayType="label"
        )

        self.locationChanged()

    def locationChanged(self):
        dctx = self.displayCtx
        ax1, ax2 = self.axis
        canvas = self.canvas
        overlay = dctx.getSelectedOverlay()

        ax1.clear()
        ax2.clear()

        ax2.ticklabel_format(scilimits=(-2, 0))

        if overlay is None or (not isinstance(overlay, Image)):
            canvas.draw()
            return

        # get voxel cooredinates of current cursor location
        opts = dctx.getOpts(overlay)
        voxel = opts.getVoxel()

        # out of bounds
        if voxel is None:
            return
        x, y, z = voxel[:3]

        probvals = [m[x, y, z] for m in self.probmaps]
        amountvals = [m[x, y, z] for m in self.amoutmaps]
        xtickslabel = [s.replace("nochange", "[]") for s in self.param_names]

        bar_color = "#3498db"
        background_color = "#ecf0f1"

        for ax, l, v in zip(
            self.axis,
            ("Change Probability", "Amount of Change"),
            (probvals, amountvals),
        ):
            ax.bar(self.param_names, v, color=bar_color)
            ax.set_ylabel(l)
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.1)
            ax.set_xticks([])
            ax.set_facecolor(background_color)
            ax.yaxis.tick_right()

        ax2.yaxis.get_offset_text().set_x(1.1)
        ax2.set_xticks(np.arange(0, len(self.param_names)))
        ax2.set_xticklabels(xtickslabel)
        ax2.xaxis.tick_top()
        ax2.tick_params(axis="x", which="major", pad=10)

        self.figure.tight_layout()
        canvas.draw()
