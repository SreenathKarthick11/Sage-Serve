"""Readable RFWS matrix comparisons: all models visible, one cohort per panel."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from inference.plots import FIGURE_DPI, FIGURE_SIZE, _COLORS


def _flag(value):
    return str(value).lower() == "true"


def generate_matrix_scores(metrics, out_dir):
    models = sorted(metrics.model.unique())
    if len(models) > 4:
        raise ValueError("Matrix panels support the four configured simulator models A-D.")
    names = [("slo", "Compliance"), ("stability", "Stability"), ("demand", "Prompt regularity"),
             ("cost", "Instance-hours (disabled)"), ("throughput", "Completion volume")]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for stacked, filename in ((False, "rfws_bar.png"), (True, "sub_scores.png")):
        fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        fig.subplots_adjust(left=.12, right=.97, bottom=.16, top=.86, hspace=.35, wspace=.38)
        try:
            for ax, model in zip(axes.flat, models):
                data = metrics[metrics.model == model].copy()
                data["_level"] = data.scaling_level.astype(int)
                data["_sts"] = data.sts.map(_flag).astype(int)
                data["_lts"] = data.lts.map(_flag).astype(int)
                data = data.sort_values(["_level", "_sts", "_lts"])
                labels = [f"L{r['_level']} S{r['_sts']} T{r['_lts']}" for _, r in data.iterrows()]
                y = np.arange(len(data))
                valid = data.rfws.notna().to_numpy()
                if stacked:
                    left = np.zeros(int(valid.sum()))
                    for color, (name, _) in zip(_COLORS, names):
                        values = data.loc[valid, "contribution_" + name].to_numpy()
                        ax.barh(y[valid], values, left=left, color=color)
                        left += values
                else:
                    ax.barh(y[valid], data.loc[valid, "rfws"], color=_COLORS[0])
                    for position, value in zip(y[valid], data.loc[valid, "rfws"]):
                        ax.text(value + 1, position, f"{value:.1f}", va="center", fontsize=6)
                for position in y[~valid]:
                    ax.text(1, position, "Unavailable: see report status", fontsize=6, va="center")
                ax.set_yticks(y, labels, fontsize=7)
                ax.tick_params(axis="x", labelsize=7)
                ax.set_xlim(0, 110)
                ax.set_ylim(len(data) - .4, -.6)
                ax.set_title(f"Model {model}", fontsize=10)
                ax.grid(axis="x", alpha=.15)
                ax.set_axisbelow(True)
            for ax in list(axes.flat)[len(models):]:
                ax.set_visible(False)
            fig.suptitle("Weighted RFWS contributions" if stacked else "Relative Final Weighted Score (RFWS)", fontsize=13)
            fig.text(.5, .91, "L = scaling level, S = STS, T = LTS (0 off, 1 on). Compare within each model's cohort.",
                     ha="center", fontsize=8)
            if stacked:
                fig.legend(handles=[Patch(facecolor=c, label=label) for c, (_, label) in zip(_COLORS, names)],
                           loc="lower center", ncol=3, fontsize=7)
            else:
                fig.text(.5, .04, "Cost-disabled score. Equal measurements can legitimately produce equal scores.",
                         ha="center", fontsize=8)
            path = out_dir / filename
            with matplotlib.rc_context({"savefig.bbox": None}):
                fig.savefig(path, dpi=FIGURE_DPI, bbox_inches=None)
            outputs.append(path)
        finally:
            plt.close(fig)
    return outputs
