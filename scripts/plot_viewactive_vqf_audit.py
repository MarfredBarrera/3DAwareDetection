"""Plot the cached VQF audit without rerunning models or rendering."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("outputs/viewactive/vqf_audit")


def main():
    report = json.loads((ROOT/"cached_audit.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, (split, title) in enumerate((("test", "Seen: 400 objects"), ("test_unseen", "Unseen: 200 objects"))):
        r = report[split]
        el = [v["elevation"] for v in r["elevations"]]
        ax = axes[row,0]
        ax.plot(el, [v["accuracy"]["ViT"] for v in r["elevations"]], "ko-",label="ViT accuracy")
        for i,name in enumerate(("Visible area", "Normal entropy", "Visual entropy")):
            ax.plot(el,[v["normalized_quality"][i] for v in r["elevations"]],label=name,alpha=.8)
        ax.set(title=title, xlabel="Camera elevation (degrees)", ylabel="Accuracy / normalized quality",ylim=(0,.85))
        ax.legend(fontsize=8,loc="lower center",ncol=2)
        ax.grid(alpha=.2)
        ax=axes[row,1]
        names=["Uniform view", "Exact VQF, 9 moves", "Global VQF maximum", "Any correct view exists"]
        for i,classifier in enumerate(("ResNet50", "ViT")):
            values=[r["all_view_accuracy"][classifier],r["exact_field_all_start_rollouts"][1]["accuracy"][classifier],
                    r["global_best"]["normalized_combined"]["accuracy"][classifier],r["any_correct_view"][classifier]]
            bars=ax.bar(np.arange(4)+.18*(2*i-1),values,width=.35,label=classifier)
            ax.bar_label(bars,labels=[f"{100*v:.1f}%" for v in values],fontsize=8,padding=3)
        ax.set(xticks=range(4),xticklabels=["Uniform\nview","Exact VQF\n9 moves","Global VQF\nmaximum","Any correct\nview exists"],ylim=(0,.95),ylabel="Recognition accuracy")
        ax.legend(fontsize=8)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("Exact VQF improves recognition, but does not select all recognizable views",fontsize=14)
    fig.savefig(ROOT/"vqf_diagnosis.png",dpi=160)
    fig.savefig(ROOT/"vqf_diagnosis.pdf")


if __name__=="__main__":
    main()
