"""Inspect exact VQF maxima while jointly sweeping the two entropy weights.

python -m scripts.visualize_viewactive_entropy_sweep plan
# Extract missing_members.txt from the original render archive into OUT.
python -m scripts.visualize_viewactive_entropy_sweep figures
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts import visualize_viewactive_gt_maxima as original
from viewactive.grid import combine

OUT = Path("outputs/viewactive/entropy_weight_sweep_visibility1")
DEFAULT_WEIGHTS = (0, .25, .5, 1, 2, 4)


def quality(norm, weight):
    weights = np.asarray([weight, weight, 1.])
    active = weights > 0
    return combine(norm[active], weights[active])


def image_path(name):
    for root in (OUT, original.OUT, original.BASE):
        if (root/name).is_file():
            return root/name
    raise FileNotFoundError(name)


def plan(weights):
    OUT.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(original.BASE/"run/vqf.npz"))
    previous = json.loads((original.OUT/"plan.json").read_text())
    rows, missing = [], []
    for source in previous:
        norm = data["norm"][source["index"]]
        start = tuple(source["start"])
        row = {k:source[k] for k in ("number","uid","category","split","shard","index",
                                    "start","start_member","start_angles")}
        row["variants"] = []
        for w in weights:
            score = quality(norm, w)
            best = tuple(map(int,np.unravel_index(np.nanargmax(score),score.shape)))
            path = original.member(source,best,data)
            try:
                image_path(path)
            except FileNotFoundError:
                missing.append(path)
            variant = dict(weight=w,weights=[w,w,1],best=list(best),best_member=path,
                best_angles=[float(data["azimuths"][best[1]]),float(data["elevations"][best[0]])],
                start_quality=float(score[start]),best_quality=float(score[best]),
                best_channels=norm[:,best[0],best[1]].tolist(),
                changed_from_equal=list(best)!=source["best"],
                ties=int(np.sum(np.isfinite(score)&(score==score[best]))))
            if w==1:
                assert list(best)==source["best"]
                assert np.isclose(score[best],source["best_quality"])
            assert variant["best_quality"] >= variant["start_quality"]-1e-9
            row["variants"].append(variant)
        rows.append(row)
    payload = {"weights":weights,"formula":"Q_w = (w*normal_entropy + w*visual_entropy + visible_ratio)/(2*w+1)",
               "channels":"cached per-object normalized channels","rows":rows}
    (OUT/"plan.json").write_text(json.dumps(payload,indent=2)+"\n")
    missing=list(dict.fromkeys(missing))
    (OUT/"missing_members.txt").write_text("\n".join(missing)+("\n" if missing else ""))
    with (OUT/"scores.csv").open("w") as stream:
        writer=csv.writer(stream)
        writer.writerow(["number","category","uid","entropy_weight","azimuth","elevation",
                         "start_Q","maximum_Q","visible_ratio","normal_entropy","visual_entropy",
                         "changed_from_equal","ties"])
        for row in rows:
            for v in row["variants"]:
                writer.writerow([row["number"],row["category"],row["uid"],v["weight"],*v["best_angles"],
                    v["start_quality"],v["best_quality"],*v["best_channels"],v["changed_from_equal"],v["ties"]])
    print(json.dumps({"objects":len(rows),"weights":weights,"missing_images":len(missing),
        "changed_maxima_vs_w1":{str(w):sum(r["variants"][i]["changed_from_equal"] for r in rows)
                                for i,w in enumerate(weights)}},indent=2),flush=True)


def draw_image(ax, path, title, color):
    from PIL import Image
    from matplotlib.patches import Rectangle
    with Image.open(image_path(path)) as im:
        im=im.convert("RGB")
        im.thumbnail((640,560))
        ax.imshow(im)
    ax.set_title(title,fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,fill=False,edgecolor=color,linewidth=3))


def image_row(axes,row):
    az,el=row["start_angles"]
    draw_image(axes[0],row["start_member"],f"Input start\naz {az:.0f}°, el {el:+.0f}°",original.START_COLOR)
    for ax,v in zip(axes[1:],row["variants"]):
        az,el=v["best_angles"]
        draw_image(ax,v["best_member"],f"w = {v['weight']:g}"+(" (equal)" if v["weight"]==1 else "")+
                   f"\naz {az:.0f}°, el {el:+.0f}° | Q = {v['best_quality']:.3f}",original.MAX_COLOR)


def weighted_row(axes,row,variant,data,numbers=False):
    az,el=row["start_angles"]
    draw_image(axes[0],row["start_member"],f"Input start\naz {az:.0f}°, el {el:+.0f}° | Q = {variant['start_quality']:.3f}",original.START_COLOR)
    az,el=variant["best_angles"]
    draw_image(axes[1],variant["best_member"],f"Ground-truth maximum\naz {az:.0f}°, el {el:+.0f}° | Q = {variant['best_quality']:.3f}",original.MAX_COLOR)
    norm=data["norm"][row["index"]]
    marked={**row,**variant}
    for ax,field,title in zip(axes[2:],[*norm,quality(norm,variant["weight"])],original.CHANNELS):
        original.field_panel(ax,field,marked,data,title,numbers=numbers)


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    payload=json.loads((OUT/"plan.json").read_text())
    rows,weights=payload["rows"],payload["weights"]
    data=dict(np.load(original.BASE/"run/vqf.npz"))
    for row in rows:
        image_path(row["start_member"])
        for v in row["variants"]:
            image_path(v["best_member"])
    ncols=len(weights)+1
    objects=OUT/"objects"
    objects.mkdir(exist_ok=True)
    legend=[Line2D([0],[0],color=original.START_COLOR,lw=3,label="Input start"),
            Line2D([0],[0],color=original.MAX_COLOR,lw=3,label="Maximum for these weights")]
    note="Q = (w × normal entropy + w × visual entropy + visible ratio) / (2w + 1); cached normalized channels"
    links=[]
    with PdfPages(OUT/"entropy_weight_comparison.pdf") as pdf:
        for row in rows:
            fig,axes=plt.subplots(2,ncols,figsize=(3.7*ncols,8.5))
            image_row(axes[0],row)
            axes[1,0].axis("off")
            axes[1,0].text(0,.9,"Weights [normal, visual, visible]\n\n[w, w, 1]\n\nCyan: input start\nRed: weighted global maximum\n\nAll 132 views searched.\nEach heatmap uses scale [0, 1].\nNo extra field normalization.",
                           transform=axes[1,0].transAxes,va="top",fontsize=12)
            norm=data["norm"][row["index"]]
            for ax,v in zip(axes[1,1:],row["variants"]):
                original.field_panel(ax,quality(norm,v["weight"]),{**row,**v},data,
                                     f"Weighted field | w = {v['weight']:g}",numbers=True)
            fig.suptitle(f"{row['number']:02d}. {row['category']} | {row['uid']}",fontsize=17)
            fig.text(.025,.025,note,fontsize=11)
            fig.subplots_adjust(left=.025,right=.98,top=.88,bottom=.12,wspace=.30,hspace=.32)
            name=f"{row['number']:02d}_{row['category'].replace('/','_')}_{row['uid'][:8]}.png"
            fig.savefig(objects/name,dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
            links.append(f"- [{row['number']:02d}. {row['category']}](<objects/{name}>)")
            print(f"[comparison] {row['number']}/12",flush=True)
    for page in range(4):
        fig,axes=plt.subplots(3,ncols,figsize=(3.7*ncols,11.5))
        for panels,row in zip(axes,rows[page*3:(page+1)*3]):
            image_row(panels,row)
            panels[0].text(0,1.30,f"{row['number']:02d}. {row['category']} | {row['uid'][:8]}",
                           transform=panels[0].transAxes,fontsize=12,weight="bold")
        fig.suptitle(f"Ground-truth maxima across entropy weights — {page+1}/4",fontsize=18)
        fig.text(.025,.025,note,fontsize=11)
        fig.subplots_adjust(left=.025,right=.98,top=.88,bottom=.09,wspace=.15,hspace=.55)
        fig.savefig(OUT/f"comparison_{page+1}.png",dpi=140)
        plt.close(fig)
    # Preserve the previous six-panel format separately for every weight.
    with PdfPages(OUT/"all_weights_fields.pdf") as pdf:
        for wi,w in enumerate(weights):
            folder=OUT/f"weight_{w:g}"
            folder.mkdir(exist_ok=True)
            for row in rows:
                v=row["variants"][wi]
                fig,axes=plt.subplots(1,6,figsize=(25,4.6),gridspec_kw={"width_ratios":[1,1,1.15,1.15,1.15,1.15]})
                weighted_row(axes,row,v,data,numbers=True)
                fig.suptitle(f"w = {w:g} | weights [{w:g}, {w:g}, 1] | {row['number']:02d}. {row['category']} | {row['uid'][:8]}",fontsize=16)
                fig.legend(handles=legend,loc="lower left",ncol=2,frameon=False,fontsize=10)
                fig.text(.43,.025,note,fontsize=8)
                fig.subplots_adjust(left=.025,right=.98,top=.80,bottom=.19,wspace=.30)
                fig.savefig(folder/f"{row['number']:02d}_{row['uid'][:8]}.png",dpi=150)
                pdf.savefig(fig)
                plt.close(fig)
            for page in range(4):
                fig,axes=plt.subplots(3,6,figsize=(25,12),gridspec_kw={"width_ratios":[1,1,1.15,1.15,1.15,1.15]})
                for panels,row in zip(axes,rows[page*3:(page+1)*3]):
                    weighted_row(panels,row,row["variants"][wi],data)
                    panels[0].text(0,1.30,f"{row['number']:02d}. {row['category']} | {row['uid'][:8]}",
                                   transform=panels[0].transAxes,fontsize=12,weight="bold")
                fig.suptitle(f"Entropy weight w = {w:g} | weights [{w:g}, {w:g}, 1] — {page+1}/4",fontsize=18)
                fig.legend(handles=legend,loc="lower left",ncol=2,frameon=False,fontsize=11)
                fig.text(.4,.025,note,fontsize=8)
                fig.subplots_adjust(left=.025,right=.98,top=.88,bottom=.10,wspace=.30,hspace=.65)
                fig.savefig(folder/f"overview_{page+1}.png",dpi=140)
                plt.close(fig)
            print(f"[fields] finished w={w:g}",flush=True)
    (OUT/"README.md").write_text("# Entropy weight sweep: visibility weight fixed at 1\n\n"+payload["formula"]+"\n\n"
        "Both entropy channels share w; visible ratio has weight 1. All channels retain "
        "the existing per-object normalization. The 12 objects and starting images are unchanged. "
        "Weights are exploratory choices; no classifier or learned fitting is used.\n\n"
        "[Compare all weights and weighted fields (12-page PDF)](entropy_weight_comparison.pdf) | "
        "[Original figure layout for every weight (72-page PDF)](all_weights_fields.pdf) | "
        "[Scores CSV](scores.csv) | [Full selection metadata](plan.json)\n\n"
        +"\n".join(f"[Comparison {i}](comparison_{i}.png)" for i in range(1,5))
        +"\n\n"+"\n".join(f"- [w={w:g}: original-layout overview](weight_{w:g}/overview_1.png)" for w in weights)
        +"\n\n## Per-object comparisons\n\n"+"\n".join(links)+"\n")
    print(f"Saved comparison and field figures to {OUT}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("plan","figures"))
    parser.add_argument("--weights",nargs="+",type=float,default=list(DEFAULT_WEIGHTS))
    args=parser.parse_args()
    if any(not np.isfinite(w) or w<0 for w in args.weights):
        parser.error("weights must be finite and nonnegative")
    plan(args.weights) if args.stage=="plan" else figures()
