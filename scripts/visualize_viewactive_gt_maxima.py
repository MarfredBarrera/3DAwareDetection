"""Pair the existing random starts with exact cached VQF maxima and fields.

python -m scripts.visualize_viewactive_gt_maxima plan
python -m scripts.visualize_viewactive_gt_maxima figures
python -m scripts.visualize_viewactive_gt_maxima plan --elevations 0 45   # likewise figures

With ``--elevations LOW HIGH`` the red maximum is searched only in that band
(optimize.elevation_range) and the unconstrained global maximum is shown in
orange beside it; outputs go to ``gt_maxima_viz_el<LOW>_<HIGH>``.

Extract the missing archive members listed by ``plan`` before making figures.
The cohort and starts are inherited from the existing 12-object visualization.

``--study outputs/upright`` makes the same figures for a viewpoint study's
ground-truth views instead: raw channels are 1 - the stored ambiguity scores
(visible ratio, normal entropy, visual entropy), normalized as in
``viewactive.labels``; one random object and start view per category (seed 0).
Outputs go to ``<study>/figures/gt_vqf_maxima[_el<LOW>_<HIGH>]``.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from viewactive.grid import Grid, combine, elevation_rows

BASE = Path("outputs/viewactive/blender_full_viz")
OUT = Path("outputs/viewactive/gt_maxima_viz")
CHANNELS = ("Visible ratio", "Normal entropy", "Visual entropy", "Combined")
START_COLOR = "#00bfff"
MAX_COLOR = "#ff3355"
GLOBAL_COLOR = "#ff9900"
LIMITS = None
STUDY = None
STUDY_METRICS = ("self_occlusion", "surface_normal_entropy", "visual_entropy")


def load_data():
    """VQF arrays in ``vqf.npz`` layout: uids, categories, splits, raw/norm [O, 3, E, A], angles."""
    if STUDY is None:
        return dict(np.load(BASE / "run/vqf.npz"))
    from study import config as settings_io
    from study import db as store
    from viewactive.labels import normalize

    config = settings_io.read(STUDY / "config.json")
    azimuths = [float(a) for a in config["dataset"]["azimuths"]]
    elevations = [float(e) for e in config["dataset"]["elevations"]]
    db = store.connect(STUDY)
    scores = store.view_scores(db, "i.source='gt'")
    rows = store.images(db, "source='gt'")
    db.close()
    uids = sorted({r["uid"] for r in rows})
    index = {uid: o for o, uid in enumerate(uids)}
    categories = {r["uid"]: r["category"] for r in rows}
    raw = np.full((len(uids), 3, len(elevations), len(azimuths)), np.nan, dtype=np.float32)
    for r in rows:
        for c, metric in enumerate(STUDY_METRICS):
            value = scores.get(r["image_id"], {}).get(metric)
            if value is not None:
                raw[index[r["uid"]], c, elevations.index(r["elevation"]), azimuths.index(r["azimuth"])] = 1 - value
    return {"uids": np.array(uids), "categories": np.array([categories[u] for u in uids]),
            "splits": np.array(["upright"] * len(uids)), "raw": raw, "norm": normalize(raw),
            "azimuths": np.array(azimuths), "elevations": np.array(elevations)}


def starting_rows(data):
    """The objects and start views to show: the earlier ViewActive sample, or for a study
    one random object and visible start view per category."""
    if STUDY is None:
        return json.loads((BASE / "plan.json").read_text())
    rng = np.random.default_rng(0)
    rows = []
    for category in sorted(set(data["categories"])):
        members = np.flatnonzero(data["categories"] == category)
        o = int(rng.choice(members))
        cells = np.argwhere(np.isfinite(combine(data["norm"][o])))
        start = [int(v) for v in cells[rng.integers(len(cells))]]
        rows.append({"uid": str(data["uids"][o]), "category": category, "split": "upright",
                     "shard": None, "views": [start]})
    return rows


def member(row, view, data):
    e, a = view
    name = f"{row['uid']}/a{data['azimuths'][a]:.1f}_e{data['elevations'][e]:.1f}_o0.0.rgb.png"
    if STUDY is not None:
        return f"renders/gt/samples/{name}"
    return f"blender_full/renders/shard-{row['shard']:03d}/samples/{name}"


def plan():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    lookup = {str(uid): o for o, uid in enumerate(data["uids"])}
    old = starting_rows(data)
    rows, missing = [], []
    for number, row in enumerate(old, 1):
        o = lookup[row["uid"]]
        assert STUDY is not None or o // 25 == row["shard"]
        norm = data["norm"][o]
        combined = combine(norm)
        start = tuple(row["views"][0])
        allowed = elevation_rows(Grid(elevations=tuple(data["elevations"])), LIMITS)[:, None]
        best = tuple(map(int, np.unravel_index(np.nanargmax(np.where(allowed, combined, np.nan)), combined.shape)))
        top = tuple(map(int, np.unravel_index(np.nanargmax(combined), combined.shape)))
        assert STUDY is not None or np.isclose(combined[start], row["quality"][0])
        start_member = member(row, start, data)
        best_member = member(row, best, data)
        global_member = member(row, top, data)
        for path in (start_member, best_member, global_member):
            if not any((d/path).exists() for d in sources()):
                missing.append(path)
        entry = {k: row[k] for k in ("uid", "category", "split", "shard")}
        entry.update(number=number, index=o, start=list(start), best=list(best),
            start_member=start_member, best_member=best_member,
            start_angles=[float(data["azimuths"][start[1]]), float(data["elevations"][start[0]])],
            best_angles=[float(data["azimuths"][best[1]]), float(data["elevations"][best[0]])],
            start_quality=float(combined[start]), best_quality=float(combined[best]),
            global_member=global_member, global_angles=[float(data["azimuths"][top[1]]),
            float(data["elevations"][top[0]])], global_quality=float(combined[top]), **{"global": list(top)},
            start_channels=norm[:,start[0],start[1]].tolist(),
            best_channels=norm[:,best[0],best[1]].tolist(),
            start_raw_channels=data["raw"][o,:,start[0],start[1]].tolist(),
            best_raw_channels=data["raw"][o,:,best[0],best[1]].tolist(),
            ties=int(np.sum(np.isfinite(combined) & allowed & (combined == combined[best]))))
        rows.append(entry)
    (OUT/"plan.json").write_text(json.dumps(rows, indent=2)+"\n")
    missing=list(dict.fromkeys(missing))
    (OUT/"missing_members.txt").write_text("\n".join(missing)+("\n" if missing else ""))
    with (OUT/"scores.csv").open("w") as stream:
        writer=csv.writer(stream)
        writer.writerow(["number","category","uid","view","azimuth","elevation",*CHANNELS])
        for row in rows:
            for which in ("start","best"):
                writer.writerow([row["number"],row["category"],row["uid"],which,
                    *row[f"{which}_angles"],*row[f"{which}_channels"],row[f"{which}_quality"]])
    print(json.dumps({"objects":len(rows),"missing_images":len(missing),"rows":[
        {k:r[k] for k in ("category","start_angles","best_angles","start_quality","best_quality")}
        for r in rows]},indent=2))


def sources():
    """Directories already holding extracted renders (earlier figure runs included)."""
    if STUDY is not None:
        return (STUDY,)
    root = BASE.parent
    return (OUT, BASE, root/"gt_maxima_viz", root/"entropy_weight_sweep",
            root/"entropy_weight_sweep_visibility1", *root.glob("gt_maxima_viz_el*"))


def image_path(name):
    for directory in sources():
        if (directory/name).is_file():
            return directory/name
    raise FileNotFoundError(name)


def image_panel(ax, row, which):
    from PIL import Image
    from matplotlib.patches import Rectangle
    with Image.open(image_path(row[f"{which}_member"])) as img:
        img=img.convert("RGB")
        img.thumbnail((960,840))
        ax.imshow(img)
    az,el=row[f"{which}_angles"]
    title={"start":"Input start","global":"Unconstrained global maximum",
           "best":"Ground-truth global maximum" if LIMITS is None else
                  f"Maximum in el {LIMITS[0]:+.0f}..{LIMITS[1]:+.0f}°"}[which]
    color={"start":START_COLOR,"best":MAX_COLOR,"global":GLOBAL_COLOR}[which]
    ax.set_title(f"{title}\naz {az:.0f}°, el {el:+.0f}° | Q = {row[f'{which}_quality']:.3f}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,fill=False,edgecolor=color,linewidth=4))


def field_panel(ax, values, row, data, title, numbers=False):
    from matplotlib.patches import Rectangle
    im=ax.imshow(values,origin="lower",cmap="viridis",vmin=0,vmax=1,aspect="auto")
    ax.set_title(title,fontsize=11)
    ax.set_xticks(np.arange(len(data["azimuths"])),labels=[f"{v:.0f}" for v in data["azimuths"]],fontsize=7)
    ax.set_yticks(np.arange(len(data["elevations"])),labels=[f"{v:+.0f}" for v in data["elevations"]],fontsize=7)
    ax.set_xlabel("Azimuth (°)",fontsize=8)
    ax.set_ylabel("Elevation (°)",fontsize=8)
    if numbers:
        peak=float(np.nanmax(values))
        precision=4 if title.startswith("Raw") and peak<.01 else 3 if title.startswith("Raw") and peak<.1 else 2
        for e in range(values.shape[0]):
            for a in range(values.shape[1]):
                v=values[e,a]
                ax.text(a,e,f"{v:.{precision}f}" if np.isfinite(v) else "—",ha="center",va="center",
                        fontsize=6,color="black" if v>.55 else "white")
    if LIMITS is not None:
        rows=np.flatnonzero(elevation_rows(Grid(elevations=tuple(data["elevations"])),LIMITS))
        for y in (rows[0]-.5,rows[-1]+.5):
            ax.axhline(y,color="white",linestyle="--",linewidth=1.5,zorder=4)
    marks=[("start",START_COLOR,0),("best",MAX_COLOR,.08)]
    if LIMITS is not None and row["global"]!=row["best"]:
        marks.append(("global",GLOBAL_COLOR,.16))
    for which,color,inset in marks:
        e,a=row[which]
        # Inset the maxima borders so all marks remain visible if they coincide.
        ax.add_patch(Rectangle((a-.5+inset,e-.5+inset),1-2*inset,1-2*inset,
                              fill=False,edgecolor=color,linewidth=2.3,zorder=5))
    return im


def make_row(fig, axes, row, data, numbers=False, raw=False):
    images=IMAGES()
    for ax,which in zip(axes,images):
        image_panel(ax,row,which)
    norm=data["norm"][row["index"]]
    values=data["raw"][row["index"]] if raw else norm
    fields=[*values,combine(norm)]
    for i,(ax,field,title) in enumerate(zip(axes[len(images):],fields,CHANNELS)):
        title=("Raw " if raw and i<3 else "")+title
        if raw and i==3:
            title="Combined (normalized selection)"
        im=field_panel(ax,field,row,data,title,numbers)
    return im


def IMAGES():
    return ("start","best") if LIMITS is None else ("start","global","best")


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    rows=json.loads((OUT/"plan.json").read_text())
    data=load_data()
    views=len(data["azimuths"])*len(data["elevations"])
    sample="original 12-object sample" if STUDY is None else f"one random object and start per category ({STUDY.name})"
    for row in rows:
        for which in IMAGES():
            image_path(row[f"{which}_member"])
    legend=[Line2D([0],[0],color=START_COLOR,lw=3,label="Input start"),
            Line2D([0],[0],color=MAX_COLOR,lw=3,label="Combined global maximum" if LIMITS is None
                   else f"Combined maximum in elevation band {LIMITS[0]:+.0f}..{LIMITS[1]:+.0f}°")]
    if LIMITS is not None:
        legend.append(Line2D([0],[0],color=GLOBAL_COLOR,lw=3,label="Unconstrained global maximum"))
    n=len(IMAGES())
    ratios=[1]*n+[1.15]*4
    individual=OUT/"objects"
    individual.mkdir(exist_ok=True)
    links=[]
    with PdfPages(OUT/"ground_truth_maxima.pdf") as pdf:
        for row in rows:
            fig,axes=plt.subplots(2,n+4,figsize=(21+2*n,9),gridspec_kw={"width_ratios":ratios})
            make_row(fig,axes[0],row,data,numbers=True)
            # Second row gives raw channels without changing the selected maximum.
            for ax in axes[1,:n]:
                ax.axis("off")
            norm=data["norm"][row["index"]]
            raw=data["raw"][row["index"]]
            for ax,field,title in zip(axes[1,n:],[*raw,combine(norm)],
                    ["Raw visible ratio","Raw normal entropy","Raw visual entropy","Combined (same field)"]):
                field_panel(ax,field,row,data,title,numbers=True)
            text=("Top: per-object normalized VQF channels.\n"
                  "Q = equal mean of the three normalized channels.\n\n"
                  "Bottom: cached raw channels; entropy channels were\n"
                  "globally divided by their maximum possible entropy.\n\n"
                  "Camera angles are in the object frame.\n"
                  +(f"The maximum is selected over all {views} views.\n" if LIMITS is None else
                    f"Red: maximum over elevations {LIMITS[0]:+.0f}..{LIMITS[1]:+.0f}° (dashed band).\n"
                    "Orange: unconstrained maximum, if different.\n")+
                  ("Objects and starting views are the same 12 examples\n"
                    "as the previous random-start visualization.\n" if STUDY is None else
                    "One random object and start view per category\n"
                    "of the viewpoint study (seed 0).\n")+
                  "No classifier output is used in this figure.")
            axes[1,0].text(0,.95,text,ha="left",va="top",fontsize=10,transform=axes[1,0].transAxes)
            axes[1,1].text(0,.95,"Normalized channels at start → maximum\n\n"+
                "\n".join(f"{n}: {s:.3f} → {b:.3f}" for n,s,b in zip(CHANNELS[:3],row["start_channels"],row["best_channels"]))+
                f"\n\nCombined: {row['start_quality']:.3f} → {row['best_quality']:.3f}\n"
                f"Tied global maximizing cells: {row['ties']}",ha="left",va="top",fontsize=10,
                transform=axes[1,1].transAxes)
            name=f"{row['number']:02d}_{row['category'].replace('/','_')}_{row['uid'][:8]}"
            links.append(f"- [{row['number']:02d}. {row['category']} — {row['uid'][:8]}](<objects/{name}.png>)")
            fig.suptitle(f"{row['number']:02d}. {row['category']} | { {'test_unseen':'unseen','upright':'upright'}.get(row['split'],'seen')} | {row['uid']}",fontsize=16)
            fig.legend(handles=legend,loc="lower left",ncol=2,frameon=False,fontsize=11)
            fig.subplots_adjust(left=.025,right=.975,top=.90,bottom=.10,wspace=.32,hspace=.40)
            cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap="viridis"),
                            ax=axes[:,n:].ravel().tolist(),fraction=.012,pad=.015)
            cb.set_label("Quality (higher is better)")
            fig.savefig(individual/f"{name}.png",dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
    pages=(len(rows)+2)//3
    for page in range(pages):
        chosen=rows[page*3:(page+1)*3]
        fig,axes=plt.subplots(3,n+4,figsize=(21+2*n,12),gridspec_kw={"width_ratios":ratios})
        for panels,row in zip(axes,chosen):
            make_row(fig,panels,row,data,numbers=False)
            panels[0].text(0,1.30,f"{row['number']:02d}. {row['category']} | {row['uid'][:8]}",
                           transform=panels[0].transAxes,fontsize=13,weight="bold")
        fig.suptitle(f"Ground-truth VQF maxima and original starting images — {page+1}/{pages}",fontsize=18)
        fig.legend(handles=legend,loc="lower left",ncol=2,frameon=False,fontsize=12)
        searched=f"all {views} grid views searched" if LIMITS is None else f"red searched in elevations {LIMITS[0]:+.0f}..{LIMITS[1]:+.0f}°"
        fig.text(.40,.025,f"Q = mean of normalized channels | {searched} | {sample}",fontsize=11)
        fig.subplots_adjust(left=.025,right=.98,top=.88,bottom=.10,wspace=.30,hspace=.65)
        fig.savefig(OUT/f"overview_{page+1}.png",dpi=140)
        plt.close(fig)
    (OUT/"README.md").write_text(
        "# Ground-truth VQF maxima\n\n"
        +("The same 12 objects and random starting images as the earlier visualization. " if STUDY is None else
          f"One random object and starting view per category of `{STUDY}` (seed 0). ")+
        "The selected view maximizes the equal mean of the three cached, per-object "
        f"normalized VQF channels over all {views} views. Cyan marks the start; red marks "
        "the combined maximum. No classifier output is used.\n\n"
        "[All objects, annotated grids and raw channels ({len(rows)}-page PDF)](ground_truth_maxima.pdf) "
        "| [View scores (CSV)](scores.csv) | [Selection details (JSON)](plan.json)\n\n"
        + "\n".join(f"[Overview {i}](overview_{i}.png)" for i in range(1,pages+1))
        + "\n\n## Detailed figures\n\n" + "\n".join(links) + "\n")
    print(f"Saved {len(rows)} detailed figures, {pages} overview pages, and {OUT/'ground_truth_maxima.pdf'}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("plan","figures"))
    parser.add_argument("--elevations",nargs=2,type=float,metavar=("LOW","HIGH"),
                        help="search the maximum only in this elevation band (degrees)")
    parser.add_argument("--study",type=Path,help="viewpoint study output dir (e.g. outputs/upright)")
    args=parser.parse_args()
    if args.study:
        STUDY=args.study
        OUT=STUDY/"figures"/"gt_vqf_maxima"
    if args.elevations:
        LIMITS=tuple(args.elevations)
        OUT=OUT.with_name(f"{OUT.name}_el{LIMITS[0]:g}_{LIMITS[1]:g}")
    {"plan":plan,"figures":figures}[args.stage]()
