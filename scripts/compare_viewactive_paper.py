"""Summarize Fig. 4 reproduction against values read from the paper's plot.

The reference values are approximate because the paper does not give tables.
This comparison is descriptive; model selection must use the validation split.
"""
import argparse
import json
from pathlib import Path


REFERENCE = {
    "seen": {
        "quality": {"visible_ratio": (0.570, 0.715), "normal_entropy": (0.485, 0.645),
                    "visual_entropy": (0.580, 0.683)},
        "accuracy": {"ResNet50": (0.155, 0.475), "EfficientNetV2": (0.150, 0.450),
                     "Swin": (0.145, 0.487), "ViT": (0.110, 0.465)},
    },
    "unseen": {
        "quality": {"visible_ratio": (0.605, 0.692), "normal_entropy": (0.528, 0.615),
                    "visual_entropy": (0.592, 0.628)},
        "accuracy": {"ResNet50": (0.095, 0.420), "EfficientNetV2": (0.095, 0.445),
                     "Swin": (0.115, 0.435), "ViT": (0.050, 0.440)},
    },
}


def compare(path):
    data = json.loads(Path(path).read_text())
    rows = []
    for split, reference in REFERENCE.items():
        policies = data[split]["policies"]
        for group, key in (("quality", "channels"), ("accuracy", "accuracy")):
            for name, (start, end) in reference[group].items():
                measured = policies["viewactive"][key][name]["mean"]
                rows.append({"split": split, "metric": group + "/" + name,
                             "paper_step0": start, "paper_step9": end,
                             "measured_step0": measured[0], "measured_step9": measured[-1],
                             "step9_gap": measured[-1] - end,
                             "measured_gain": measured[-1] - measured[0]})
        for baseline in ("random_walk", "stay"):
            a = policies["viewactive"]["mean_accuracy"]["mean"][-1]
            b = policies[baseline]["mean_accuracy"]["mean"][-1]
            rows.append({"split": split, "metric": "gain_over/" + baseline,
                         "measured_step9": a - b})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fig4", type=Path)
    args = parser.parse_args()
    rows = compare(args.fig4)
    print("split,metric,paper_step0,measured_step0,paper_step9,measured_step9,step9_gap,measured_gain")
    for row in rows:
        print(",".join(str(row.get(key, "")) for key in
                       ("split", "metric", "paper_step0", "measured_step0", "paper_step9",
                        "measured_step9", "step9_gap", "measured_gain")))


if __name__ == "__main__":
    main()
