"""Create a paired visual check of PyTorch3D and Blender viewpoints."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def image_paths(run_dir, azimuth=0.0, elevation=0.0):
    paths = {}
    for index in sorted((Path(run_dir) / "renders").glob("shard-*/index.jsonl")):
        for line in index.read_text().splitlines():
            row = json.loads(line)
            if row["azimuth"] == azimuth and row["elevation"] == elevation:
                paths[row["uid"]] = index.parent / row["files"]["rgb"]
    return paths


def make(source, target, out, count=8):
    source_paths = image_paths(source)
    target_paths = image_paths(target)
    objects = json.loads((Path(target) / "objects.json").read_text())
    uids = sorted(set(source_paths) & set(target_paths))[:count]
    if not uids:
        raise ValueError("no matching rendered object/view pairs")
    sheet = Image.new("RGB", (640, 310 * len(uids) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "PyTorch3D baseline", fill="black")
    draw.text((328, 8), "Blender", fill="black")
    for i, uid in enumerate(uids):
        for col, paths in enumerate((source_paths, target_paths)):
            with Image.open(paths[uid]) as image:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                rgb = background.convert("RGB")
                rgb.thumbnail((300, 280))
                sheet.paste(rgb, (320 * col + (320 - rgb.width) // 2,
                                  30 + 310 * i + (280 - rgb.height) // 2))
        draw.text((8, 30 + 310 * i + 282), f"{objects[uid]['category']}  {uid[:8]}", fill="black")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"Wrote {out} ({len(uids)} pairs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    make(args.source, args.target, args.out, args.count)


if __name__ == "__main__":
    main()
