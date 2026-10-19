"""Train the ViewActive decoders on cached features (paper Sec. III-D).

One sample is one rendered view (o, e, a): its frozen-encoder feature and the
object's normalized VQF rolled into that camera's relative frame. AdamW, lr
1e-3 with exponential decay 0.9 per epoch, Eq. (4) loss; the checkpoint with
the lowest validation loss is kept (the last one when there is no val split).
"""
import json
import time

import numpy as np
import torch

from viewactive import losses
from viewactive.labels import CHANNELS
from viewactive.model import ViewActiveNet


def relative_targets(fields, az_index):
    """``fields [B, C, E, A]`` rolled so column 0 is absolute column ``az_index [B]``."""
    columns = fields.shape[-1]
    index = (az_index[:, None] + torch.arange(columns, device=fields.device)) % columns
    index = index[:, None, None, :].expand(-1, fields.shape[1], fields.shape[2], -1)
    return torch.gather(fields, -1, index)


def _samples(features, fields, flat):
    """Features ``[O, E, A, D]`` and fields ``[O, C, E, A]`` -> batch for flat view indices."""
    rows, columns = features.shape[1:3]
    o, e, a = flat // (rows * columns), (flat // columns) % rows, flat % columns
    return features[o, e, a].float(), relative_targets(fields[o], a)


def evaluate_loss(net, features, fields, weights, lam, batch=4096):
    total, parts, count = 0.0, {}, 0
    views = features.shape[0] * features.shape[1] * features.shape[2]
    with torch.inference_mode():
        for start in range(0, views, batch):
            flat = torch.arange(start, min(start + batch, views), device=features.device)
            x, y = _samples(features, fields, flat)
            loss, items = losses.total(net(x), y, torch.isfinite(y), weights, lam)
            total += float(loss) * len(flat)
            for k, v in items.items():
                parts[k] = parts.get(k, 0.0) + v * len(flat)
            count += len(flat)
    return total / count, {k: v / count for k, v in parts.items()}


def train(config, features, fields, splits, out, *, encoders, grid, device="cuda", log=print):
    """Fit on ``splits == "train"``, select on ``"val"``; write ``out`` and return its path.

    ``features [O, E, A, D]`` (numpy fp16), ``fields [O, 3, E, A]`` normalized VQF.
    """
    settings = config["train"]
    torch.manual_seed(settings["seed"])
    generator = torch.Generator(device=device).manual_seed(settings["seed"])
    splits = np.asarray(splits)

    def load(name):
        index = np.flatnonzero(splits == name)
        return (torch.from_numpy(np.asarray(features[index])).to(device),
                torch.from_numpy(np.asarray(fields[index], dtype=np.float32)).to(device))

    train_x, train_y = load("train")
    val_x, val_y = load("val")
    dim = train_x.shape[-1]
    net = ViewActiveNet(dim, config["model"]["hidden"], grid.shape).to(device)
    flat_train = train_x.reshape(-1, dim).float()
    net.feature_mean.copy_(flat_train.mean(0))
    net.feature_std.copy_(flat_train.std(0).clamp_min(1e-6))
    del flat_train
    optimizer = torch.optim.AdamW(net.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=settings["gamma"])
    weights, lam = settings["loss_weights"], settings["silog_lambda"]
    views = train_x.shape[0] * train_x.shape[1] * train_x.shape[2]
    log(f"[train] {train_x.shape[0]} train objects ({views} views), {val_x.shape[0]} val objects; "
        f"{sum(p.numel() for p in net.decoders[0].parameters())} parameters per decoder")
    best, history = None, []
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, settings["epochs"] + 1):
        start, running, steps = time.time(), 0.0, 0
        net.train()
        order = torch.randperm(views, device=device, generator=generator)
        for i in range(0, views, settings["batch"]):
            x, y = _samples(train_x, train_y, order[i:i + settings["batch"]])
            loss, _ = losses.total(net(x), y, torch.isfinite(y), weights, lam)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running, steps = running + float(loss), steps + 1
        scheduler.step()
        net.eval()
        record = {"epoch": epoch, "train_loss": running / steps, "lr": scheduler.get_last_lr()[0],
                  "seconds": round(time.time() - start, 2)}
        if val_x.shape[0]:
            record["val_loss"], record["val_parts"] = evaluate_loss(net, val_x, val_y, weights, lam)
        history.append(record)
        score = record.get("val_loss", -epoch)  # no val split: keep the last epoch
        if best is None or score < best:
            best = score
            torch.save({"state_dict": net.state_dict(), "dim": dim, "hidden": config["model"]["hidden"],
                        "grid": {"azimuths": list(grid.azimuths), "elevations": list(grid.elevations)},
                        "channels": list(CHANNELS), "encoders": list(encoders), "epoch": epoch,
                        "record": record, "config": {k: config[k] for k in ("labels", "model", "train")}},
                       out)
        log(f"[train] {json.dumps(record)}")
    (out.parent / "history.json").write_text(json.dumps(history, indent=1) + "\n")
    return out
