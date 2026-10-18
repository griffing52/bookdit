#!/usr/bin/env python
"""Generate and officially score the local Painting-the-Market reproduction.

The adapter uses the released LOB-Bench GOOG windows as conditioning contexts,
samples the 100-state Painting future (`--blocks` > 1 rolls it recursively,
100 states per block), decodes those states with the repository's
book-to-message inverse-dynamics decoder (scripts/lib/idm_decoder.py), and
writes the exact data_real/data_cond/data_gen tree consumed by LOB-Bench.
The paper's Painting arm is one block (`--blocks 1`) over every window of the
nine-day source tree (`--windows 0`); see run_painting.sh.

Generation is resumable at window granularity: existing complete message/book
pairs are retained.  Scoring covers the ten pure book-state statistics of the
``book_10`` aggregate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "baselines"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

from train_painting_market import HISTORY, FUTURE, PaintingUNet  # noqa: E402
from idm_decoder import decode_book_sequence  # noqa: E402
import run_lobbench_worldmodel_official as OFFICIAL  # noqa: E402

BOOK_10 = [
    "spread", "orderbook_imbalance", "ask_volume_touch", "bid_volume_touch",
    "ask_volume", "bid_volume", "ofi", "ofi_up", "ofi_stay", "ofi_down",
]
RID_RE = re.compile(r"real_id_(\d+)")
# The official nine-day set reuses window ids across days, so a window is keyed by (date, rid);
# keying by rid alone would silently collapse nine days onto one.
WIN_RE = re.compile(r"(GOOG_\d{4}-\d{2}-\d{2})_orderbook_real_id_(\d+)")


def layout20(book: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return price/size images (20,T) and the first-state mid from (T,40)."""
    ask_p, ask_v = book[:, 0::4], book[:, 1::4]
    bid_p, bid_v = book[:, 2::4], book[:, 3::4]
    price = torch.cat((bid_p.flip(1), ask_p), 1).T
    size = torch.cat((bid_v.flip(1), -ask_v), 1).T
    mid0 = (ask_p[0, 0] + bid_p[0, 0]) / 2
    return price, size, mid0


def history_condition(book: torch.Tensor, clip_quantile: float | None = None
                      ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Normalize from the observed history only (never peek at the future)."""
    price, size, mid0 = layout20(book)
    pstd = (price - mid0).std().clamp_min(1e-6)
    smean, sstd = size.mean(), size.std().clamp_min(1e-6)
    norm = torch.stack(((price - mid0) / pstd, (size - smean) / sstd))[None]
    if clip_quantile is not None:
        q = float(clip_quantile)
        pclip = torch.quantile(norm[:, 0].abs().flatten(), q).clamp_min(1e-6)
        sclip = torch.quantile(norm[:, 1].abs().flatten(), q).clamp_min(1e-6)
        norm[:, 0] = norm[:, 0].clamp(-pclip, pclip)
        norm[:, 1] = norm[:, 1].clamp(-sclip, sclip)
    else:
        pclip = torch.tensor(float("inf"), device=book.device)
        sclip = torch.tensor(float("inf"), device=book.device)
    # Expand only the 20-level spatial axis; never resize the time axis.
    norm = F.interpolate(norm, size=(256, HISTORY), mode="nearest")[0]
    hist = torch.zeros((2, 256, HISTORY + FUTURE), dtype=norm.dtype,
                       device=norm.device)
    hist[..., :HISTORY] = norm
    return hist, (mid0, pstd, smean, sstd, pclip, sclip)


# Exact inverse of the nearest-neighbour 20 -> 256 level expansion the encoder applies.
# `F.interpolate(..., mode="nearest")` maps output row i to input row floor(i * 20/256), so level j
# occupies rows [ceil(j*12.8), ceil((j+1)*12.8)).  Collapsing back with a SECOND `interpolate` does
# NOT invert that: its index map is [0,0,1,2,3,5,5,6,7,8,10,10,11,12,13,15,15,16,17,18], which never
# reads levels 4, 9, 14, 19 and reads 0, 5, 10, 15 twice, so a real, noiseless book round-tripped
# that way already comes back with corrupted prices and a wider spread -- a penalty charged to the
# model before it generated anything.  Averaging each source band is
# exact on an encoded image (the band is constant there) and denoises a generated one.
_BANDS = [(-(-j * 256 // 20), -(-(j + 1) * 256 // 20)) for j in range(20)]


def unexpand_levels(x: torch.Tensor) -> torch.Tensor:
    """(C,256,T) -> (C,20,T): the exact inverse of the encoder's level expansion."""
    return torch.stack([x[:, a:b, :].mean(dim=1) for a, b in _BANDS], dim=1)


def image_to_book(x: torch.Tensor, scale: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Invert the paper's nearest-neighbor 20-to-256 raster expansion."""
    z = unexpand_levels(x)
    mid0, pstd, smean, sstd, pclip, sclip = scale
    price = z[0].clamp(-pclip, pclip) * pstd + mid0
    size = z[1].clamp(-sclip, sclip) * sstd + smean
    bid_p, ask_p = price[:10].flip(0).T, price[10:].T
    bid_v, ask_v = size[:10].flip(0).T, -size[10:].T
    out = torch.empty(FUTURE, 40, device=x.device)
    out[:, 0::4], out[:, 1::4] = ask_p, ask_v
    out[:, 2::4], out[:, 3::4] = bid_p, bid_v
    # Match the decoder's native quantization and keep recursive contexts finite.
    out[:, 0::4] = (out[:, 0::4] / 100).round() * 100
    out[:, 2::4] = (out[:, 2::4] / 100).round() * 100
    out[:, 1::4] = out[:, 1::4].round().clamp_min(0)
    out[:, 3::4] = out[:, 3::4].round().clamp_min(0)
    return out


@torch.inference_mode()
def ddim_future(model, hist, diffusion_steps, sample_steps, seed):
    """Deterministic DDIM (eta=0), conditioned by the clean history channels."""
    dev = hist.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((1, 2, 256, HISTORY + FUTURE), generator=gen, device=dev)
    # Fixed noise for the re-imposed known region, drawn from the same seeded generator right after
    # x, so a (seed, window, block) triple fixes the whole sample.
    known_noise = torch.randn(x.shape, generator=gen, device=dev)
    mask = torch.zeros((1, 1, 256, HISTORY + FUTURE), device=dev)
    mask[..., HISTORY:] = 1
    beta = torch.linspace(1e-4, 0.02, diffusion_steps, device=dev)
    abar = torch.cumprod(1 - beta, 0)
    times = torch.linspace(diffusion_steps - 1, 0, sample_steps, device=dev).long()
    for i, ti in enumerate(times):
        t = ti.expand(1)
        at = abar[ti]
        # RE-IMPOSE THE KNOWN REGION BEFORE EVERY MODEL CALL.  Training noises the WHOLE image
        # (train_painting_market.py:267) but masks the loss to the future columns (:270), so the
        # model gets no supervision on its epsilon in the history block.  Letting those columns
        # free-run through the DDIM update feeds unsupervised noise back in: the history block
        # reaches 27x its trained scale by the last step and the convolutions carry that into the
        # future block, which is what produced ~70% crossed books.  This is standard diffusion
        # inpainting, and it is how the paper's Painting arm was sampled.
        known_t = at.sqrt() * hist[None] + (1 - at).sqrt() * known_noise
        x = x * mask + known_t * (1 - mask)
        eps = model(torch.cat((x, hist[None], mask), 1), t)
        x0 = (x - (1 - at).sqrt() * eps) / at.sqrt()
        if i + 1 == len(times):
            x = x0
        else:
            an = abar[times[i + 1]]
            x = an.sqrt() * x0 + (1 - an).sqrt() * eps
    return x[0, :, :, HISTORY:]


def hash_day(day: str) -> int:
    """Stable per-day seed offset. rids repeat across days, so `seed + rid*100 + block` alone would
    reuse the same noise draw for the same rid on every day."""
    return int(day.replace("GOOG_", "").replace("-", "")) % 100003 * 1000


def validity(book: np.ndarray) -> dict[str, float]:
    """Structural rates on the NATIVE generated frames, per row.

    Three definitions this function pins down:

    1. Crossed is ``bid > ask``; a locked book (ask == bid) is legal if unusual and is reported
       separately.
    2. The ladder test is strict: two adjacent levels at the same price is a violation, since
       LOBSTER levels are distinct.
    3. An absent level carries price 0, which a naive ladder test reads as a huge downward step.
       Rates are therefore computed over rows whose relevant side is fully populated, with the
       populated fraction reported alongside.  ``raw_crossed_or_locked`` is the all-row figure.
    """
    ask, bid = book[:, 0::4], book[:, 2::4]
    touch = (ask[:, 0] > 0) & (bid[:, 0] > 0)
    ask_full, bid_full = (ask > 0).all(axis=1), (bid > 0).all(axis=1)

    def rate(mask, cond):
        return float(np.mean(cond[mask])) if mask.any() else float("nan")

    return {
        "locked": rate(touch, ask[:, 0] == bid[:, 0]),
        "crossed": rate(touch, bid[:, 0] > ask[:, 0]),
        "ask_nonmonotone": rate(ask_full, ~np.all(np.diff(ask, axis=1) > 0, axis=1)),
        "bid_nonmonotone": rate(bid_full, ~np.all(np.diff(bid, axis=1) < 0, axis=1)),
        "touch_populated": float(np.mean(touch)),
        "levels_populated": float(np.mean(((ask > 0) & (bid > 0)).sum(axis=1))),
        "raw_crossed_or_locked": float(np.mean(ask[:, 0] <= bid[:, 0])),
    }


def source_windows(source: Path, n: int):
    """(date, rid, cond_path) for every window in the source tree, ordered by (date, rid)."""
    out = []
    for p in (source / "data_cond").glob("*orderbook*"):
        m = WIN_RE.search(p.name)
        if m:
            out.append((m.group(1), int(m.group(2)), p))
    out.sort(key=lambda r: (r[0], r[1]))
    return out[:n] if n else out


def link_reference(source: Path, tree: Path, day: str, rid: int):
    for sub in ("data_real", "data_cond"):
        dst = tree / sub
        dst.mkdir(parents=True, exist_ok=True)
        for src in (source / sub).glob(f"{day}_*real_id_{rid}.csv"):
            target = dst / src.name
            if not target.exists():
                target.symlink_to(src.resolve())


def write_generated(tree: Path, day: str, rid: int, msgs: pd.DataFrame, books: pd.DataFrame):
    dst = tree / "data_gen"
    dst.mkdir(parents=True, exist_ok=True)
    stem = f"{day}_{{kind}}_real_id_{rid}_gen_id_0.csv"
    msgs.to_csv(dst / stem.format(kind="message"), header=False, index=False)
    books.to_csv(dst / stem.format(kind="orderbook"), header=False, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", required=True,
                   help="GOOG directory containing official data_real/data_cond")
    p.add_argument("--out", required=True)
    p.add_argument("--windows", type=int, default=0,
                   help="cap on windows; 0 = every window in the source tree")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--blocks", type=int, default=1,
                   help="100-state blocks per window; >1 rolls the sampler recursively "
                        "(the paper uses 1)")
    p.add_argument("--seed", type=int, default=20260801)
    p.add_argument("--device", default="cuda")
    p.add_argument("--generate-only", action="store_true")
    a = p.parse_args()

    source, out = Path(a.source), Path(a.out)
    tree = out / "model"
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = PaintingUNet(ck["base"]).to(a.device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    norm_cfg = ck.get("normalization", {"history": HISTORY, "clip_quantile": None})
    print(f"checkpoint_step={ck['step']} params={ck['params']} windows={a.windows} "
          f"sample_steps={a.sample_steps} blocks={a.blocks} normalization={norm_cfg}", flush=True)

    audits = []
    todo = source_windows(source, a.windows)
    print(f"windows={len(todo)} over {len({d for d, _, _ in todo})} days", flush=True)
    for j, (day, rid, cond_path) in enumerate(todo):
        link_reference(source, tree, day, rid)
        msg_out = tree / "data_gen" / f"{day}_message_real_id_{rid}_gen_id_0.csv"
        book_out = tree / "data_gen" / f"{day}_orderbook_real_id_{rid}_gen_id_0.csv"
        if msg_out.exists() and book_out.exists() and msg_out.stat().st_size and book_out.stat().st_size:
            print(f"[{j + 1}/{len(todo)}] {day} rid={rid} resume-skip", flush=True)
            continue
        cond = torch.from_numpy(pd.read_csv(cond_path, header=None).values.astype("float32"))
        hist = cond[-HISTORY:].to(a.device)
        generated = []
        for block in range(a.blocks):
            h_img, scale = history_condition(
                hist, clip_quantile=norm_cfg.get("clip_quantile"))
            future = ddim_future(model, h_img.to(a.device), ck["diffusion_steps"],
                                 a.sample_steps, a.seed + hash_day(day) + rid * 100 + block)
            raw = image_to_book(future, tuple(x.to(a.device) for x in scale))
            generated.append(raw.cpu())
            hist = torch.cat((hist, raw), 0)[-HISTORY:]
        raw = torch.cat(generated).numpy()
        audit = {"day": day, "rid": rid, **validity(raw)}
        real_msg = next((source / "data_real").glob(f"{day}_message_real_id_{rid}.csv"))
        rm = pd.read_csv(real_msg, header=None)
        t0, t1 = float(rm.iloc[0, 0]), float(rm.iloc[-1, 0])
        frame_times = np.linspace(t0, t1, len(raw))
        msgs, books = decode_book_sequence(raw, frame_times=frame_times)
        write_generated(tree, day, rid, msgs, books)
        audit["decoded_messages"] = len(msgs)
        audits.append(audit)
        print(f"[{j + 1}/{len(todo)}] {day} rid={rid} decoded={len(msgs)} "
              f"crossed={audit['crossed']:.4f} locked={audit['locked']:.4f} ask_bad={audit['ask_nonmonotone']:.4f} "
              f"bid_bad={audit['bid_nonmonotone']:.4f}", flush=True)

    if audits:
        pd.DataFrame(audits).to_csv(out / "generation_audit.csv", index=False)
    meta = {"checkpoint": str(Path(a.checkpoint).resolve()), "checkpoint_step": ck["step"],
            "params": ck["params"], "windows_requested": a.windows,
            "sample_steps": a.sample_steps, "blocks": a.blocks, "seed": a.seed,
            "source": str(source.resolve()), "stats": BOOK_10,
            "normalization": norm_cfg}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if a.generate_only:
        return
    per = OFFICIAL.score_tree(tree, BOOK_10)
    score = {s: per[s][0] for s in BOOK_10}
    result = {"book_10": float(np.mean(list(score.values()))), "per_stat": score,
              "n_windows": a.windows, **meta}
    (out / "book10.json").write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
