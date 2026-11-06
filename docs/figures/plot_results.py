"""Builds docs/figures/results-{light,dark}.svg for the README. Standard library only.

    python docs/figures/plot_results.py

Ablation numbers are the eval.py output on the held-out 20% split; the loss curve is read
from model/full_data/config.json.
"""
import json
import os
import re
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

STAGES = ["top-1", "+ local mode", "+ CSLS", "+ chamfer"]
MEDIAN = [60.52, 56.61, 49.21, 34.16]
HIT = {"< 25 km": [42.0, 42.3, 43.3, 47.4],
       "< 200 km": [62.7, 62.7, 63.5, 65.9],
       "< 750 km": [80.7, 80.8, 81.3, 83.5]}

# Validated with the dataviz palette checker (adjacent CVD and normal-vision floors, both modes).
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", start="#898781",
                  stages=["#eb6834", "#1baf7a", "#eda100"], final="#2a78d6",
                  ramp=["#0d366b", "#2a78d6", "#86b6ef"], area="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", start="#898781",
                 stages=["#d95926", "#199e70", "#c98500"], final="#3987e5",
                 ramp=["#cde2fb", "#5598e7", "#1c5cab"], area="#3987e5"),
}
W, H = 960, 640
FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'


def loss_curve():
    with open(os.path.join(ROOT, "model", "full_data", "config.json")) as f:
        hist = json.load(f)["history"]
    return [float(re.search(r"loss ([\d.]+)", h).group(1)) for h in hist]


def text(x, y, s, size=12, fill="", weight=400, anchor="start"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>')


def top_rounded_bar(x, y, w, h, r, fill):
    """Bar anchored to the baseline, rounded only at its data end."""
    r = min(r, h, w / 2)
    return (f'<path d="M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} '
            f'H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z" '
            f'fill="{fill}"/>')


def header(x, y, title, sub, t):
    return [text(x, y + 16, title, 15, t["ink"], 600), text(x, y + 35, sub, 12, t["ink2"])]


def waterfall(t, x0, y0, w, h):
    out = header(x0, y0, "Where the error goes", "median km on held-out 20% · lower is better", t)
    left, right, top, bot = x0 + 36, x0 + w - 4, y0 + 70, y0 + h - 38
    ymax = 70
    sy = lambda v: bot - v / ymax * (bot - top)
    for v in (0, 20, 40, 60):
        out.append(f'<line x1="{left}" x2="{right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" '
                   f'stroke="{t["axis"] if v == 0 else t["grid"]}" stroke-width="1"/>')
        out.append(text(left - 8, sy(v) + 4, str(v), 11, t["muted"], anchor="end"))
    labels = STAGES + ["final"]
    band = (right - left) / len(labels)
    bw = band * 0.58
    cx = lambda i: left + band * (i + 0.5)
    # start and end bars sit on the baseline; the three gains float between running totals
    out.append(top_rounded_bar(cx(0) - bw / 2, sy(MEDIAN[0]), bw, sy(0) - sy(MEDIAN[0]), 4,
                               t["start"]))
    out.append(text(cx(0), sy(MEDIAN[0]) - 8, f"{MEDIAN[0]:.1f}", 12, t["ink"], 600, "middle"))
    for i in range(1, 4):
        hi, lo = MEDIAN[i - 1], MEDIAN[i]
        out.append(f'<line x1="{cx(i - 1) + bw / 2:.1f}" x2="{cx(i) - bw / 2:.1f}" '
                   f'y1="{sy(hi):.1f}" y2="{sy(hi):.1f}" stroke="{t["muted"]}" '
                   f'stroke-width="1" stroke-dasharray="3 3"/>')
        out.append(f'<rect x="{cx(i) - bw / 2:.1f}" y="{sy(hi):.1f}" width="{bw:.1f}" '
                   f'height="{sy(lo) - sy(hi):.1f}" rx="4" fill="{t["stages"][i - 1]}"/>')
        out.append(text(cx(i), sy(hi) - 8, f"−{hi - lo:.1f}", 12, t["ink"], 600, "middle"))
    out.append(f'<line x1="{cx(3) + bw / 2:.1f}" x2="{cx(4) - bw / 2:.1f}" y1="{sy(MEDIAN[3]):.1f}" '
               f'y2="{sy(MEDIAN[3]):.1f}" stroke="{t["muted"]}" stroke-dasharray="3 3"/>')
    out.append(top_rounded_bar(cx(4) - bw / 2, sy(MEDIAN[3]), bw, sy(0) - sy(MEDIAN[3]), 4,
                               t["final"]))
    out.append(text(cx(4), sy(MEDIAN[3]) - 8, f"{MEDIAN[3]:.1f} km", 13, t["ink"], 700, "middle"))
    for i, s in enumerate(labels):
        out.append(text(cx(i), bot + 20, s, 11.5, t["ink2"], anchor="middle"))
    return out


def hit_rates(t, x0, y0, w, h):
    out = header(x0, y0, "Hit rate within distance", "% of held-out photos · higher is better", t)
    left, right, top, bot = x0 + 36, x0 + w - 112, y0 + 92, y0 + h - 38
    lo, hi = 35, 90
    sy = lambda v: bot - (v - lo) / (hi - lo) * (bot - top)
    sx = lambda i: left + 14 + i * (right - left - 28) / 3
    # legend row (identity never rests on colour alone: lines are also labelled at their ends)
    lx = x0
    for (name, _), c in zip(HIT.items(), t["ramp"]):
        out.append(f'<rect x="{lx}" y="{y0 + 52}" width="14" height="4" rx="2" fill="{c}"/>')
        out.append(text(lx + 20, y0 + 58, name, 11.5, t["ink2"]))
        lx += 84
    for v in (40, 60, 80):
        out.append(f'<line x1="{left}" x2="{right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" '
                   f'stroke="{t["grid"]}" stroke-width="1"/>')
        out.append(text(left - 8, sy(v) + 4, f"{v}%", 11, t["muted"], anchor="end"))
    for (name, ys), c in zip(HIT.items(), t["ramp"]):
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(ys))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        for i, v in enumerate(ys):
            out.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="4.5" fill="{c}" '
                       f'stroke="{t["surface"]}" stroke-width="2"/>')
        out.append(text(sx(3) + 12, sy(ys[-1]) + 4, f"{ys[-1]:.1f}%", 12, t["ink"], 600))
        out.append(text(sx(3) + 54, sy(ys[-1]) + 4, name, 11, t["ink2"]))
    for i, s in enumerate(STAGES):
        out.append(text(sx(i), bot + 20, s, 11.5, t["ink2"], anchor="middle"))
    return out


def loss_panel(t, x0, y0, w, h, loss):
    out = header(x0, y0, "Training loss of the shipped model",
                 f"KL(Q ‖ P) per epoch · {len(loss)} epochs on all 11,758 images", t)
    left, right, top, bot = x0 + 36, x0 + w - 56, y0 + 62, y0 + h - 34
    lo, hi = int(min(loss) * 2) / 2, int(max(loss) * 2 + 1) / 2
    n = len(loss) - 1
    sx = lambda i: left + i / n * (right - left)
    sy = lambda v: bot - (v - lo) / (hi - lo) * (bot - top)
    v = lo
    while v <= hi + 1e-9:
        out.append(f'<line x1="{left}" x2="{right}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" '
                   f'stroke="{t["axis"] if v == lo else t["grid"]}" stroke-width="1"/>')
        out.append(text(left - 8, sy(v) + 4, f"{v:.1f}", 11, t["muted"], anchor="end"))
        v += 0.5
    pts = [(sx(i), sy(l)) for i, l in enumerate(loss)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    out.append(f'<polygon points="{left:.1f},{bot:.1f} {line} {right:.1f},{bot:.1f}" '
               f'fill="url(#area)"/>')
    peak = round(0.15 * len(loss))  # OneCycle pct_start=0.15
    out.append(f'<line x1="{sx(peak):.1f}" x2="{sx(peak):.1f}" y1="{top:.1f}" y2="{bot:.1f}" '
               f'stroke="{t["muted"]}" stroke-dasharray="3 3"/>')
    out.append(text(sx(peak) + 6, top + 10, "peak learning rate (OneCycle)", 11, t["ink2"]))
    out.append(f'<polyline points="{line}" fill="none" stroke="{t["area"]}" stroke-width="2" '
               f'stroke-linejoin="round"/>')
    for i in (0, n):
        x, y = pts[i]
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{t["area"]}" '
                   f'stroke="{t["surface"]}" stroke-width="2"/>')
    out.append(text(pts[0][0] + 10, pts[0][1] + 4, f"{loss[0]:.2f}", 12, t["ink"], 600))
    out.append(text(pts[n][0] + 10, pts[n][1] + 4, f"{loss[-1]:.2f}", 12, t["ink"], 600))
    for i in (0, 9, 19, 29, n):
        out.append(text(sx(i), bot + 18, str(i + 1), 11, t["muted"], anchor="middle"))
    out.append(text(right + 10, bot + 18, "epoch", 11, t["muted"]))
    return out


def render(mode):
    t = THEMES[mode]
    loss = loss_curve()
    body = (waterfall(t, 28, 24, 436, 340) + hit_rates(t, 500, 24, 436, 340)
            + loss_panel(t, 28, 392, 904, 228, loss))
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family=\'{FONT}\' role="img" aria-labelledby="t">',
        '<title id="t">Ablation: median error falls from 60.5 km to 34.2 km across decode '
        'stages; hit rates rise at every threshold; training loss decreases over 40 epochs.'
        '</title>',
        f'<defs><linearGradient id="area" x1="0" x2="0" y1="0" y2="1">'
        f'<stop offset="0" stop-color="{t["area"]}" stop-opacity="0.30"/>'
        f'<stop offset="1" stop-color="{t["area"]}" stop-opacity="0.02"/></linearGradient></defs>',
        f'<rect width="{W}" height="{H}" rx="12" fill="{t["surface"]}"/>',
        *body, "</svg>"])


if __name__ == "__main__":
    for mode in THEMES:
        path = os.path.join(HERE, f"results-{mode}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render(mode))
        print("wrote", path)
