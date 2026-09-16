"""
The single publication style for every figure in the paper.

Palette
-------
Okabe-Ito
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Page geometry (verified against neurips_2026.sty)
TEXTWIDTH_IN = 5.5
TEXTHEIGHT_IN = 9.0
COL_FULL = 5.5
COL_HALF = 2.65


# Palette
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermil": "#D55E00",
    "purple": "#CC79A7",
}

# Structural (non-data) colours
NEUTRAL = {
    "reference": "#7F7F7F",
    "annot": "#444444",
}

PALETTE = set(OKABE_ITO.values()) | set(NEUTRAL.values())


SERIES = {
    "MDN": dict(
        label="MDN",
        color=OKABE_ITO["blue"],
        marker="o",
        linestyle="-",
        linewidth=1.0,
        markersize=2.6,
        alpha=1.0,
        zorder=5,
    ),
    "CFM": dict(
        label="CFM",
        color=OKABE_ITO["vermil"],
        marker="s",
        linestyle="-",
        linewidth=1.0,
        markersize=2.4,
        alpha=1.0,
        zorder=5,
    ),
    "KNN": dict(
        label=r"$k$-NN ceiling",
        color=OKABE_ITO["green"],
        marker="^",
        linestyle=(0, (4, 1.6)),
        linewidth=1.0,
        markersize=2.8,
        alpha=1.0,
        zorder=4,
    ),
    # the two subordinate controls: thinner, dashed, lower alpha
    "MARGINAL": dict(
        label="Marginal sampler",
        color=NEUTRAL["reference"],
        marker="None",
        linestyle=(0, (5, 2)),
        linewidth=0.7,
        markersize=0,
        alpha=0.85,
        zorder=2,
    ),
    "LOOKUP": dict(
        label="Lookup table",
        color=OKABE_ITO["purple"],
        marker="None",
        linestyle=(0, (1, 1.4)),
        linewidth=0.7,
        markersize=0,
        alpha=0.85,
        zorder=2,
    ),
}

# The unity reference line. Grey, dashed, thin, never in a legend.
REFERENCE = dict(
    color=NEUTRAL["reference"],
    linestyle=(0, (3, 2.5)),
    linewidth=0.5,
    alpha=0.9,
    zorder=1,
)

METRICS = {
    "val_nll": dict(
        label="validation NLL",
        color=OKABE_ITO["blue"],
        marker="None",
        linestyle="-",
        linewidth=0.9,
        alpha=1.0,
    ),
    "val_crps": dict(
        label="validation CRPS",
        color=OKABE_ITO["orange"],
        marker="None",
        linestyle="-",
        linewidth=0.9,
        alpha=1.0,
    ),
}

STAGES = {
    "parton2reco": dict(
        label=r"parton $\to$ reco",
        color=OKABE_ITO["black"],
        marker="o",
        linestyle="-",
        linewidth=0.9,
        markersize=2.4,
        alpha=1.0,
    ),
    "parton2genjet": dict(
        label=r"parton $\to$ genjet",
        color=OKABE_ITO["orange"],
        marker="s",
        linestyle=(0, (4, 1.6)),
        linewidth=0.9,
        markersize=2.2,
        alpha=1.0,
    ),
    "genjet2reco": dict(
        label=r"genjet $\to$ reco",
        color=OKABE_ITO["skyblue"],
        marker="^",
        linestyle=(0, (1.5, 1.5)),
        linewidth=0.9,
        markersize=2.6,
        alpha=1.0,
    ),
}

TARGETS = {
    "log_pT_resp": dict(color=OKABE_ITO["blue"], marker="o", linestyle="-"),
    "delta_eta": dict(color=OKABE_ITO["orange"], marker="s", linestyle=(0, (4, 1.6))),
    "delta_phi": dict(color=OKABE_ITO["green"], marker="^", linestyle=(0, (1.5, 1.5))),
    "log_mass_frac": dict(
        color=OKABE_ITO["purple"], marker="D", linestyle=(0, (5, 1.5, 1, 1.5))
    ),
}

TARGET_ORDER = ["log_pT_resp", "delta_eta", "delta_phi", "log_mass_frac"]

# Short column headers
TARGET_HEADERS = {
    "log_pT_resp": r"$\log\,p_{\mathrm{T}}$ response",
    "delta_eta": r"$\Delta\eta$",
    "delta_phi": r"$\Delta\phi$",
    "log_mass_frac": r"$\log\,m/p_{\mathrm{T}}$",
}

# Plain-text variants, for the one figure that needs them in a legend.
TARGET_HEADERS_TT = {
    "log_pT_resp": r"$\log p_{\mathrm{T}}$ resp.",
    "delta_eta": r"$\Delta\eta$",
    "delta_phi": r"$\Delta\phi$",
    "log_mass_frac": r"$\log m/p_{\mathrm{T}}$",
}


# rcParams
_RC_BEFORE: dict | None = None


def use_paper_style() -> None:
    """
    Install the paper rcParams. Every paper figure calls this first.
    """
    global _RC_BEFORE
    if _RC_BEFORE is None:
        _RC_BEFORE = dict(mpl.rcParams)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.minor.size": 1.2,
            "ytick.minor.size": 1.2,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "savefig.format": "pdf",
            "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


# Helpers
def style(entity: str, registry: dict = SERIES, **override) -> dict:
    """Keyword arguments for ``ax.plot`` for one registry entity."""
    if entity not in registry:
        raise KeyError(
            f"{entity!r} is not in the registry "
            f"({sorted(registry)}). Add it to common/paper_style.py rather "
            "than passing a colour locally."
        )
    kw = dict(registry[entity])
    kw.update(override)
    return kw


def plot_series(ax, x, y, entity, registry: dict = SERIES, **override):
    """``ax.plot`` with the registry's colour / marker / linestyle."""
    return ax.plot(x, y, **style(entity, registry, **override))


def faint(entity: str, registry: dict = SERIES, **override) -> dict:
    """One data split drawn behind its bold mean: same colour and marker, fainter, no legend."""
    kw = style(entity, registry)
    kw.update(alpha=0.3, linewidth=0.5, label=None)
    if "markersize" in kw:
        kw["markersize"] = kw["markersize"] * 0.7
    if "zorder" in kw:
        kw["zorder"] = kw["zorder"] - 1
    kw.update(override)
    return kw


def plot_faint(ax, x, y, entity, registry: dict = SERIES, **override):
    """``ax.plot`` in the faint per-split style."""
    return ax.plot(x, y, **faint(entity, registry, **override))


def reference_line(ax, value: float = 1.0, axis: str = "y"):
    """The grey dashed reference at ``value``. Never gets a legend entry."""
    fn = ax.axhline if axis == "y" else ax.axvline
    return fn(value, **REFERENCE)


def legend_handles(entities, registry: dict = SERIES):
    """Proxy artists for a figure-level legend, in the given order."""
    out = []
    for e in entities:
        s = registry[e]
        out.append(
            Line2D(
                [],
                [],
                color=s["color"],
                marker=s.get("marker", "None"),
                linestyle=s.get("linestyle", "-"),
                linewidth=s.get("linewidth", 1.0),
                markersize=s.get("markersize", 3),
                alpha=s.get("alpha", 1.0),
                label=s["label"],
            )
        )
    return out


def column_header(ax, target: str) -> None:
    """Short target name above a column. A name, not a sentence."""
    ax.set_title(TARGET_HEADERS.get(target, target), pad=2.5)


# The palette guard
def _norm(c) -> str | None:
    try:
        return mpl.colors.to_hex(c, keep_alpha=False).upper()
    except Exception:
        return None


_ALLOWED = {c.upper() for c in PALETTE} | {"#FFFFFF", "#000000"}


def check_palette(fig, extra: set | None = None) -> None:
    """
    Fail loudly if a figure draws data in a colour outside the registry.

    Raises ``ValueError`` listing the offending colours, so a figure that reaches for a local ``color="#1f77b4"`` cannot land silently.
    """
    allowed = set(_ALLOWED) | {c.upper() for c in (extra or set())}
    bad = {}

    def note(c, where):
        h = _norm(c)
        if h is not None and h not in allowed:
            bad.setdefault(h, set()).add(where)

    for ax in fig.get_axes():
        for ln in ax.get_lines():
            note(ln.get_color(), "Line2D")
            mfc, mec = ln.get_markerfacecolor(), ln.get_markeredgecolor()
            if ln.get_marker() not in ("None", None, ""):
                note(mfc, "marker face")
                note(mec, "marker edge")
        for coll in ax.collections:
            for arr, w in (
                (coll.get_facecolor(), "collection face"),
                (coll.get_edgecolor(), "collection edge"),
            ):
                for c in (arr if getattr(arr, "ndim", 1) > 1 else [arr]):
                    if getattr(c, "__len__", lambda: 0)() and float(c[-1]) == 0.0:
                        continue  # fully transparent
                    note(c, w)
        for p in ax.patches:
            note(p.get_facecolor(), "patch face")
        for t in ax.texts:
            note(t.get_color(), "text")
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            note(t.get_color(), "label")

    if bad:
        raise ValueError(
            "off-registry colours in this figure: "
            + "; ".join(f"{k} ({', '.join(sorted(v))})" for k, v in sorted(bad.items()))
            + ". Every data colour must come from common/paper_style.py."
        )


# Save
def save(fig, path, verbose: bool = True) -> tuple[float, float]:
    """
    Check the palette, write the PDF, and report the *rendered* size.
    """
    global _RC_BEFORE
    check_palette(fig)
    fig.savefig(path)
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = bb.width + 0.02, bb.height + 0.02  # + savefig.pad_inches, both sides
    if verbose:
        print(
            f"[paper_style] {path}  rendered {w:.3f} x {h:.3f} in "
            f"({100*h/TEXTHEIGHT_IN:.1f}% of the 9in text height)"
        )
    plt.close(fig)
    if _RC_BEFORE is not None:
        mpl.rcParams.update(_RC_BEFORE)
        _RC_BEFORE = None
    return w, h
