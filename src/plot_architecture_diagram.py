"""
=============================================================================
THOR-V8 STREAMLINED PUBLICATION BLOCK DIAGRAM (HIGH-DENSITY COMPACT)
Direct left-to-right dataflow with zero nested outer boxes and zero dead space
=============================================================================
"""

import sys
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUT_DIRS = [
    Path(r"C:\Users\levib\Desktop\SEMCITEC26\CONIC\THOR\THOR-PIML\results\figures"),
    Path(r"C:\Users\levib\.gemini\antigravity\worktrees\THOR-PIML\debug_mars_runtime_error\results\figures")
]
for od in OUT_DIRS:
    od.mkdir(parents=True, exist_ok=True)

ARTIFACT_DIR = Path(r"C:\Users\levib\.gemini\antigravity\brain\a3c6cc3a-1fc2-40ea-841b-b73b117484e2")

def draw_publication_diagram(dark_mode=False):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
    })

    if dark_mode:
        bg_col = "#090d16"
        block_bg = "#111827"
        border_col = "#334155"
        txt_title = "#ffffff"
        txt_body = "#cbd5e1"
        txt_dim = "#94a3b8"
        shadow_col = "#000000"
        shadow_a = 0.4
    else:
        bg_col = "#ffffff"
        block_bg = "#f8fafc"
        border_col = "#cbd5e1"
        txt_title = "#0f172a"
        txt_body = "#334155"
        txt_dim = "#64748b"
        shadow_col = "#94a3b8"
        shadow_a = 0.15

    # Accent colors for distinct functional components
    c_syn = "#0284c7"   # Blue: Synoptic 2D
    c_surf = "#0d9488"  # Teal: Surface 1D
    c_lstm = "#d97706"  # Amber: ResLSTM
    c_tcn = "#7c3aed"   # Purple: TCN
    c_fuse = "#2563eb"  # Royal: Attention/Fusion
    c_hurd = "#16a34a"  # Green: Hurdle Heads / Rain
    c_phys = "#dc2626"  # Red: Physics Constraint

    # Canvas: 14.0 x 4.6 inches (Aspect Ratio ~3.04:1)
    fig = plt.figure(figsize=(14.0, 4.6), dpi=300, facecolor=bg_col)
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99], facecolor=bg_col)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 31)
    ax.axis("off")

    # Helper: Dense Layer Block
    def block(x, y, w, h, title, sub="", lines=None, color=c_syn):
        # Drop shadow
        ax.add_patch(FancyBboxPatch((x+0.18, y-0.18), w, h, boxstyle="round,pad=0.1,rounding_size=0.35",
                                    facecolor=shadow_col, edgecolor="none", alpha=shadow_a, zorder=1))
        # Card body
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.35",
                                    facecolor=block_bg, edgecolor=color, linewidth=1.1, zorder=2))
        
        # Header banner
        header_h = 2.6
        ax.add_patch(FancyBboxPatch((x, y + h - header_h), w, header_h, boxstyle="round,pad=0.08,rounding_size=0.3",
                                    facecolor=color, edgecolor="none", zorder=3))
        
        if sub:
            ax.text(x + w/2, y + h - 1.0, title, fontsize=6.2, fontweight="black", color="#ffffff", ha="center", va="center", zorder=4)
            ax.text(x + w/2, y + h - 2.0, sub, fontsize=4.8, color="#e2e8f0", ha="center", va="center", zorder=4)
        else:
            ax.text(x + w/2, y + h - header_h/2, title, fontsize=6.5, fontweight="black", color="#ffffff", ha="center", va="center", zorder=4)

        # Content lines
        if lines:
            n = len(lines)
            avail_h = h - header_h - 0.6
            line_step = avail_h / max(n, 1)
            for i, line_txt in enumerate(lines):
                ly = (y + h - header_h - 0.35) - (i + 0.5) * line_step
                if isinstance(line_txt, tuple):
                    ax.text(x + 0.6, ly, line_txt[0], fontsize=5.1, fontweight="bold", color=txt_body, va="center", zorder=4)
                    ax.text(x + w - 0.6, ly, line_txt[1], fontsize=4.8, fontfamily="monospace", color=txt_dim, ha="right", va="center", zorder=4)
                else:
                    ax.text(x + w/2, ly, line_txt, fontsize=5.1, fontweight="medium", color=txt_body, ha="center", va="center", zorder=4)

    # Helper: Arrow
    def arrow(x1, y1, x2, y2, color=border_col, lw=1.2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7.5,
                                     color=color, linewidth=lw, linestyle=ls, zorder=5))

    # Helper: Operator Circle
    def op(x, y, symbol, color=c_fuse, r=0.9):
        ax.add_patch(plt.Circle((x, y), r, facecolor=block_bg, edgecolor=color, linewidth=1.3, zorder=6))
        ax.text(x, y, symbol, fontsize=7.5, fontweight="black", color=color, ha="center", va="center", zorder=7)

    # =========================================================================
    # COLUMN 1: DUAL INPUT PREDICTORS (x: 0.8 to 15.0, w: 14.2)
    # =========================================================================
    # Top: 2D Synoptic Field
    block(0.8, 16.0, 14.2, 13.5, "2D SYNOPTIC FIELD (ERA5 PL)", "Domain: 6°×8° | (30, 25, 33, 5)", [
        ("z500 (Geopotential)", "Deep Troughs"),
        ("u700 (Zonal Wind)", "Westerly Inflow"),
        ("v700 (Meridional Wind)", "SACZ Jet Advection"),
        ("q700 (Specific Humidity)", "Mid-Tropo Vapor"),
        ("w500 (Vertical Velocity)", "Convective Ascent")
    ], color=c_syn)

    # Bottom: 1D Surface Predictors
    block(0.8, 1.5, 14.2, 13.5, "SURFACE PREDICTORS (ERA5-LAND)", "Local Grid: (30, 84 Features)", [
        ("16 Surface Drivers", "T2m, D2m, TCWV, CAPE"),
        ("80 Multi-Lags", "t-1, t-2, t-3, t-7, t-14"),
        ("4 Regional Contexts", "Domain Grid Mean & Std"),
        ("Zero-Leak Scaler", "Fit on Train Split Only")
    ], color=c_surf)

    # =========================================================================
    # COLUMN 2: 2D SYNOPTIC CNN ENCODER (x: 17.5 to 32.5, w: 15.0)
    # =========================================================================
    arrow(15.0, 22.75, 17.5, 22.75, color=c_syn, lw=1.3)
    block(17.5, 16.0, 15.0, 13.5, "2D SYNOPTIC CNN ENCODER", "Vorticity & Pressure Gradients", [
        ("Conv2D (k=3, 32ch) + GN + SiLU", "32 × 25 × 33"),
        ("Conv2D (k=3, s=2, 64ch) + SiLU", "64 × 13 × 17"),
        ("Conv2D (k=3, s=2, 128ch) + Drop", "128 × 7 × 9"),
        ("AdaptiveAvgPool2D (1×1)", "128 × 1 × 1"),
        ("Linear(128 → 64) + GELU", "Z_syn ∈ ℝ^(30×64)")
    ], color=c_syn)

    # =========================================================================
    # CONCATENATION NODE (x: 34.6, y: 15.5)
    # =========================================================================
    op(34.6, 15.5, "⊕", color=c_fuse, r=0.9)
    ax.text(34.6, 13.8, "Concat", fontsize=5.8, fontweight="black", color=c_fuse, ha="center")
    ax.text(34.6, 12.6, "(30×148)", fontsize=4.8, fontfamily="monospace", fontweight="bold", color=txt_dim, ha="center")

    arrow(32.5, 22.75, 33.7, 16.2, color=c_syn, lw=1.2)
    arrow(15.0, 8.25, 33.7, 14.8, color=c_surf, lw=1.2)

    # =========================================================================
    # COLUMN 3: BI-BRANCH TEMPORAL TRUNK (x: 36.6 to 50.4, w: 13.8)
    # =========================================================================
    # Branch A: Residual LSTM
    block(36.6, 16.0, 13.8, 13.5, "BRANCH A: RESIDUAL LSTM", "Hydrological Memory (14d)", [
        ("2-Layer LSTM", "Hidden = 128"),
        ("Residual & Norm", "LayerNorm(h + Skip)"),
        ("Memory State", "Soil Moisture"),
        ("Latent Output", "h_lstm ∈ ℝ^(30×128)")
    ], color=c_lstm)

    # Branch B: Multi-Scale TCN
    block(36.6, 1.5, 13.8, 13.5, "BRANCH B: MULTI-SCALE TCN", "Frontal Triggers (1-8d)", [
        ("Conv1D (k∈{3,5,7})", "64ch / scale"),
        ("Dilations (d∈{1,2,4,8})", "Causal Masking"),
        ("Receptive Field", "31 Days (Zero-Leak)"),
        ("Frontal State", "h_tcn ∈ ℝ^(30×128)")
    ], color=c_tcn)

    arrow(35.5, 16.0, 36.6, 22.75, color=c_lstm, lw=1.2)
    arrow(35.5, 15.0, 36.6, 8.25, color=c_tcn, lw=1.2)

    # =========================================================================
    # GATED FUSION & ATTENTION (x: 54.5 to 69.4)
    # =========================================================================
    op(54.5, 15.5, "⊗", color=c_fuse, r=0.9)
    ax.text(54.5, 13.8, "Fusion", fontsize=5.8, fontweight="black", color=c_fuse, ha="center")
    ax.text(54.5, 12.6, "(30×128)", fontsize=4.8, fontfamily="monospace", fontweight="bold", color=txt_dim, ha="center")

    arrow(50.4, 22.75, 53.6, 16.2, color=c_lstm, lw=1.2)
    arrow(50.4, 8.25, 53.6, 14.8, color=c_tcn, lw=1.2)

    # SDPA Attention
    arrow(55.4, 15.5, 57.2, 15.5, color=c_fuse, lw=1.3)
    block(57.2, 8.5, 12.2, 14.0, "SDPA ATTENTION", "Causal Temporal Focus", [
        ("8-Head Attention", "Causal Mask"),
        ("Residual Add", "LayerNorm"),
        ("Slice Target t", "h_T = h[:, -1, :]"),
        ("Output Latent", "h_T ∈ ℝ^128")
    ], color=c_fuse)

    # =========================================================================
    # COLUMN 5: HURDLE DUAL-HEAD PREDICTOR (x: 71.2 to 84.0, w: 12.8)
    # =========================================================================
    arrow(69.4, 15.5, 71.2, 22.75, color=c_hurd, lw=1.2)
    arrow(69.4, 15.5, 71.2, 8.25, color=c_hurd, lw=1.2)

    # Occurrence Head
    block(71.2, 16.0, 12.8, 13.5, "OCCURRENCE HEAD", "Dry/Rain Binary Gate", [
        ("Linear(128 → 64)", "SiLU + Norm"),
        ("Linear(64 → 1)", "Sigmoid"),
        ("Rain Probability", "p_occ ∈ [0, 1]"),
        ("WMO Threshold", "≥ 1.0 mm/day")
    ], color=c_hurd)

    # Intensity Head
    block(71.2, 1.5, 12.8, 13.5, "INTENSITY HEAD", "Rain Amount Regressor", [
        ("Linear(128 → 96)", "SiLU + Norm"),
        ("Linear(96 → 1)", "Softplus"),
        ("Expected Amount", "μ_int ≥ 0 mm/d"),
        ("Tail Sensitivity", "Gamma Param")
    ], color=c_hurd)

    # =========================================================================
    # COLUMN 6: OUTPUT & CLAUSIUS-CLAPEYRON BARRIER (x: 85.5 to 99.2)
    # =========================================================================
    op(85.5, 15.5, "×", color=c_hurd, r=0.85)
    ax.text(85.5, 13.8, "Precip", fontsize=5.6, fontweight="black", color=c_hurd, ha="center")
    ax.text(85.5, 12.6, "(ŷ)", fontsize=4.8, fontfamily="monospace", fontweight="bold", color=txt_dim, ha="center")

    arrow(84.0, 22.75, 84.7, 16.2, color=c_hurd, lw=1.1)
    arrow(84.0, 8.25, 84.7, 14.8, color=c_hurd, lw=1.1)

    arrow(86.4, 15.5, 87.2, 22.75, color=c_hurd, lw=1.2)
    arrow(86.4, 15.5, 87.2, 8.25, color=c_phys, lw=1.2, ls="--")

    # Downscaled Precipitation
    block(87.2, 16.0, 12.0, 13.5, "DOWNSCALED PRECIP.", "Target Prediction", [
        ("ŷ = p_occ × μ_int", "mm/day"),
        ("Record KGE", "+0.4101"),
        ("R10mm Detection", "279 / 287 days"),
        ("Zero Drizzle", "Strictly Enforced")
    ], color=c_hurd)

    # Clausius-Clapeyron Barrier
    block(87.2, 1.5, 12.0, 13.5, "CLAUSIUS-CLAPEYRON", "Physics Loss Barrier", [
        ("Cap: P_max", "1.35 × TCWV"),
        ("Loss: L_CC", "λ · Softplus(...)"),
        ("Zero Dead Grads", "Softplus Barrier"),
        ("Physical Violation", "0.00% (Strict)")
    ], color=c_phys)

    # Save to canonical figure names
    tag = "dark" if dark_mode else "light"
    for od in OUT_DIRS:
        png_f = od / f"fig_thor_v8_architecture_{tag}.png"
        pdf_f = od / f"fig_thor_v8_architecture_{tag}.pdf"
        plt.savefig(png_f, dpi=300, bbox_inches="tight", pad_inches=0.01, facecolor=bg_col)
        plt.savefig(pdf_f, bbox_inches="tight", pad_inches=0.01, facecolor=bg_col)
        print(f"✓ Saved {png_f} and {pdf_f}")
    plt.close()

    if ARTIFACT_DIR.exists():
        art_f = ARTIFACT_DIR / f"fig_thor_v8_architecture_{tag}.png"
        shutil.copy(OUT_DIRS[0] / f"fig_thor_v8_architecture_{tag}.png", art_f)

if __name__ == "__main__":
    draw_publication_diagram(dark_mode=False)  # Light paper standard
    draw_publication_diagram(dark_mode=True)   # Dark standard
