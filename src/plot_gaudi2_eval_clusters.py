#!/usr/bin/env python3
"""
Generate a sensitivity plot for WikiText PMPQ quantization results.
All layers shown as one line; markers and dotted reference lines are
color-coded by bit cluster (16 / 8 / 4).
"""

import os
import re

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# Configuration
FOLDER = "/pscratch/sd/a/ananda/SMPQuant/Evaluation/Gaudi2"
BIT_ALLOCATION = [16, 8, 4]
DATASETS = ["WikiText"]
OUTPUT_FOLDER = "/pscratch/sd/a/ananda/SMPQuant/plots"

# Match plot_submodule_sensitivity.py typography/padding style.
TITLE_FS = 30
AXIS_LABEL_FS = 30
XTICK_FS = 25
YTICK_FS = 25
LEGEND_TITLE_FS = 25
LEGEND_TEXT_FS = 25
TITLE_PAD = 30
LABEL_PAD = 15

# Distinct colors for 16 / 8 / 4-bit clusters
BIT_COLORS = {
    16: "#d62728",   # red
    8:  "#9467bd",   # purple
    4:  "#1f77b4",   # blue
}

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_sensitivity_file(filepath):
    """Parse a PMPQ evaluation txt file and extract layer sensitivity values."""
    with open(filepath, 'r') as f:
        content = f.read()

    bit_alloc_match = re.search(r'Bit Allocation:\s*\[([\d,\s]+)\]', content)
    if not bit_alloc_match:
        return None

    bit_allocation = [int(x.strip()) for x in bit_alloc_match.group(1).split(',')]
    if bit_allocation != BIT_ALLOCATION:
        return None

    layer_pattern = r'Layer\s+(\d+):\s+(\d+)-bit\s+\(sensitivity:\s*([-\d.]+)\)'
    matches = re.findall(layer_pattern, content)
    if not matches:
        return None

    layers, sensitivities, bits = [], [], []
    for match in matches:
        layers.append(int(match[0]))
        sensitivities.append(float(match[2]))
        bits.append(int(match[1]))

    return {
        'layers': layers,
        'sensitivities': sensitivities,
        'bits': bits,
        'bit_allocation': bit_allocation,
    }


def find_dataset_file(dataset_name):
    """Find the first matching file for a given dataset with correct bit allocation."""
    files = [f for f in os.listdir(FOLDER) if f.endswith('.txt')]
    for filename in sorted(files):
        if dataset_name in filename and 'PMPQ' in filename:
            filepath = os.path.join(FOLDER, filename)
            data = parse_sensitivity_file(filepath)
            if data is not None:
                return filepath, data
    return None, None


# ---------------------------------------------------------------------------
# Shared axis helpers
# ---------------------------------------------------------------------------

def _set_yaxis(ax, values, padding_frac=0.08, num_ticks=8):
    """Set y-axis limits and ticks spanning the data min→max of *values*."""
    y_min = min(values)
    y_max = max(values)
    y_range = y_max - y_min
    pad = y_range * padding_frac if y_range > 0 else 0.1
    lo, hi = y_min - pad, y_max + pad
    ax.set_ylim(lo, hi)
    ticks = np.linspace(lo, hi, num_ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:.6f}' for t in ticks], fontsize=YTICK_FS)


def _place_legend_inside(ax, handles, title=None):
    """Place legend inside the axes at the best available position."""
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),  # slightly lower than default lower-right anchor
        prop={'weight': 'bold', 'size': LEGEND_TEXT_FS},
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.7",
        title=title,
        title_fontproperties={"weight": "bold", "size": LEGEND_TITLE_FS},
        borderaxespad=0.25,
        borderpad=0.7,
        labelspacing=0.7,
        handlelength=1.8,     # Default is ~2.0; higher = wider symbols/box
        handletextpad=0.7     # Default is ~0.8; higher = more space after the icon
    )


# ---------------------------------------------------------------------------
# Plot 1 — cluster overview (single line, markers & reference lines by cluster)
# ---------------------------------------------------------------------------

def create_cluster_overview_plot(data, dataset_name, output_path):
    """
    One line connecting all layers in order.
    Each marker is colored by its bit-cluster assignment.
    Dotted horizontal reference lines (mean sensitivity per cluster) are drawn
    in the matching cluster color.
    """
    layers        = data['layers']
    sensitivities = data['sensitivities']
    bits          = data['bits']
    bit_allocation = data['bit_allocation']

    n = len(layers)
    x = np.arange(n)

    # Compact width for Overleaf; taller to give y-axis room
    fig, ax = plt.subplots(figsize=(18, 10))

    # --- grey backbone line ---
    ax.plot(x, sensitivities, '-', color='#aaaaaa', linewidth=2.5, zorder=1)

    # --- colored markers per layer ---
    for xi, s, b in zip(x, sensitivities, bits):
        ax.plot(xi, s, 'o', color=BIT_COLORS[b], markersize=15, zorder=2)

    # --- dotted horizontal reference lines (mean per cluster) ---
    unique_bits = sorted(set(bit_allocation), reverse=True)
    ref_handles = []
    for bit_val in unique_bits:
        cluster_s = [s for s, b in zip(sensitivities, bits) if b == bit_val]
        if cluster_s:
            y_mean = np.mean(cluster_s)
            c = BIT_COLORS[bit_val]
            ax.axhline(y=y_mean, color=c, linestyle=':', linewidth=3.5,
                       alpha=0.9, zorder=0)
            ref_handles.append(
                mlines.Line2D([], [], color=c, linestyle=':', linewidth=3.5,
                              marker='o', markersize=15, label=f'{bit_val}-bit cluster')
            )

    # --- axes ---
    # Large fontsizes so text stays legible when scaled down in Overleaf
    ax.set_xlabel('TinyLlama-1.1B Layers', fontsize=AXIS_LABEL_FS, fontweight='bold', labelpad=LABEL_PAD)
    ax.set_xticks(x)
    ax.set_xticklabels([f'layer_{l}' for l in layers], rotation=90, ha='center', fontsize=XTICK_FS)
    ax.set_ylabel('Sensitivity Scores', fontsize=AXIS_LABEL_FS, fontweight='bold', labelpad=LABEL_PAD)
    _set_yaxis(ax, sensitivities)

    ax.grid(True, alpha=0.5, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    ax.set_title(
        f'Magnitude-Pruning based Sensitivity Clustering per Linear Layer',
        fontsize=TITLE_FS, fontweight='bold', pad=TITLE_PAD
    )

    _place_legend_inside(ax, ref_handles)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PMPQ Sensitivity Plotting Tool — WikiText cluster overview")
    print("=" * 70)
    print(f"Target Bit Allocation: {BIT_ALLOCATION}")
    print(f"Output Folder: {OUTPUT_FOLDER}")
    print()

    for dataset in DATASETS:
        print(f"Processing {dataset}...")
        filepath, data = find_dataset_file(dataset)

        if data is None:
            print(f"  No matching file found for {dataset} with bit allocation {BIT_ALLOCATION}")
            continue

        print(f"  Found: {os.path.basename(filepath)}")
        s = data['sensitivities']
        print(f"  Layers: {len(data['layers'])}")
        print(f"  Sensitivity range: [{min(s):.6f}, {max(s):.6f}]")
        print()

        out = os.path.join(OUTPUT_FOLDER, f"PMPQ_TinyLlama_{dataset}_sensitivity.png")
        create_cluster_overview_plot(data, dataset, out)
        print()

    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
