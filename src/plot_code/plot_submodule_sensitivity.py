                      
"""
Plot submodule-level sensitivity scores for a mixed-precision FP8/BF16
quantization run on Gaudi 2 (Intel Neural Compressor).

Reads:
  - submodule sensitivity JSON  (pruning-based sensitivity per linear layer)
  - run_summary.json            (FP8/BF16 assignment + perplexity results)

Produces two plots:
  1. mlp_only  — only the MLP submodules targeted by the run
  2. all       — every submodule present in the sensitivity file
Both plots colour markers and reference lines by precision (FP8 / BF16).
"""

import json
import os
import re

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

                                                                             
       
                                                                             
SUBMODULE_DIR = "/pscratch/sd/a/ananda/SMPQuant/submodule"
OUTPUT_FOLDER = "/pscratch/sd/a/ananda/SMPQuant/plots"

                                                                   
DATASETS = [
    (
        "WikiText",
        os.path.join(SUBMODULE_DIR, "sensitivity_wikitext.json"),
        os.path.join(SUBMODULE_DIR, "run_wikitext.json"),
        "submodule_sensitivity_mlp_fp8_bf16_wikitext.png",
    ),
    (
        "HellaSwag",
        os.path.join(SUBMODULE_DIR, "sensitivity_hellaswag.json"),
        os.path.join(SUBMODULE_DIR, "run_hellaswag.json"),
        "submodule_sensitivity_mlp_fp8_bf16_hellaswag.png",
    ),
]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

                                                          
PREC_COLORS = {
    "FP8":  "#d62728",        
    "BF16": "#1f77b4",         
}

                                                                     
PROJ_ORDER = {
    "down_proj": 0, "gate_proj": 1, "up_proj": 2,        
    "k_proj": 3,    "o_proj": 4,    "q_proj": 5, "v_proj": 6,             
}


                                                                             
              
                                                                             

def load_data(sens_json, run_json):
    """Load sensitivity and run-summary JSON files."""
    with open(sens_json) as f:
        sens_data = json.load(f)
    with open(run_json) as f:
        summary = json.load(f)

    sensitivities = sens_data["sensitivities"]                         
    fp8_set       = set(summary["fp8_modules"])                             
    target_family = summary.get("target_family", "all")

                              
                                                              
    pattern = re.compile(
        r"model\.layers\.(\d+)\.(mlp|self_attn)\.(\w+)"
    )
    records = []
    for name, sens in sensitivities.items():
        m = pattern.fullmatch(name)
        if not m:
            continue
        layer_num  = int(m.group(1))
        family     = m.group(2)                            
        proj       = m.group(3)                                      
        precision  = "FP8" if name in fp8_set else "BF16"
        records.append({
            "name":      name,
            "layer":     layer_num,
            "family":    family,
            "proj":      proj,
            "sens":      sens,
            "precision": precision,
        })

                                                     
    def sort_key(r):
        fam_order = 0 if r["family"] == "mlp" else 1
        return (r["layer"], fam_order, PROJ_ORDER.get(r["proj"], 99))

    records.sort(key=sort_key)

    return records, summary, target_family


                                                                             
                                                           
                                                                             

                                               
TITLE_FS        = 30
AXIS_LABEL_FS   = 30
XTICK_FS        = 25
YTICK_FS        = 25
LEGEND_TITLE_FS = 25
LEGEND_TEXT_FS  = 25
TITLE_PAD       = 30
LABEL_PAD       = 15
MARKER_SIZE     = 15


def _set_yaxis(ax, values, padding_frac=0.08, num_ticks=8):
    y_min, y_max = min(values), max(values)
    y_range = y_max - y_min
    pad = y_range * padding_frac if y_range > 0 else 0.01
    lo, hi = y_min - pad, y_max + pad
    ax.set_ylim(lo, hi)
    ticks = np.linspace(lo, hi, num_ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.5f}" for t in ticks], fontsize=YTICK_FS)


                                                                             
                    
                                                                             

def create_submodule_plot(records, summary, title_suffix, output_path, dataset_label=""):
    """
    One grey backbone line + per-submodule coloured markers.
    Dotted horizontal reference lines at the mean sensitivity of each
    precision cluster (FP8, BF16).
    Subtitle row shows key quantization stats.
    """
    names      = [r["name"]      for r in records]
    sens_vals  = [r["sens"]      for r in records]
    precisions = [r["precision"] for r in records]

    n = len(records)
    x = np.arange(n)

                                                                     
    def short_label(r):
        if r["family"] == "mlp":
            proj = r["proj"].replace("_proj", "")                     
            return f"L{r['layer']}.{proj}"
        else:
            proj = r["proj"].replace("_proj", "")
            return f"L{r['layer']}.a.{proj}"

    xlabels = [short_label(r) for r in records]

                                                                                             
                                   
                                                    
    fig, ax = plt.subplots(figsize=(18, 10))

                           
    ax.plot(x, sens_vals, "-", color="#aaaaaa", linewidth=2.5, zorder=1)

                                                                      
                                                                 
    ref_handles = []
    for prec in ("FP8", "BF16"):
        cluster_s = [s for s, p in zip(sens_vals, precisions) if p == prec]
        if cluster_s:
            y_mean = np.mean(cluster_s)
            c = PREC_COLORS[prec]
            ax.axhline(y=y_mean, color=c, linestyle=":", linewidth=3.5,
                       alpha=0.9, zorder=0)
                                                                                   
            ref_handles.append(
                mlines.Line2D(
                    [], [], color=c, linestyle=":", linewidth=3.5,
                    label=f"{prec}",
                )
            )

                  
                                                                           
                                                          
    from itertools import groupby as _groupby

    layer_groups = {}
    for xi, r in zip(x, records):
        layer_groups.setdefault(r["layer"], []).append(xi)

                                            
    layer_tick_pos = [int(np.mean(idxs)) for idxs in layer_groups.values()]
    layer_tick_lbl = [f"layer_{lyr}" for lyr in layer_groups]

    ax.set_xticks(layer_tick_pos)
    ax.set_xticklabels(layer_tick_lbl, rotation=90, ha="center", fontsize=XTICK_FS, fontweight="normal")

                                                                      
    ax.set_xticks(x, minor=True)
    ax.tick_params(axis="x", which="minor", length=0)

                                                                                      
    all_layer_indices = list(layer_groups.values())
    for grp in all_layer_indices[:-1]:
        boundary = grp[-1] + 0.5
        ax.axvline(boundary, color="#cccccc", linewidth=1.0, linestyle="-", zorder=0)

                                                     
    proj_shape_map = {"down_proj": "^", "gate_proj": "s", "up_proj": "D"}
    for xi, r in zip(x, records):
        mk = proj_shape_map.get(r["proj"], "o")
        ax.plot(xi, r["sens"], mk, color=PREC_COLORS[r["precision"]],
                markersize=MARKER_SIZE, zorder=3)

                                                                        
    proj_markers = {"down": "^", "gate": "s", "up": "D"}
    proj_handles = [
        mlines.Line2D([], [], color="#555555", marker=m, markersize=MARKER_SIZE,
                      linestyle="None", label=p)
        for p, m in proj_markers.items()
    ]

    ax.set_xlabel("TinyLlama-1.1B Submodule Layers (MLP Only)",
                  fontsize=AXIS_LABEL_FS, fontweight="bold", labelpad=LABEL_PAD)

    ax.set_ylabel("Sensitivity Score",
                  fontsize=AXIS_LABEL_FS, fontweight="bold", labelpad=LABEL_PAD)
    _set_yaxis(ax, sens_vals)

    ax.grid(True, alpha=0.4, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

                              
    model_short = summary.get("model_name_or_path", "").split("/")[-1]
    ppl_base  = summary.get("ppl_baseline", float("nan"))
    ppl_mixed = summary.get("ppl_mixed",    float("nan"))
    ppl_delta = summary.get("ppl_delta_pct", float("nan"))
    fp8_fmt   = summary.get("fp8_format",   "FP8")
    n_fp8     = len(summary.get("fp8_layers",  []))
    n_bf16    = len(summary.get("bf16_layers", []))

    title = "Magnitude-Pruning based Sensitivity Clustering - MLP Submodules"
                       
                                        
    ax.set_title(title, fontsize=TITLE_FS, fontweight='bold', pad=TITLE_PAD)

    _legend_kw = dict(
        prop={"weight": "bold", "size": LEGEND_TEXT_FS},
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.7",
        borderaxespad=0.25,
        borderpad=0.7,
        labelspacing=0.7,
        handlelength=1.8,
        handletextpad=0.7,
    )

                                                                             
    leg_prec = ax.legend(
        handles=ref_handles,
        loc="upper right",
        bbox_to_anchor=(0.9, 0.99),
        title="Precision cluster",
        title_fontproperties={"weight": "bold", "size": LEGEND_TITLE_FS},
        labelcolor=[PREC_COLORS[p] for p in ("FP8", "BF16")],
        **_legend_kw,
    )
    ax.add_artist(leg_prec)

                                                                                     
    ax.legend(
        handles=proj_handles,
        loc="upper right",
        bbox_to_anchor=(0.63, 0.99),
        title="Projection",
        title_fontproperties={"weight": "bold", "size": LEGEND_TITLE_FS},
        ncol=3,
        columnspacing=1.0,
        **_legend_kw,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close(fig)


                                                                             
      
                                                                             

def main():
    print("=" * 70)
    print("Submodule Sensitivity Plot — FP8/BF16 Mixed Precision (Gaudi 2)")
    print("=" * 70)

    for dataset_label, sens_json, run_json, out_filename in DATASETS:
        print(f"\n── {dataset_label} ──")
        records, summary, target_family = load_data(sens_json, run_json)
        print(f"  Total submodules loaded : {len(records)}")
        print(f"  Target family           : {target_family}")

        fp8_count  = sum(1 for r in records if r["precision"] == "FP8")
        bf16_count = sum(1 for r in records if r["precision"] == "BF16")
        print(f"  FP8 submodules          : {fp8_count}")
        print(f"  BF16 submodules         : {bf16_count}")

        mlp_records = [r for r in records if r["family"] == "mlp"]
        print(f"  Generating MLP-only plot ({len(mlp_records)} submodules)...")
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)
        create_submodule_plot(mlp_records, summary,
                              title_suffix="MLP submodules",
                              output_path=out_path,
                              dataset_label=dataset_label)

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
