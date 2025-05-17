

import matplotlib
matplotlib.use("TkAgg")      # or "Qt5Agg"

import os
from dotenv import load_dotenv

import wandb
import matplotlib.pyplot as plt

def main(run_path : str):
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("Please set WANDB_API_KEY in your environment or .env file")
    wandb.login(key=api_key)

    api = wandb.Api()
    run = api.run(run_path)

    layers = [0, 1]
    heads  = list(range(8))

    # 4) pull history for all spans + step
    keys = [f"learner_step/adaptive_span/layer_{L}/head_{H}"
            for L in layers for H in heads]
    df = (run.history(keys=keys + ["_step"], pandas=True)
             .sort_values("_step")
             .reset_index(drop=True))
    if df.empty:
        raise RuntimeError("No data found for those keys. Check your run path and metric names.")

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(6*n_layers, 4), sharey=True)
    if n_layers == 1:
        axes = [axes]

    for ax, L in zip(axes, layers):
        # extract step + all heads for this layer
        plot_df = df[["_step"] + [f"learner_step/adaptive_span/layer_{L}/head_{H}" for H in heads]]
        plot_df = plot_df.set_index("_step")
        plot_df.columns = [f"Head {H+1}" for H in heads]  # make 1-based labels

        # draw a curve per head
        for head in plot_df.columns:
            ax.plot(plot_df.index, plot_df[head], label=head)

        ax.set_yscale("log")
        ax.set_xlabel("Training Step")
        ax.set_title(f"Layer {L+1}")
        ax.legend(fontsize="small", ncol=2, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.grid(True, which="both", ls="--", alpha=0.3)

    axes[0].set_ylabel("Learned Span (log scale)")
    plt.suptitle(f"Adaptive Span Evolution — run {run_path.split('/')[-1]}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
