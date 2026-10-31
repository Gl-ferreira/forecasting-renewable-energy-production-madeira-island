"""
make_validation_chart.py

Sanity check: does the synthetic production data track the real monthly
pattern for each energy source? Produces synthetic_vs_real_seasonal.png.

The real comparison line only shows up if you have access to the original,
confidential Production.pkl (the author does, nobody else should). Without
it, this still runs and plots the synthetic side on its own, so the script
isn't dead weight for anyone cloning the repo.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TARGETS = ["Hydropower", "Solar", "Wind", "Biomass"]


def monthly_means(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Month")[TARGETS].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-pkl", default=None, help="Path to the real Production.pkl (author-only, not in this repo)")
    parser.add_argument("--out", default="synthetic_vs_real_seasonal.png")
    args = parser.parse_args()

    synth = pd.read_csv(Path(__file__).parent / "data" / "synthetic_production.csv")
    synth_monthly = monthly_means(synth)

    real_monthly = None
    if args.real_pkl and Path(args.real_pkl).exists():
        real = pd.read_pickle(args.real_pkl)
        real_monthly = monthly_means(real)

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(18, 4.2))
    for ax, target in zip(axes, TARGETS):
        if real_monthly is not None:
            ax.plot(real_monthly.index, real_monthly[target], marker="o", label="real", color="#2a5783")
        ax.plot(synth_monthly.index, synth_monthly[target], marker="s", linestyle="--", label="synthetic", color="#d9713c")
        ax.set_title(target)
        ax.set_xlabel("Month")
        ax.set_ylim(bottom=0)
        ax.legend()
    fig.suptitle("Synthetic vs. real monthly means, all 4 renewable sources")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}" + ("" if real_monthly is not None else " (synthetic only, no real-data comparison available)"))


if __name__ == "__main__":
    main()
