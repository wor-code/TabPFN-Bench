import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import t as t_dist
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import MultiComparison


def mean_ci95(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan

    mean = np.mean(values)

    if n == 1:
        return mean, mean, mean

    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)
    tcrit = t_dist.ppf(0.975, df=n - 1)

    return mean, mean - tcrit * se, mean + tcrit * se


def build_nonsig_matrix(models, tukey_df):
    """
    Construct non-significance matrix from Tukey HSD results.
    """
    nonsig = pd.DataFrame(False, index=models, columns=models)

    for m in models:
        nonsig.loc[m, m] = True

    for _, row in tukey_df.iterrows():
        a = row["group1"]
        b = row["group2"]
        reject = bool(row["reject"])

        nonsig.loc[a, b] = not reject
        nonsig.loc[b, a] = not reject

    return nonsig


def build_cld_letters(models_sorted, nonsig_matrix):
    """
    Construct Compact Letter Display (CLD).
    Models sharing a letter are not significantly different.
    """
    letters = "abcdefghijklmnopqrstuvwxyz"
    assigned = {m: "" for m in models_sorted}
    letter_sets = []

    for m in models_sorted:
        placed = False
        for s in letter_sets:
            if all(nonsig_matrix.loc[m, other] for other in s):
                s.add(m)
                placed = True
                break

        if not placed:
            letter_sets.append(set([m]))

    for i, s in enumerate(letter_sets):
        for m in s:
            assigned[m] += letters[i]

    return assigned


def rm_anova_and_tukey(long_df, alpha=0.05):
    """
    Repeated-measures ANOVA + Tukey HSD.
    """

    df = long_df.copy()
    df["subject"] = df["dataset"] + "__" + df["repeat"].astype(str)

    # RM-ANOVA
    aov = AnovaRM(df, depvar="value", subject="subject", within=["model"]).fit()
    p_anova = float(aov.anova_table["Pr > F"].iloc[0])

    # Subject-centering (for paired structure)
    df["centered"] = df["value"] - df.groupby("subject")["value"].transform("mean")

    mc = MultiComparison(df["centered"], df["model"])
    tuk = mc.tukeyhsd(alpha=alpha)

    tuk_df = pd.DataFrame(
        tuk._results_table.data[1:],
        columns=tuk._results_table.data[0],
    )

    return p_anova, tuk_df


def plot_aggregated(long_df, metric="MCC", alpha=0.05):
    """
    Aggregated comparison with:
    - Mean
    - 95% CI
    - RM-ANOVA
    - Tukey HSD
    - CLD-based coloring
    """

    models = sorted(long_df["model"].unique())

    p_anova, tuk_df = rm_anova_and_tukey(long_df, alpha=alpha)

    nonsig = build_nonsig_matrix(models, tuk_df)

    # Compute mean + CI
    stats = []
    for m in models:
        vals = long_df[long_df["model"] == m]["value"].values
        mean, lo, hi = mean_ci95(vals)
        stats.append((m, mean, lo, hi))

    stat_df = pd.DataFrame(
        stats,
        columns=["Model", "mean", "ci_lo", "ci_hi"]
    )

    # Sort by performance (higher better)
    stat_df = stat_df.sort_values("mean", ascending=False).reset_index(drop=True)

    # CLD
    cld_map = build_cld_letters(
        stat_df["Model"].tolist(),
        nonsig
    )
    stat_df["cld"] = stat_df["Model"].map(cld_map)

    # Best model
    best_model = stat_df.iloc[0]["Model"]
    best_letters = set(stat_df.iloc[0]["cld"])

    # Colors
    BEST_COLOR = "#1f77b4"
    WORSE_COLOR = "#d62728"
    NEUTRAL_COLOR = "#7f7f7f"

    fig, ax = plt.subplots(figsize=(6, 4))

    xmin = stat_df["ci_lo"].min()
    xmax = stat_df["ci_hi"].max()
    pad = 0.05 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)

    for _, row in stat_df.iterrows():

        model = row["Model"]
        mean = row["mean"]
        lo = row["ci_lo"]
        hi = row["ci_hi"]
        letters = set(row["cld"])

        if model == best_model:
            color = BEST_COLOR
        elif letters.isdisjoint(best_letters):
            color = WORSE_COLOR
        else:
            color = NEUTRAL_COLOR

        ax.errorbar(
            mean,
            model,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color=color,
            elinewidth=2,
            capsize=4,
        )

        ax.text(
            xmin,
            model,
            row["cld"],
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_title(f"{metric} (RM-ANOVA p={p_anova:.4f})")
    ax.set_xlabel(metric)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Expected results columns:
    # dataset | repeat | model | value
    long_df = pd.read_csv("results.csv")

    plot_aggregated(
        long_df,
        metric="MCC",
        alpha=0.05
    )
