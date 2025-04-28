from typings import FactorMatrix
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


def patternify_Ls(Lss: list[list[FactorMatrix]]) -> list[list[FactorMatrix]]:
    L_estim_patterns = []
    for Ls in Lss:
        to_add = []
        for L in Ls:
            pattern = np.abs(L) > 1e-16
            np.fill_diagonal(pattern, 0)
            to_add.append(pattern)
        L_estim_patterns.append(to_add)
    return L_estim_patterns

@dataclass
class Metrics:
    TPs: np.ndarray
    FPs: np.ndarray
    TNs: np.ndarray
    FNs: np.ndarray

    shared_TPs: np.ndarray
    shared_FPs: np.ndarray
    shared_TNs: np.ndarray
    shared_FNs: np.ndarray

    precs: list[np.ndarray]
    recs: list[np.ndarray]
    f1s: list[np.ndarray]
    mccs: list[np.ndarray]

    shared_precs: np.ndarray
    shared_recs: np.ndarray
    shared_f1s: np.ndarray
    shared_mccs: np.ndarray

def get_metrics(
    L_patterns: list[list[FactorMatrix]],
    L_estim_patterns: list[list[FactorMatrix]]
) -> Metrics:
    dims = [L[0].shape[0] for L in L_patterns]
    TPs = np.array([
        [
            np.tril(L_pattern & L_estim_pattern, k=-1).sum()
            for L_pattern, L_estim_pattern in zip(L_patterns, version)
        ]
        for version in L_estim_patterns
    ]).T
    FPs = np.array([
        [
            np.tril(~L_pattern & L_estim_pattern, k=-1).sum()
            for L_pattern, L_estim_pattern in zip(L_patterns, version)
        ]
        for version in L_estim_patterns
    ]).T
    TNs = np.array([
        [
            np.tril(~L_pattern & ~L_estim_pattern, k=-1).sum()
            for L_pattern, L_estim_pattern in zip(L_patterns, version)
        ]
        for version in L_estim_patterns
    ]).T
    FNs = np.array([
        [
            np.tril(L_pattern & ~L_estim_pattern, k=-1).sum()
            for L_pattern, L_estim_pattern in zip(L_patterns, version)
        ]
        for version in L_estim_patterns
    ]).T

    shared_TPs = TPs.sum(axis=0)
    shared_FPs = FPs.sum(axis=0)
    shared_TNs = TNs.sum(axis=0)
    shared_FNs = FNs.sum(axis=0)

    precs = [TPs[i] / (TPs[i] + FPs[i]) for i in range(len(dims))]
    shared_precs = shared_TPs / (shared_TPs + shared_FPs)
    recs = [TPs[i] / (TPs[i] + FNs[i]) for i in range(len(dims))]
    shared_recs = shared_TPs / (shared_TPs + shared_FNs)

    f1s = [2*precs[i]*recs[i]/(precs[i]+recs[i]) for i in range(len(dims))]
    mccs = [
        (TPs[i]*TNs[i] - FPs[i]*FNs[i])
        / np.sqrt((TPs[i]+FPs[i])*(TPs[i]+FNs[i])*(TNs[i]+FPs[i])*(TNs[i]+FNs[i]))
        for i in range(len(dims))
    ]
    for i in range(len(f1s)):
        f1s[i][np.isnan(f1s[i])] = 0
        mccs[i][np.isnan(mccs[i])] = 0

    shared_f1s =  2 * (shared_precs * shared_recs)/(shared_precs + shared_recs)
    shared_f1s[np.isnan(shared_f1s)] = 0
    shared_mccs = (
        (shared_TPs * shared_TNs - shared_FPs * shared_FNs)
        / np.sqrt(
            (shared_TPs + shared_FPs)
            * (shared_TPs + shared_FNs)
            * (shared_TNs + shared_FPs)
            * (shared_TNs + shared_FNs)
        )
    )
    shared_mccs[np.isnan(shared_mccs)] = 0

    return Metrics(
        TPs=TPs,
        FPs=FPs,
        TNs=TNs,
        FNs=FNs,
        shared_TPs=shared_TPs,
        shared_FPs=shared_FPs,
        shared_TNs=shared_TNs,
        shared_FNs=shared_FNs,
        precs=precs,
        recs=recs,
        f1s=f1s,
        mccs=mccs,
        shared_precs=shared_precs,
        shared_recs=shared_recs,
        shared_f1s=shared_f1s,
        shared_mccs=shared_mccs
    )

def one_example_prs(*, L_patterns, Lss, Lss_lgam, glassoregs, sparsity, source_distr=""):
    dims = [L[0].shape[0] for L in L_patterns]

    L_estim_patterns = patternify_Ls(Lss)
    L_lgam_estim_patterns = patternify_Ls(Lss_lgam)

    metrics = get_metrics(L_patterns, L_estim_patterns)
    metrics_lgam = get_metrics(L_patterns, L_lgam_estim_patterns)

    Ls_maxF1 = {
        "Cartesian LGAM": [np.argmax(f1) for f1 in metrics.f1s],
        "LGAM": [np.argmax(f1) for f1 in metrics_lgam.f1s]
    }
    sh_maxF1 = {
        "Cartesian LGAM": np.argmax(metrics.shared_f1s),
        "LGAM": np.argmax(metrics_lgam.shared_f1s)
    }
    Ls_maxMCC = {
        "Cartesian LGAM": [np.argmax(mcc) for mcc in metrics.mccs],
        "LGAM": [np.argmax(mcc) for mcc in metrics_lgam.mccs]
    }
    sh_maxMCC = {
        "Cartesian LGAM": np.argmax(metrics.shared_mccs),
        "LGAM": np.argmax(metrics_lgam.shared_mccs)
    }

    fig, (pr_axes, f1mcc_axes) = plt.subplots(ncols=len(dims)+1, nrows=2, figsize=(4+4*len(dims), 8))

    for i, ax in enumerate(pr_axes):
        for model, metr in zip(["Cartesian LGAM", "LGAM"], [metrics, metrics_lgam]):
            if i < len(pr_axes) - 1:
                precision = metr.precs[i]
                recall = metr.recs[i]
                best_greg_f1_idx = Ls_maxF1[model][i]
                best_greg_mcc_idx = Ls_maxMCC[model][i]
                best_greg_f1 = glassoregs[best_greg_f1_idx]
                best_greg_mcc = glassoregs[best_greg_mcc_idx]
                ax.set_title(f"L{i+1} PR Curve")
            else:
                precision = metr.shared_precs
                recall = metr.shared_recs
                best_greg_f1_idx = sh_maxF1[model]
                best_greg_mcc_idx = sh_maxMCC[model]
                best_greg_f1 = glassoregs[best_greg_f1_idx]
                best_greg_mcc = glassoregs[best_greg_mcc_idx]
                ax.set_title(f"Shared PR Curve")
            ax.plot(recall, precision, label=model)

            best_prec_f1 = precision[best_greg_f1_idx]
            best_rec_f1 = recall[best_greg_f1_idx]
            ax.scatter(best_rec_f1, best_prec_f1, marker='x', color='black', label=f"Best F1 @ {best_greg_f1:.2e}")

            best_prec_mcc = precision[best_greg_mcc_idx]
            best_rec_mcc = recall[best_greg_mcc_idx]
            ax.scatter(best_rec_mcc, best_prec_mcc, marker='o', color='orange', label=f"Best MCC @ {best_greg_mcc:.2e}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axhline(sparsity, label="Random Model", linestyle='--', color='red')

        ax.legend()

    for i, ax in enumerate(f1mcc_axes):
        for model, metr in zip(["Cartesian LGAM", "LGAM"], [metrics, metrics_lgam]):
            if i < len(pr_axes) - 1:
                cur_f1 = metr.f1s[i]
                cur_mcc = metr.mccs[i]
                ax.set_title(f"F1 and MCC Scores for L{i+1}")
            else:
                cur_f1 = metr.shared_f1s
                cur_mcc = metr.shared_mccs
                ax.set_title("Shared F1 and MCC Scores")
            if model == "Cartesian LGAM":
                color = 'tab:blue'
            elif model == "LGAM":
                color = 'tab:orange'
            ax.plot(glassoregs, cur_f1, label=f"F1 ({model})", color=color)
            ax.plot(glassoregs, cur_mcc, label=f"MCC ({model})", color=color, linestyle='--')

        ax.set_xlabel("GLasso Penalty")
        ax.set_ylabel("F1/MCC")
        ax.set_ylim(0, 1)
        ax.set_xscale('log')
        ax.legend()

    if source_distr != "":
        source_distr += " "
    fig.suptitle(f"Problem Size: {dims} {source_distr}data with true sparsity {sparsity:.2%}")
    fig.tight_layout()

    return fig, (pr_axes, f1mcc_axes)