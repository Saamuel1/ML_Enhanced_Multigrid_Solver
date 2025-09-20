#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:57:20 2025

@author: duarteds

This module contains plotting utilities for visualizing the convergence,
stencil evolution, and training/validation losses of a multigrid solver.
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

def plot_convergence(ERR_learned, ERR_classique_jacobi, ERR_classique_gauss_seidel, N, L, epochs, nbr_iter, loss):
    """
    Plot the convergence of the learned multigrid solver compared to classical methods.

    Parameters
    ----------
    ERR_learned : np.ndarray
        2D array (n_vectors, n_iters) of errors for the learned solver.
    ERR_classical_jacobi : np.ndarray
        2D array of errors for the classical Jacobi-based multigrid solver.
    ERR_classical_gauss_seidel : np.ndarray
        2D array of errors for the classical Gauss-Seidel-based multigrid solver.
    N : int
        Grid size.
    L : int
        Number of levels in the multigrid hierarchy.
    epochs : int
        Number of training epochs.
    nbr_iter : int
        Number of multigrid iterations per epoch.

    Returns
    -------
    None. Saves a `.png` plot and displays it.
    """
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 24})
    threshold = 1e-17

    # ───── Copier et masquer les valeurs < 1e-12 ─────
    ERR_learned_plot = ERR_learned.copy()
    ERR_learned_plot[ERR_learned_plot < threshold] = np.nan

    ERR_jacobi_plot = ERR_classique_jacobi.copy()
    ERR_jacobi_plot[ERR_jacobi_plot < threshold] = np.nan

    ERR_gs_plot = ERR_classique_gauss_seidel.copy()
    ERR_gs_plot[ERR_gs_plot < threshold] = np.nan

    # ───── Courbes individuelles de la méthode apprise (optionnel, très transparent) ─────
    for i in range(ERR_learned_plot.shape[0]):
        plt.semilogy(ERR_learned_plot[i], linestyle='-', color='salmon', alpha=0.1)

    # ───── Moyenne des erreurs apprises (rouge) ─────
    mean_err = np.nanmean(ERR_learned_plot, axis=0)
    plt.semilogy(mean_err, marker='o', color='darkred', linewidth=3, markersize=6,
                 label=r"Learned")

    # ───── Courbes classiques avec coupure < 1e-12 ─────
    plt.semilogy(ERR_jacobi_plot[0], linestyle='--', marker='s', color='navy', linewidth=3, markersize=6,
                 label=r"Classical multigrid (Jacobi)")

    plt.semilogy(ERR_gs_plot[0], linestyle='-', marker='^', color='forestgreen', linewidth=3, markersize=6,
                 label=r"Classical multigrid (Gauss-Seidel)")

    # ───── Mise en forme ─────
    plt.xlabel("Number of multigrid iterations", fontsize=26)
    if loss == "res":
        plt.ylabel(r"Residual: $\|f - Au^{(m)}\|_2 / \|f\|_2$", fontsize=26)
    else:
        plt.ylabel(r"Error: $\|u_{ref} - u^{(m)}\|_2 / \|u_{ref}\|_2$", fontsize=26)
    plt.grid(True, which="both", linestyle=":")
    plt.legend(loc="upper right", fontsize=20)
    plt.ylim(1e-16, 1e0) 

    plt.tight_layout()

    # ───── Enregistrement ─────
    filename = f"Convergence_Learned_vs_Classical_N={N}_L={L}_epochs={epochs}_iter={nbr_iter}_loss={loss}.png"
    save_path = os.path.join(
        "/scratch/algo/duarteds/stage/Fully_adaptive_smoother_per_level/Images/Convergence_Multigrille",
        filename
    )
    plt.savefig(save_path, dpi=300)
    plt.show()

    

def plot_compare(train_losses, val_losses, k_history, N, L, mode = "mode"):
    """
    Plot the evolution of the middle value of the learned K_D_inv values across epochs,
    for each level of the multigrid, separately for descent and ascent phases.

    Parameters
    ----------
    k_desc : list[list[float]]
        History of K_D_inv values during the descent phase (epochs × levels).
    k_rem : list[list[float]]
        History of K_D_inv values during the ascent phase (epochs × levels).
    N : int
        Grid size.
    L : int
        Number of levels in the multigrid hierarchy.
    epochs : int
        Total number of training epochs.
    nbr_iter : int
        Number of V-cycle iterations used per epoch.

    Returns
    -------
    None. Saves a `.png` plot and displays it.
    """
    fig1, axs1 = plt.subplots(2, 1, figsize=(8, 8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    epochs = np.shape(train_losses)[0]
    train_losses = train_losses[:epochs]
    val_losses = val_losses[:epochs]
    epochs = np.shape(train_losses)[0]
    # 1. Stencils de descente
    for i, values in enumerate(zip(*k_history)):
        V=[]
        color = colors[i % len(colors)]

        for j in range(epochs):
            V.append(values[j][4])
        
        axs1[0].plot(range(epochs), V, label=f"Niveau {i+1}")
        last_val = float(values[-1][4])      
        axs1[0].annotate(f"{last_val:.4f}",
                     xy=(epochs, last_val),
                     color=color,
                     fontsize=9,
                     fontweight='bold',
                     ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1))
    axs1[0].set_title(f"Évolution de K_D_inv ({mode}) pour N = {N}, L = {L} et epochs = {epochs}")
    axs1[0].set_xlabel("Epoch")
    axs1[0].set_ylabel("Valeur")
    axs1[0].legend()
    axs1[0].grid(True)
    
    # 2. Courbes de loss (train + val)
    axs1[1].semilogy(range(epochs), train_losses, label="Loss d'entraînement", color="blue")
    axs1[1].semilogy(range(epochs), val_losses, label="Loss de validation", color="orange")
    axs1[1].set_title("Loss d'entraînement et de validation")
    axs1[1].set_xlabel("Epoch")
    axs1[1].set_ylabel("Loss")
    axs1[1].legend()
    axs1[1].grid(True)
    
    filename = f"({mode}) - Evolution de K_D_inv superposé aux loss pour N = {N}, L = {L} et nombre d'epochs = {epochs}.png"
    save_path = os.path.join("/scratch/algo/duarteds/stage/Fully_adaptive_smoother_per_level/Images/Compare_Loss_Stencils", filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_valeurs_stencils(KD_stencils, N, L, epochs, nbr_iter, mode="descente"):
    L = len(KD_stencils)
    stencils_array = []

    print(f"\nLearned stencils K_D_inv — {mode} phase:")

    for i, param in enumerate(KD_stencils):
        stencil = param.detach().cpu().squeeze().numpy()
        stencils_array.append(stencil)

        print(f"\nLevel {i+1} :")
        print(np.round(stencil, 3))

    stencils_array = np.stack(stencils_array, axis=0)  # Shape: (L, 3, 3)
    mean_stencil = np.mean(stencils_array, axis=0)

    print(f"\n=== Mean stencil for {mode} phase ===")
    print(np.round(mean_stencil, 3))

    # Commented out: Heatmap display of individual stencils
    """
    fig, axs = plt.subplots(1, L, figsize=(3.5 * L, 3.5))
    if L == 1:
        axs = [axs]

    for i, (stencil, ax) in enumerate(zip(stencils_array, axs)):
        sns.heatmap(
            stencil,
            cmap='binary_r',
            annot=True,
            fmt=".2f",
            cbar=False,
            ax=ax,
            square=True,
            linewidths=0.5,
            linecolor='gray',
            annot_kws={"size": 11, "color": "black", "weight": "bold"}
        )

        ax.set_title(f"Level {i+1} — {mode.capitalize()} phase", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    filename = f"Stencil_{mode}_N{N}_L{L}_epochs{epochs}_iter{nbr_iter}.png"
        save_path = os.path.join("/scratch/algo/duarteds/stage/Fully_adaptive_smoother_per_level/Images/Compare_Loss_Stencils", filename)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    """
    return mean_stencil

