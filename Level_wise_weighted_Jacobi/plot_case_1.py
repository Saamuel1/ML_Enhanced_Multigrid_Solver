#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 10 10:57:20 2025

@author: duarteds

This module contains plotting utilities for visualizing the convergence,
stencil evolution, and training/validation losses of a multigrid solver.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(ERR_learned, ERR_classical_jacobi, ERR_classical_gauss_seidel, N, L, epochs, nbr_iter, loss):
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

    ERR_jacobi_plot = ERR_classical_jacobi.copy()
    ERR_jacobi_plot[ERR_jacobi_plot < threshold] = np.nan

    ERR_gs_plot = ERR_classical_gauss_seidel.copy()
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
    
    plt.ylim(1e-16, 1e0) 
    
    plt.legend(loc="upper right", fontsize=20)
    plt.tight_layout()

    # ───── Enregistrement ─────
    filename = f"Convergence_Learned_vs_Classical_N={N}_L={L}_epochs={epochs}_iter={nbr_iter}_loss={loss}.png"
    save_path = os.path.join(
        "/scratch/algo/duarteds/stage/Level_wise_weighted_Jacobi/Images/Convergence_Multigrille",
        filename
    )
    plt.savefig(save_path, dpi=300)
    plt.show()




def plot_compare_stencil(k_desc, k_rem, N, L, epochs, nbr_iter):
    """
    Plot the evolution of the learned K_D_inv values across epochs,
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
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (desc_values, rem_values) in enumerate(zip(zip(*k_desc), zip(*k_rem))):
        color = colors[i % len(colors)]
        axs[0].plot(range(epochs), desc_values, label=f"Level {i+1}", color=color)
        last_val_desc = float(desc_values[-1])
        axs[0].annotate(f"{last_val_desc:.4f}",
                        xy=(epochs, last_val_desc),
                        color=color,
                        fontsize=9,
                        fontweight='bold',
                        ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1))

        axs[1].plot(range(epochs), rem_values, label=f"Level {i+1}", color=color)
        last_val_rem = float(rem_values[-1])
        axs[1].annotate(f"{last_val_rem:.4f}",
                        xy=(epochs, last_val_rem),
                        color=color,
                        fontsize=9,
                        fontweight='bold',
                        ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1))

    axs[0].set_title("Evolution of K_D_inv (downward)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Value")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_title("Evolution of K_D_inv (upward)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Value")
    axs[1].legend()
    axs[1].grid(True)

    save_path = os.path.join(
        "/scratch/algo/duarteds/stage/Level_wise_weighted_Jacobi/Images/Compare_Loss_Stencils",
        f"K_D_inv_Evolution_down_vs_up_N={N}_L={L}_epochs={epochs}_iter={nbr_iter}.png"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_losses(train_losses, val_losses, N, L, epochs, nbr_iter):
    """
    Plot training and validation loss over epochs.

    Parameters
    ----------
    train_losses : list[float]
        List of training losses per epoch.
    val_losses : list[float]
        List of validation losses per epoch.
    N : int
        Grid size.
    L : int
        Number of multigrid levels.
    epochs : int
        Number of training epochs.
    nbr_iter : int
        Number of V-cycle iterations per epoch.

    Returns
    -------
    None. Saves a `.png` plot and displays it.
    """
    plt.figure(figsize=(8, 6))
    plt.semilogy(range(epochs), train_losses, label="Training loss", color="blue")
    plt.semilogy(range(epochs), val_losses[1:], label="Validation loss", color="orange")
    plt.title(f"Training and validation loss — N={N}, L={L}, epochs={epochs}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(
        "/scratch/algo/duarteds/stage/Level_wise_weighted_Jacobi/Images/Compare_Loss_Stencils",
        f"Loss_train_val_N={N}_L={L}_epochs={epochs}_iter={nbr_iter}.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.show()
