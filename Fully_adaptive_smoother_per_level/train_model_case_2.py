#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:57:00 2025je l

@author: duarteds
"""

import numpy as np
from model_lightning_case_1 import MultigridLightningSolver, TrainLossPerCycleHistory, ValLossPerCycleHistory, KDInvLogger
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import RichProgressBar


"""
Script to launch the training of a convolutional multigrid solver using PyTorch Lightning.
It initializes the model, sets up the training callbacks, and logs checkpoints and metrics.
"""

save_dir_checkpoint = "Model_Checkpoint"
save_dir_metriques = "Sauvegardes_Metriques"

os.makedirs(save_dir_checkpoint, exist_ok=True)

def run_training(N, L, nu1, nu2, batch_size, lr, max_epochs, method, nbr_iter, loss):
    """
    Run the training of a multigrid solver model with learned relaxation kernels.

    Parameters
    ----------
    N : int
        Grid size (number of total points per axis).
    L : int
        Number of multigrid levels.
    nu1 : int
        Number of pre-smoothing steps per V-cycle.
    nu2 : int
        Number of post-smoothing steps per V-cycle.
    batch_size : int
        Batch size used during training and validation.
    lr : float
        Learning rate for the optimizer.
    max_epochs : int
        Total number of training epochs.
    method : str
        Relaxation method ('jacobi', 'gauss-seidel', or 'learned').
    nbr_iter : int
        Number of V-cycle iterations per training step.

    Returns
    -------
    model : MultigridLightningSolver
        Trained PyTorch Lightning model instance.
    train_loss_tracker : TrainLossPerCycleHistory
        Callback object containing per-iteration training loss history.
    val_loss_tracker : ValLossPerCycleHistory
        Callback object containing per-iteration validation loss history.
    k_tracker : KDInvLogger
        Callback object tracking the evolution of learned kernels.
    """
    model = MultigridLightningSolver(N, L, nu1, nu2, batch_size, lr, nbr_iter, method, loss)
    train_loss_tracker = TrainLossPerCycleHistory(nbr_iter)
    val_loss_tracker = ValLossPerCycleHistory(nbr_iter)
    k_tracker = KDInvLogger()

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
        callbacks=[RichProgressBar(), train_loss_tracker, val_loss_tracker, k_tracker],
        accelerator="auto",
        enable_checkpointing=False
    )
    trainer.fit(model)
    
    # Sauvegarde du modèle appris
    
    checkpoint_name = f"model_checkpoint_pour_N={N}_L={L}_max_epochs={max_epochs}_nombre iter={nbr_iter}_loss={loss}.ckpt"
    save_path = os.path.join(save_dir_checkpoint, checkpoint_name)
    trainer.save_checkpoint(save_path)
    
    # Sauvegarde des métriques de suivi
    
    np.save(f"{save_dir_metriques}/train_losses_N={N}_L={L}_nbr_iter={nbr_iter}.npy", train_loss_tracker.train_losses_per_cycle)
    np.save(f"{save_dir_metriques}/val_losses_N={N}_L={L}_nbr_iter={nbr_iter}.npy", val_loss_tracker.val_losses_per_cycle)
    np.save(f"{save_dir_metriques}/stencils_descente_N={N}_L={L}_nbr_iter={nbr_iter}.npy", k_tracker.history_descente)
    np.save(f"{save_dir_metriques}/stencils_remontee_N={N}_L={L}_nbr_iter={nbr_iter}.npy", k_tracker.history_remontee)
    

    return model, train_loss_tracker, val_loss_tracker, k_tracker
