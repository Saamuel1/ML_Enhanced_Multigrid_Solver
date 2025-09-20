#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:57:32 2025

@author: duarteds
"""

import numpy as np
from train_model_case_1 import run_training
from inference_case_1 import run_inference_batch
from plot_case_1 import plot_convergence, plot_compare, plot_valeurs_stencils
import torch
from model_lightning_case_1 import MultigridLightningSolver, TrainLossPerCycleHistory, ValLossPerCycleHistory, KDInvLogger
import os
# %%
save_dir_checkpoint = "Model_Checkpoint"
save_dir_metriques = "Sauvegardes_Metriques"
os.makedirs(save_dir_checkpoint, exist_ok=True)

# Paramètres
N = 64
L = 4
nu1 = 1
nu2 = 1
batch_size = 64
lr = 1e-4
omega = 4/5
max_epochs = 100
nbr_iter = 10
loss = "res"


load_model = False

# Vérification de la validité de N et L
if 2**(L + 1) >= N:
    raise ValueError(f"Condition invalide : 2^(L + 1) < N n'est pas respectée.")


if load_model:
    checkpoint_name = f"model_checkpoint_pour_N={N}_L={L}_max_epochs={max_epochs}_nombre iter={nbr_iter}_loss={loss}.ckpt"
    save_path = os.path.join(save_dir_checkpoint, checkpoint_name)
    print(f" Chargement du modèle depuis : {save_path}")

    model = MultigridLightningSolver.load_from_checkpoint(save_path)
    model.eval()
else:
    
    print("Entraînement du modèle")
    model, train_loss_tracker, val_loss_tracker, k_tracker = run_training(N, L, nu1, nu2, batch_size, lr, max_epochs, "jacobi", nbr_iter, loss)
    
seed = 42
torch.manual_seed(seed)
max_iter = 40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Données test
nbr_vecteur = 1   
v_ref_test = torch.rand((nbr_vecteur, 1, (N - 1)**2, 1), dtype=torch.float64, device = device)
A = model.A_Laplace
f_test = A @ v_ref_test

train_losses = np.load(f"{save_dir_metriques}/train_losses_N={N}_L={L}_nbr_iter={nbr_iter}.npy")
val_losses = np.load(f"{save_dir_metriques}/val_losses_N={N}_L={L}_nbr_iter={nbr_iter}.npy")
stencils_desc = np.load(f"{save_dir_metriques}/stencils_descente_N={N}_L={L}_nbr_iter={nbr_iter}.npy")
stencils_rem = np.load(f"{save_dir_metriques}/stencils_remontee_N={N}_L={L}_nbr_iter={nbr_iter}.npy")

KD_stencils_descente = [stencil.detach().cpu() for stencil in model.K_D_inv_list_descente]
KD_stencils_remontee = [stencil.detach().cpu() for stencil in model.K_D_inv_list_remontee]

ERR_appris = run_inference_batch(model, v_ref_test, A, f_test, max_iter, N, L, nu1, nu2, KD_stencils_descente, KD_stencils_remontee, loss, "jacobi")

omega_const_jacobi = [torch.tensor([[[[omega]]]], dtype=torch.float64, device = device) for _ in range(L)]
print(omega_const_jacobi[L-1])
ERR_classique_jacobi = run_inference_batch(model, v_ref_test, A, f_test, max_iter, N, L, nu1, nu2, omega_const_jacobi, omega_const_jacobi, loss, "jacobi scalaire")
omega_const_gauss_seidel = [torch.tensor([[[[1.8]]]], dtype=torch.float64) for _ in range(L)]
ERR_classique_gauss_seidel = run_inference_batch(model, v_ref_test, A, f_test, max_iter, N, L, nu1, nu2, omega_const_gauss_seidel, omega_const_gauss_seidel, loss, "gauss seidel")

epochs = np.shape(train_losses)[0]

plot_compare(train_losses, val_losses, stencils_desc, N, L, mode="descente")
plot_compare(train_losses, val_losses, stencils_rem, N, L, mode="remontee")

plot_convergence(ERR_appris, ERR_classique_jacobi, ERR_classique_gauss_seidel, N, L, epochs, nbr_iter, loss)
    
plot_valeurs_stencils(KD_stencils_descente, N, L, epochs,  nbr_iter, mode="descente")
plot_valeurs_stencils(KD_stencils_remontee, N, L, epochs, nbr_iter, mode="remontee")

    
    
    
    
    
    
    
    
    
    
    
