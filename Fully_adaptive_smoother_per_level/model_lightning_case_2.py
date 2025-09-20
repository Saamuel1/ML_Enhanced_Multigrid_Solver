#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:56:28 2025

@author: duarteds
"""

# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import Callback

# =============================================================================
# The following fonctions reffers to the solving of the 2D poisson equation with
# Dirichlet condition, with multigrid, expressing the algorithm in terms of 
# convolution kernel composition
# =============================================================================

def get_A_Laplace(N:int) -> torch.tensor:
    """Built the discretised Laplace matrix in 2D with Dirichlet conditions"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h = 1 / N
    A_i = 1/h**2 * (np.diag([4] * (N-1)) + np.diag([-1] * (N-2), -1) + np.diag([-1] * (N-2), 1))
    I_y = 1/h**2 * np.eye(N-1)
    
    A = np.zeros(((N-1)*(N-1), (N-1)*(N-1)))
    for i in range(N-1):
        start = i * (N-1)
        end = start + (N-1)
        A[start:end, start:end] = A_i
        if i < N-2:
            A[start + (N-1):end + (N-1), start:end] = -I_y
            A[start:end, start + (N-1):end + (N-1)] = -I_y
    return torch.tensor(A, dtype=torch.float64, device=device)

def relaxation(method, K_A, v, f, m, N, K_D_inv, A):
    """Relaxation fonction, wich depending on the "method" that we choose
    will use either weighted Jacobi or weighted Gauss Seidel"""
    
    batch_size = v.shape[0]
    h = 1 / N
    D = torch.diag(torch.diag(A.squeeze(0).squeeze(0)))
    L = torch.tril(A.squeeze(0).squeeze(0))

    for _ in range(m):
        v_reshaped = v.reshape(batch_size, 1, N-1, N-1)
        r1 = f.reshape(batch_size, 1, N-1, N-1) - F.conv2d(v_reshaped, K_A, stride = 1, padding = 'same')        
        r1_scaled = h**2 * r1  # important pour l'équilibre numérique
        
        K_D_inv = K_D_inv.to(r1_scaled.device)

        if method == "jacobi":
            r2 = F.conv2d(r1_scaled, K_D_inv, stride=1, padding='same')
    
            v = v + r2.reshape(batch_size, 1, (N-1)**2, 1)
            
        elif method == "jacobi scalaire":
            r2 = r1 * h**2/4 * K_D_inv.view(1, 1, 1, 1)
            v = v + r2.reshape(batch_size, 1, (N-1)**2, 1)

        elif method == "gauss seidel":
            r2 = torch.linalg.solve(L + D, r1.reshape((N-1)**2)) * K_D_inv.view(1, 1, 1, 1)
            v = v + r2.reshape(batch_size, 1, (N-1)**2, 1)
                
    return v

def convul_V_cycle(A:torch.tensor, v:torch.tensor, f:torch.tensor, 
                   nu1:int, nu2:int, N:int, L:int, K_A:torch.tensor, K_R:torch.tensor, 
                   K_P:torch.tensor, K_D_inv_list_descente:torch.tensor, K_D_inv_list_remontee:torch.tensor, method) -> torch.tensor:
    """
    Apply the V Cycle with convolution kernel composition. This function
    uses the relaxation function defined above as the smoothing operator.

    Parameters
    ----------
    A : torch.Tensor
        Discretized Laplace matrix (on finest grid).
    v : torch.Tensor
        Initial guess for the solution (batch_size, 1, (N-1)**2, 1).
    f : torch.Tensor
        Right-hand side of the equation Av = f (same shape as v).
    nu1 : int
        Number of pre-smoothing steps.
    nu2 : int
        Number of post-smoothing steps.
    N : int
        Grid size (number of interior points in one dimension + 1).
    L : int
        Number of multigrid levels.
    K_A : list[torch.Tensor]
        List of convolution kernels approximating A at each level.
    K_R : list[torch.Tensor]
        List of restriction kernels (fine to coarse).
    K_P : list[torch.Tensor]
        List of prolongation kernels (coarse to fine).
    K_𝜔_ℓ_descent_list : list[torch.Tensor]
        List of learned relaxation weights (descent phase).
    K_𝜔_ℓ_ascent_list : list[torch.Tensor]
        List of learned relaxation weights (ascent phase).
    method : str
        Relaxation method to use ('jacobi', 'gauss-seidel', 'learned', etc.).

    Returns
    -------
    v : torch.Tensor
        Approximated solution after applying one V-cycle.
    """
    
    batch_size = np.shape(v)[0]
    
    K_D_inv_descente = K_D_inv_list_descente[L-1]
    K_D_inv_remontee = K_D_inv_list_remontee[L-1]

    # Étape 1 : Lissage préliminaire
    v = relaxation(method, K_A, v, f, nu1, N, K_D_inv_descente, A)
    v = v.reshape(v.shape[0], 1, (N-1), N-1)
    
    # Étape 2 : Calcul du résidu
    conv_out = F.conv2d(v, K_A, stride = 1, padding='same')
    residus_h = f - conv_out.reshape(v.shape[0], 1, (N-1)**2, 1)
    residus_h = residus_h.reshape(v.shape[0], 1, N-1, N-1)

    # Étape 3 : Restriction du résidu
    residus_2h = F.conv2d(residus_h, K_R, stride = 2, padding = "valid")

    # Construction de l'opérateur sur la grille grossière
    N_coarse = N // 2
    A2h = get_A_Laplace(N_coarse).view(1, 1, (N_coarse-1)**2, (N_coarse-1)**2)

    # Étape 4 : Résolution sur la grille grossière
    if L == 0:
        A2h_cpu = A2h.to("cpu")
        residus_2h_cpu = residus_2h.reshape(batch_size, 1, -1, 1).to("cpu")
        e_2h = torch.linalg.solve(A2h_cpu, residus_2h_cpu)
        e_2h = e_2h.to(v.device)
        e_2h = e_2h.reshape(batch_size, 1, N_coarse-1, N_coarse-1)
    else:
        K_A2h = (K_A / 4).to(dtype=torch.float64, device=K_A.device)
        e_2h = torch.zeros(batch_size, 1, (N_coarse-1)**2, 1, dtype=torch.float64, device=v.device)
        e_2h = convul_V_cycle(A2h, e_2h, residus_2h.reshape(v.shape[0], 1, (N_coarse-1)**2, 1), nu1, nu2, N_coarse, L - 1, K_A2h, K_R, K_P, K_D_inv_list_descente, K_D_inv_list_remontee, method).reshape(v.shape[0], 1, N_coarse-1, N_coarse-1)

    # Étape 5 : Prolongation et correction
    e_h = F.conv_transpose2d(e_2h, K_P, stride=2, padding=0).reshape(batch_size, 1, (N-1)**2, 1)

    v = v.reshape(batch_size, 1, (N-1)**2, 1) + e_h
    v = relaxation(method, K_A, v, f, nu2, N, K_D_inv_remontee, A)
    
    return v

# =============================================================================
# The following fonctions reffers to the LightningModule principal
# =============================================================================

class MultigridLightningSolver(pl.LightningModule):
    
    def __init__(self, N, L, nu1, nu2, batch_size, lr, nbr_iter, method, loss):
        
        super().__init__()
        
        self.save_hyperparameters()
        
        h = 1 / N
        self.device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.K_A = (1/h**2) * torch.tensor([[[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]], dtype=torch.float64, device=self.device_)
        self.K_R = (1/16) * torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], dtype=torch.float64, device=self.device_)
        self.K_P = torch.tensor([[[[1/4, 1/2, 1/4], [1/2, 1, 1/2], [1/4, 1/2, 1/4]]]], dtype=torch.float64, device=self.device_)
        
        self.K_D_inv_list_descente = nn.ParameterList([nn.Parameter(torch.tensor([[[[0., 0., 0.], [0., h**2/4, 0.], [0., 0., 0.]]]], dtype=torch.float64, device=self.device_)) for _ in range(L)])
        self.K_D_inv_list_remontee = nn.ParameterList([nn.Parameter(torch.tensor([[[[0., 0., 0.], [0., h**2/4, 0.], [0., 0., 0.]]]], dtype=torch.float64, device=self.device_)) for _ in range(L)])


        self.A_Laplace = get_A_Laplace(N).view(1, 1, (N-1)**2, (N-1)**2)

    def forward(self, v, f, v_ref):
        """
        Apply multiple V-cycle iterations and return all losses.

        Parameters
        ----------
        v : torch.Tensor
            Initial guess.
        f : torch.Tensor
            Right-hand side (source term).
        v_ref : torch.Tensor
            Ground-truth solution (for loss computation).

        Returns
        -------
        v : torch.Tensor
            Approximate solution after V-cycles.
        losses : list[torch.Tensor]
            List of MSE losses for each iteration.
        """
        losses = []
        batch_size = np.shape(v)[0]

        for _ in range(self.hparams.nbr_iter):
            v = convul_V_cycle(self.A_Laplace, v, f, self.hparams.nu1, self.hparams.nu2,
                                  self.hparams.N, self.hparams.L, self.K_A, self.K_R, self.K_P,
                                  self.K_D_inv_list_descente, self.K_D_inv_list_remontee, self.hparams.method)
            
            v_image = v.reshape(batch_size, 1, self.hparams.N-1, self.hparams.N-1) 
            f_image = f.reshape(batch_size, 1, self.hparams.N-1, self.hparams.N-1)
            
            residual = f_image - F.conv2d(v_image, self.K_A, stride=1, padding='same')
            
            if self.hparams.loss == "res":
            
                r = torch.norm(residual) / torch.norm(f_image)
            else:
                
                r = torch.norm(v_ref - v)/torch.norm(v_ref)

            losses.append(r)
            
        return v, losses

    def configure_optimizers(self):
        """
        Configure optimizer.

        Returns
        -------
        torch.optim.Optimizer
            Adam optimizer with learning rate from hyperparameters.
        """
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

    def generate_dataset(self, num_samples):
        """
        Generate synthetic training/validation data.

        Parameters
        ----------
        num_samples : int
            Number of (v_ref, f) pairs to generate.

        Returns
        -------
        torch.utils.data.TensorDataset
            Dataset with v_ref as targets and f = A v_ref as inputs.
        """
        torch.manual_seed(42)
        v_refs = torch.rand(num_samples, 1, (self.hparams.N - 1)**2, 1, device=self.device_, dtype=torch.float64)
        f_all = torch.matmul(self.A_Laplace, v_refs)
        return TensorDataset(v_refs, f_all)
    
    def training_step(self, batch, batch_idx):
        v_ref, f = batch
        v = torch.zeros_like(v_ref)
        v_pred, losses = self(v, f, v_ref)
        for i, loss in enumerate(losses):
            self.log(f'train_loss_cycle_{i}', loss, prog_bar=True)
        weight_list = [1.0]
        
        for i in range(len(losses) - 1):
            ratio = (losses[i] / (losses[i + 1])).detach()
            weight_list.append(weight_list[-1] * ratio)
            
        weighted_losses = [w * l for w, l in zip(weight_list, losses)]
        loss_finale = torch.stack(weighted_losses).sum()
        self.log("train_loss", loss_finale)
        
        return loss_finale

    
    def validation_step(self, batch, batch_idx):
        v_ref, f = batch
        v = torch.zeros_like(v_ref)
        v_pred, losses = self(v, f, v_ref)
        for i, loss in enumerate(losses):
            self.log(f'val_loss_cycle_{i}', loss, prog_bar=True)
        weight_list = [1.0]
        
        for i in range(len(losses) - 1):
            ratio = (losses[i] / (losses[i + 1])).detach()
            weight_list.append(weight_list[-1] * ratio)
            
        weighted_losses = [w * l for w, l in zip(weight_list, losses)]
        loss_finale = torch.stack(weighted_losses).sum()
        self.log("val_loss", loss_finale)
        
        return loss_finale


    def train_dataloader(self):
        """
        Returns the training DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
        """
        dataset = self.generate_dataset(10000)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
        """
        dataset = self.generate_dataset(5000)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False)
    
# ===============================
#         Callbacks
# ===============================

class TrainLossPerCycleHistory(pl.Callback):
    """
    Callback to track training loss at each V-cycle iteration.
    
    Parameters
    ----------
    nbr_iter : int
        Number of V-cycle iterations.
    """
    def __init__(self, nbr_iter):
        self.nbr_iter = nbr_iter
        self.train_losses_per_cycle = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Log training loss per cycle at the end of each epoch.
        """
        cycle_losses_train = []
        for i in range(self.nbr_iter):
            key = f"train_loss_cycle_{i}"
            if key in trainer.callback_metrics:
                loss_value_train = trainer.callback_metrics[key].detach().cpu().item()
                cycle_losses_train.append(loss_value_train)
        if cycle_losses_train:
            self.train_losses_per_cycle.append(cycle_losses_train)


class ValLossPerCycleHistory(pl.Callback):
    """
    Callback to track validation loss at each V-cycle iteration.
    
    Parameters
    ----------
    nbr_iter : int
        Number of V-cycle iterations.
    """
    def __init__(self, nbr_iter):
        self.nbr_iter = nbr_iter
        self.val_losses_per_cycle = []

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Log validation loss per cycle at the end of each epoch.
        """
        cycle_losses_val = []
        for i in range(self.nbr_iter):
            key = f"val_loss_cycle_{i}"
            if key in trainer.callback_metrics:
                loss_value_val = trainer.callback_metrics[key].detach().cpu().item()
                cycle_losses_val.append(loss_value_val)
        if cycle_losses_val:
            self.val_losses_per_cycle.append(cycle_losses_val)


class KDInvLogger(pl.Callback):
    """
    Callback to log the evolution of learned K_D_inv weights across epochs.
    """
    def __init__(self):
        self.history_descente = []
        self.history_remontee = []

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Save the current learned stencil weights at the end of each training epoch.
        """
        desc = [param.detach().cpu().flatten().tolist() for param in pl_module.K_D_inv_list_descente]
        rem = [param.detach().cpu().flatten().tolist() for param in pl_module.K_D_inv_list_remontee]
        self.history_descente.append(desc)
        self.history_remontee.append(rem)
        
        
        
    
    
    
    
    
    
    
    
    
