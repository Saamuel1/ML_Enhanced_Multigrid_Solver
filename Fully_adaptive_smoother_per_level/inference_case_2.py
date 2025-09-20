#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:57:10 2025

@author: duarteds
"""

import torch
import numpy as np
from model_lightning_case_1 import convul_V_cycle
import torch.nn.functional as F


def run_inference_batch(model, v_ref_batch, A, f_batch, max_iter, N, L, nu1, nu2, omega_desc, omega_rem, loss, method):
    """
    Giving a list of tensors omega_desc and omega_rem, this function will run the inference, computing the 
    residual for further plots.

    Parameters
    ----------
    model : list[torch.Tensor]
        Call everything that is set in the function __init__ in the class MultigridLightningSolver.
    v_ref_batch : torch.tensor
        Ground-truth solution, with size: batch size, used for training and validation.
    A : torch.tensor
        Discretized Laplace matrix (on finest grid).
    f_batch : torch.tensor
        Right-hand side of the equation Av = f, same shap as v_ref_batch
    max_iter : int
        Number of multigrid iterations
    N : int
        Grid size (number of interior points in one dimension + 1).
    L : int
        Number of multigrid levels.
    nu1 : int
        Number of pre-smoothing steps.
    nu2 : int
        Number of post-smoothing steps.
    omega_desc : list[torch.Tensor]
        DESCRIPTION.
    omega_rem : list[torch.Tensor]
        DESCRIPTION.
    method : str
        Relaxation method to use ('jacobi', 'gauss-seidel')

    Returns
    -------
    ERR_all : numpy.ndarray
        Contains the residual for the max_iter iterations.

    """
    nbr_vecteur = v_ref_batch.shape[0]
    ERR_all = np.zeros((nbr_vecteur, max_iter))

    for j in range(nbr_vecteur):
        v = torch.zeros_like(v_ref_batch[j])
        v_ref = v_ref_batch[j].unsqueeze(0)
        f = f_batch[j].unsqueeze(0)
        v = v.unsqueeze(0)

        for i in range(max_iter):
            v = convul_V_cycle(A, v, f, nu1, nu2, N, L, model.K_A, model.K_R, model.K_P, omega_desc, omega_rem, method)
            v_image = v.reshape(1, 1, N-1, N-1) 
            f_image = f.reshape(1, 1, N-1, N-1)
            if loss == "res":
                ERR_all[j, i] = torch.norm(f_image - F.conv2d(v_image, model.K_A, stride = 1, padding='same')).item()/torch.norm(f_image).item()
            else:
                ERR_all[j, i] = torch.norm(v_ref - v).item()/torch.norm(v_ref).item()


    return ERR_all


