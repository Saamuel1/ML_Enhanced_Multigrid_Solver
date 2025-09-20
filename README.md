# Projet de Stage – Solveur Multigrille Appris avec PyTorch  

Durant mon stage, j’ai développé et expérimenté un **solveur multigrille basé sur l’apprentissage automatique** pour l’équation de Poisson 2D, en m’appuyant sur **PyTorch Lightning**.  
L’objectif était d’évaluer différentes stratégies d’optimisation des *smoothers* (opérateurs de relaxation) et des noyaux appris, afin d’améliorer la vitesse de convergence par rapport aux méthodes classiques.  

## 🔑 Contributions techniques
- **Level-wise Weighted Jacobi** : optimisation du poids central du noyau pour chaque niveau du multigrille.  
- **Fully Adaptive Smoother** (plusieurs variantes) :  
  - *Per level* : un noyau différent par niveau.  
  - *Generalized approach* : un noyau pour la descente et un autre pour la remontée.  
  - *TSK* (*Transfer Smoothing Kernels*) : deux noyaux de transfert distincts (descente / remontée).  

## 📂 Structure technique
Chaque expérimentation suit la même architecture :  
- `model_lightning_case_x.py` : définition du solveur de Poisson 2D et du **LightningModule**.  
- `train_model_case_x.py` : script d’entraînement du solveur multigrille.  
- `inference_case_x.py` : calcul des résidus et des erreurs.  
- `plot_case_x.py` : fonctions de visualisation (convergence, comparaison, noyaux appris).  
- `main_case_x.py` : configuration des paramètres, lancement des entraînements et génération des figures.  

## 📊 Résultats
- Comparaison graphique entre le **multigrille appris** et les méthodes classiques → vitesse de convergence améliorée.  
- Analyse détaillée des noyaux appris par niveau et par phase (descente / remontée).  
- Sauvegarde automatisée des modèles et des métriques via des checkpoints.  

---

👉 Ce projet illustre ma capacité à :  
- Concevoir et implémenter des **méthodes numériques hybrides** (IA + méthodes classiques).  
- Expérimenter différentes variantes d’architectures et analyser leurs performances.  
- Mettre en place une pipeline complète : **modélisation – entraînement – inférence – visualisation**.  
