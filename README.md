# IML_for_Imbalanced_Learning

![Framework](/framework1.png)

This repository is for the paper, "Interpretable ML for Imbalanced Data."
It contains the code and links to obtain pre-trained models, as well as the steps, to reproduce several of the visualizations listed in the paper.  Please note that the code provided below is for the CIFAR-10 dataset.
## Steps
1. Extract FE from a trained model.
  - Start with extract_FE.py
    - link to pre-trained model used by Extract_FE: https://drive.google.com/file/d/18yWsVlUNVrSMq4qTlY2d9MPHThZaSWUK/view?usp=sharing
2. Display class accuracy.
  - class_accuracy.py
3. Visualize class archetypes (safe, border, rare, outliers) for a specific class
  - First, generate nearest neighbors with arch_NNB.py
  - Alternatively, use the CE_cif_trn_NNB.csv file in the data folder.
  - Run k_medoid_viz.py
3. Visualize nearest adversary neighbors bar chart
  - Run NNB_FP_bar.py 
4. Display feature embedding (FE) top-10 indices and FE densities
  - Run FE_idx_density.py
5. Visualize color bands for a specific class of interest
  - Run saliency_texture.py
    
    





