DeSCOPE
=======

DeSCOPE is a single-cell perturbation prediction framework designed for scRNA-seq, scATAC-seq, and general single-cell–level perturbation modeling. It is built on a conditional Variational Autoencoder (cVAE) architecture, in which perturbed genes are represented by embeddings derived from the ESM2 protein language model and used as conditioning information to model cellular responses to genetic perturbations. Through this design, DeSCOPE delivers strong predictive performance in challenging scenarios, including unseen genes and unseen cell types.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   Installation
   Datasets
   Tutorials
   API
   Acknowledgements