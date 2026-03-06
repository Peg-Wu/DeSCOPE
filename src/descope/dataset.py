import torch
import random
import logging
import datasets
import numpy as np
import pandas as pd
import scanpy as sc

from typing import Union, Optional
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from .tokenizer import _filter_perturbations
from .utils import (
    load_gene_embs,
    preprocess_atac_perturbation_adata_consistent_with_epiagent,
    preprocess_rna_perturbation_adata
)

logger = logging.getLogger(__name__)


# Slow | Deprecated
class BaseDataset(Dataset, ABC):
    MAIN_INPUT_NAME = None
    RANDOM_MAPPING_CONTROL_TO_CONTROL = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is BaseDataset:
            return
        
        if not hasattr(cls, 'MAIN_INPUT_NAME') or cls.MAIN_INPUT_NAME == None:
            raise NotImplementedError(
                f"Class {cls.__name__} must define the class attribute `MAIN_INPUT_NAME` with a non-None value."
            )

    def __init__(
        self,
        adata: Union[str, sc.AnnData],
        pert_col: str = "perturbation",
        ctrl_name: str = "control",
        perts_to_include: Optional[list] = None,
        perts_to_exclude: Optional[list] = None,
        gene_embs_file: str = "./ESM2_pert_features.pt"
    ):
        super().__init__()
        self.pert_col = pert_col
        self.ctrl_name = ctrl_name
        self.perts_to_include = perts_to_include
        self.perts_to_exclude = perts_to_exclude

        if isinstance(adata, str):
            logger.info(f"Read anndata from {adata} ...")
            adata = sc.read_h5ad(adata)

        if hasattr(adata.X, "toarray"):
            adata.X = adata.X.toarray()

        adata = self.preprocess_adata(adata)  # abstract
        self.adata = _filter_perturbations(
            adata=adata,
            pert_col=pert_col,
            ctrl_name=ctrl_name,
            perts_to_include=perts_to_include,
            perts_to_exclude=perts_to_exclude
        )
        self.ctrl_cell_indices = self.get_ctrl_cell_indices(self.adata)
        self.gene_embs = load_gene_embs(
            gene_embs_file=gene_embs_file,
            perts_to_emb=self.adata.obs[self.pert_col].unique().tolist()
        )
    
    def get_ctrl_cell_indices(self, adata: sc.AnnData) -> list[int]:
        ctrl_cell_indices = np.where(adata.obs[self.pert_col] == self.ctrl_name)[0]
        if len(ctrl_cell_indices) == 0:
            raise ValueError("No control cells found!")
        return ctrl_cell_indices

    def __getitem__(self, idx):
        adata_pert = self.adata[idx]
        pert_name = adata_pert.obs[self.pert_col].item()
        pert_gene_emb = self.gene_embs[pert_name].to(torch.float32)

        if pert_name != self.ctrl_name or self.RANDOM_MAPPING_CONTROL_TO_CONTROL:
            random_ctrl_idx = np.random.choice(self.ctrl_cell_indices)
            adata_ctrl = self.adata[random_ctrl_idx]
            basal_sequence = torch.tensor(
                adata_ctrl.X.reshape(-1), dtype=torch.float32
            )
            labels = torch.tensor(
                adata_pert.X.reshape(-1), dtype=torch.float32
            )
        else:
            basal_sequence = torch.tensor(
                adata_pert.X.reshape(-1), dtype=torch.float32
            )
            labels = basal_sequence.clone()

        return {self.MAIN_INPUT_NAME: basal_sequence, "pert_gene_emb": pert_gene_emb, "labels": labels}

    def __len__(self) -> int:
        return len(self.adata)

    @abstractmethod
    def preprocess_adata(self, adata: sc.AnnData) -> sc.AnnData:
        raise NotImplementedError()


# Slow | Deprecated
class DatasetForATAC(BaseDataset):
    MAIN_INPUT_NAME = "ctrl_cell_tf_idf"
    RANDOM_MAPPING_CONTROL_TO_CONTROL = False
    def __init__(
        self,
        adata: Union[str, sc.AnnData],
        pert_col: str = "perturbation",
        ctrl_name: str = "control",
        topk_ccres: int = 50000,
        perts_to_include: Optional[list] = None,
        perts_to_exclude: Optional[list] = None,
        gene_embs_file: str = "./ESM2_pert_features.pt"
    ):
        self.topk_ccres = topk_ccres
        super().__init__(
            adata=adata,
            pert_col=pert_col,
            ctrl_name=ctrl_name,
            perts_to_include=perts_to_include,
            perts_to_exclude=perts_to_exclude,
            gene_embs_file=gene_embs_file
        )

    def preprocess_adata(self, adata: sc.AnnData) -> sc.AnnData:
        return preprocess_atac_perturbation_adata_consistent_with_epiagent(
            adata, self.topk_ccres, self.pert_col
        )


# Slow | Deprecated
class DatasetForRNA(BaseDataset):
    MAIN_INPUT_NAME = "ctrl_cell_expr"
    RANDOM_MAPPING_CONTROL_TO_CONTROL = False
    def __init__(
        self,
        adata: Union[str, sc.AnnData],
        pert_col: str = "target_gene",
        ctrl_name: str = "non-targeting",
        target_sum: float = 1e4,
        skip_raw_counts_check: bool = False,
        perts_to_include: Optional[list] = None,
        perts_to_exclude: Optional[list] = None,
        gene_embs_file: str = "./ESM2_pert_features.pt"
    ):
        self.target_sum = target_sum
        self.skip_raw_counts_check = skip_raw_counts_check
        super().__init__(
            adata=adata,
            pert_col=pert_col,
            ctrl_name=ctrl_name,
            perts_to_include=perts_to_include,
            perts_to_exclude=perts_to_exclude,
            gene_embs_file=gene_embs_file
        )

    def preprocess_adata(self, adata: sc.AnnData) -> sc.AnnData:
        return preprocess_rna_perturbation_adata(
            adata=adata, 
            target_sum=self.target_sum, 
            pert_col=self.pert_col,
            skip_raw_counts_check=self.skip_raw_counts_check
        )


# Fast
class HFBaseDataset(Dataset):
    MAIN_INPUT_NAME = None
    RANDOM_MAPPING_CONTROL_TO_CONTROL = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is HFBaseDataset:
            return
        
        if not hasattr(cls, 'MAIN_INPUT_NAME') or cls.MAIN_INPUT_NAME == None:
            raise NotImplementedError(
                f"Class {cls.__name__} must define the class attribute `MAIN_INPUT_NAME` with a non-None value."
            )

    def __init__(
        self,
        hf_dataset: datasets.Dataset,
        ctrl_name: str = "control",
        gene_embs_file: str = "./ESM2_pert_features.pt"
    ):
        super().__init__()
        self._check_hf_dataset_features(hf_dataset)
        self.ds = hf_dataset  # features in self.ds: labels, pert_gene, celltype
        self.ctrl_name = ctrl_name
        self.ctrl_cell_indices = self.get_ctrl_cell_indices_for_each_celltype()
        self.gene_embs = load_gene_embs(
            gene_embs_file=gene_embs_file,
            perts_to_emb=self.ds.unique("pert_gene")
        )  # {gene: torch.Tensor}

        # preprocess hf dataset
        self._preprocess_hf_dataset()  # features in self.ds: labels, pert_gene, pert_gene_emb, celltype
    
    @staticmethod
    def _check_hf_dataset_features(hf_dataset: datasets.Dataset):
        missing_features = ["labels", "pert_gene", "celltype"]
        for feature in hf_dataset.features:
            if feature in ["labels", "pert_gene", "celltype"]:
                missing_features.remove(feature)
        if len(missing_features) > 0:
            raise ValueError(
                f"The following features are missing from the HuggingFace dataset: {missing_features}. "
                "Please make sure that the dataset contains the following features: labels, pert_gene, celltype."
            )
    
    def get_ctrl_cell_indices_for_each_celltype(self) -> dict[str, list[int]]:
        celltype = np.array(self.ds["celltype"])
        pert_gene = np.array(self.ds["pert_gene"])

        df = pd.DataFrame({
            "celltype": celltype,
            "pert_gene": pert_gene
        })

        ctrl_cell_indices = {
            celltype: indices
            for (celltype, pert_gene), indices in df.groupby(["celltype", "pert_gene"]).groups.items() 
            if pert_gene == self.ctrl_name
        }

        for celltype, indices in ctrl_cell_indices.items():
            if len(indices) == 0:
                raise ValueError(f"No control cells found for celltype {celltype}!")
            
        return ctrl_cell_indices

    def _preprocess_hf_dataset(self):
        # Step1: Add pert_gene_emb to hf dataset
        gene_embs = {gene: embs.numpy() for gene, embs in self.gene_embs.items()}
        self.ds = self.ds.add_column("pert_gene_emb", pd.Series(self.ds["pert_gene"]).map(gene_embs).tolist())

        # Step2: Set format to torch
        self.ds.set_format("torch", columns=["pert_gene_emb", "labels"], output_all_columns=True)

    def __getitem__(self, idx) -> dict:
        return self.ds[idx]
    
    def __getitems__(self, keys: list) -> list:
        """Can be used to get a batch using a list of integers indices."""
        batch = self.ds.__getitem__(keys)
        selected_ctrl_indices = []
        if self.RANDOM_MAPPING_CONTROL_TO_CONTROL:
            for ct in batch["celltype"]:
                selected_ctrl_indices.append(
                    int(random.choice(self.ctrl_cell_indices[ct]))
                )
        else:
            for ct, pg, cell_idx in zip(batch["celltype"], batch["pert_gene"], keys):
                if pg != "non-targeting":
                    selected_ctrl_indices.append(
                        int(random.choice(self.ctrl_cell_indices[ct]))
                    )
                else:
                    selected_ctrl_indices.append(int(cell_idx))
        # MAIN_INPUT_NAME, pert_gene_emb, labels, pert_gene, celltype
        batch[self.MAIN_INPUT_NAME] = self.ds.__getitem__(selected_ctrl_indices)["labels"]

        return batch
    
    def __len__(self) -> int:
        return len(self.ds)

    @staticmethod
    def collate_fn(batch):
        return batch


# Fast
class HFDatasetForATAC(HFBaseDataset):
    MAIN_INPUT_NAME = "ctrl_cell_tf_idf"
    RANDOM_MAPPING_CONTROL_TO_CONTROL = False


# Fast
class HFDatasetForRNA(HFBaseDataset):
    MAIN_INPUT_NAME = "ctrl_cell_expr"
    RANDOM_MAPPING_CONTROL_TO_CONTROL = False
