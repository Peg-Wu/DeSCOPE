import os
import torch
import logging
import numpy as np
import polars as pl
import scanpy as sc
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import sklearn.metrics as skm

from enum import Enum
from tqdm.auto import tqdm
from anndata import AnnData
from scipy.stats import pearsonr
from wppkg import guess_is_lognorm
from typing import Union, Optional, Any, overload
from pdex import parallel_differential_expression
from cell_eval._evaluator import _build_pdex_kwargs
from cell_eval._types import PerturbationAnndataPair

logger = logging.getLogger(__name__)


class DuplicatedFeatureHandling(Enum):
    mean_pooling = "mean_pooling"
    max_pooling = "max_pooling"


class UniformFeatureForAnnData:
    r"""
    Align an AnnData object to a predefined set of feature names (e.g., genes, ccres etc.) by reindexing and zero-padding missing features.

    This class ensures that the input AnnData is transformed to have exactly the same feature space as a provided target list.
    Features present in the target but missing in the input are filled with `zeros`; features not in the target are dropped.
    Duplicate feature names in the input are resolved via mean- or max-pooling before alignment.

    **Example:**

    Suppose you have an AnnData with 3 cells and genes ['A', 'B', 'C']:

    >>> import scanpy as sc
    >>> import numpy as np
    >>> from descope.utils import UniformFeatureForAnnData, DuplicatedFeatureHandling

    >>> X = np.array([[1, 2, 3],
    ...               [4, 5, 6],
    ...               [7, 8, 9]])  # shape: (3, 3)
    >>> adata = sc.AnnData(X)
    >>> adata.var_names = ['A', 'B', 'C']

    >>> processor = UniformFeatureForAnnData(adata)
    >>> target_genes = ['B', 'C', 'D']  # Note: 'D' is not in original adata
    >>> aligned = processor(target_genes)

    >>> print(aligned.X)
    [[2. 3. 0.]
     [5. 6. 0.]
     [8. 9. 0.]]

    >>> print(aligned.var_names.tolist())
    ['B', 'C', 'D']

    Here, gene 'A' is dropped, 'B' and 'C' are kept in the new order, and 'D' (missing in input) is added as a zero column.
    """
    def __init__(
        self,
        input_h5ad: Union[str, AnnData],
        feature_names_col: Optional[str] = None,
        duplicated_features_handling: DuplicatedFeatureHandling = DuplicatedFeatureHandling.max_pooling,
    ):
        r"""
        **Parameters:**

        input_h5ad : str or AnnData
            | Input data, either a path to an H5AD file or an in-memory AnnData object to be aligned.
        feature_names_col : str or None, optional (default: None)
            | Column name in `adata.var` to use as feature identifiers (e.g., `"gene_symbol"`, `"cCRE_id"`).
            | If `None`, the existing `adata.var_names` are used directly.
        duplicated_features_handling : DuplicatedFeatureHandling, optional (default: DuplicatedFeatureHandling.max_pooling)
            | Strategy for resolving duplicated feature names:
            | - `DuplicatedFeatureHandling.mean_pooling`: replace duplicates with their mean expression across cells.
            | - `DuplicatedFeatureHandling.max_pooling`: retain the duplicate showing the highest average expression.
        """
        # Read the input adata
        if isinstance(input_h5ad, str):
            logger.info(f"Read AnnData from {input_h5ad} ...")
            self.adata = sc.read_h5ad(input_h5ad)
        else:
            self.adata = input_h5ad.copy()

        logger.info(f"Input adata shape: {self.adata.shape}")

        # Convert adata.X to dense matrix
        if hasattr(self.adata.X, "toarray"):
            logger.info("Converting adata.X to dense matrix ...")
            self.adata.X = self.adata.X.toarray()

        if feature_names_col is not None:
            logger.info(f"Set adata.var_names from adata.var['{feature_names_col}'].")
            self.adata.var_names = self.adata.var[feature_names_col].astype(str)
        else:
            logger.warning("Using adata.var_names as feature names.")
        
        # Preprocess duplicated features if needed
        self.duplicated_features_handling = duplicated_features_handling
        # Ensure float for mean_pooling
        if np.issubdtype(self.adata.X.dtype, np.integer) and self.duplicated_features_handling == DuplicatedFeatureHandling.mean_pooling:
            logger.warning("adata.X uses integer dtype — converting to float32 to prevent silent truncation in mean pooling.")
            self.adata.X = self.adata.X.astype(np.float32)
        self._preprocess_duplicated_features()

        logger.info("Input AnnData prepared.")

        # Placeholder for the output adata
        self.output_adata = None

    def _preprocess_duplicated_features(self):
        unique_features, counts = np.unique(self.adata.var_names, return_counts=True)
        if not all(counts == 1):
            duplicated_features = [unique_features[idx] for idx in np.where(counts > 1)[0]]
            logger.info(f"Found {len(duplicated_features)} duplicated features ({duplicated_features}).")

            duplicated_features_X = []
            for feature in duplicated_features:
                if self.duplicated_features_handling == DuplicatedFeatureHandling.mean_pooling:
                    # NOTE: Mean-pooling for each duplicated feature
                    feature_X = self.adata[:, self.adata.var_names == feature].X.mean(axis=1).reshape(-1, 1)
                elif self.duplicated_features_handling == DuplicatedFeatureHandling.max_pooling:
                    # NOTE: Max-pooling for each duplicated feature
                    feature_X = self.adata[:, self.adata.var_names == feature].X
                    max_index = np.argmax(feature_X.mean(axis=0))
                    feature_X = feature_X[:, max_index].reshape(-1, 1)
                else:
                    raise ValueError(
                        f"Unknown duplicated features handling method: {self.duplicated_features_handling}"
                    )
                duplicated_features_X.append(feature_X)
            self.adata.var_names_make_unique()  # feature, feature-1, feature-2, ...
            duplicated_features_indices = self.adata.var_names.get_indexer(duplicated_features)
            self.adata.X[:, duplicated_features_indices] = np.concatenate(duplicated_features_X, axis=1)
        else:
            logger.info("No duplicated features found.")
        
    def __call__(
        self,
        target_feature_names: list[str]
    ) -> AnnData:
        r"""
        Align the internal AnnData to the given target feature names by reindexing and zero-padding.

        Features in `target_feature_names` that are missing from the input AnnData are filled with zeros.
        The order of features in the output exactly matches `target_feature_names`.

        **Parameters:**

        target_feature_names : list[str]
            | List of feature names (e.g., gene symbols) defining the target feature space.
            | The output AnnData will have these as its `var_names`, in this exact order.

        **Returns:**

        AnnData
            | A new AnnData object with shape `(n_cells, len(target_feature_names))`.
            | Its `X` matrix is dense, and missing features are filled with zeros.
            | This h5ad file contains only X and var_names; all other attributes will be cleared.
        """
        logger.info("Unifying feature names ... (This will take some time.)")

        adata_origin_feature_names = self.adata.var_names.tolist()
        logger.info(f"Number of overlapping features: {len(set(target_feature_names) & set(adata_origin_feature_names))}")
        logger.info(f"Number of features to add (all values set to zero): {len(set(target_feature_names) - set(adata_origin_feature_names))}")

        target_feature_names_indices = self.adata.var_names.get_indexer(target_feature_names)
        valid = target_feature_names_indices != -1
        X = np.zeros((self.adata.n_obs, len(target_feature_names)), dtype=self.adata.X.dtype)
        X[:, valid] = self.adata.X[:, target_feature_names_indices[valid]]
        output_adata = AnnData(X=X)
        output_adata.var_names = target_feature_names

        self.output_adata = output_adata
        logger.info(f"Output AnnData has registered to `self.output_adata`, shape: {output_adata.shape}")

        return output_adata
    
    def save(self, save_path: str, convert_to_sparse: bool = False):
        r"""
        Save the aligned AnnData (`self.output_adata`) to an H5AD file.

        **Parameters:**

        save_path : str
            | File path where the output AnnData will be saved (in H5AD format).
        convert_to_sparse : bool, optional (default: False)
            | If `True`, converts the dense matrix `X` in `output_adata` to a sparse CSR matrix before saving.
            | This can significantly reduce file size when the data is large and sparse-friendly.
        """
        if convert_to_sparse:
            logger.info(f"Converting output AnnData to sparse matrix and saving to {save_path} ...")
            self.output_adata.X = sp.csr_matrix(self.output_adata.X)
        else:
            logger.info(f"Saving output AnnData to {save_path} ...")
        self.output_adata.write_h5ad(save_path)    


def intersect_adatas_for_celltype_transfer(
    adata_source: Union[str, AnnData],
    adata_target: Union[str, AnnData],
    pert_col: str = "gene",
    ctrl_name: str = "non-targeting",
    feature_names_col: Optional[str] = None,
    duplicated_features_handling: DuplicatedFeatureHandling = DuplicatedFeatureHandling.max_pooling,
    gene_embs_file: Optional[str] = None,
    save_dir: str = "./intersection-outdir"
):
    r"""
    Intersect two AnnData objects (source and target) on both perturbations and features to prepare them for cell type transfer.

    **Parameters:**

    adata_source : str or AnnData
        | Path to H5AD file or in-memory AnnData object representing the source adata.
    adata_target : str or AnnData
        | Path to H5AD file or in-memory AnnData object representing the target adata.
    pert_col : str, optional (default: `"gene"`)
        | Column name in `adata.obs` that indicates the perturbation or condition label.
    ctrl_name : str, optional (default: `"non-targeting"`)
        | Name used to denote control (unperturbed) cells. Must be present in both adatas.
    feature_names_col : str or None, optional (default: None)
        | Column name in `adata.var` to use as feature identifiers (e.g., `"gene_symbol"`).
        | If `None`, uses `adata.var_names` directly.
    duplicated_features_handling : DuplicatedFeatureHandling, optional (default: DuplicatedFeatureHandling.max_pooling)
        | Strategy to resolve duplicate feature names in either adata before alignment.
    gene_embs_file : str or None, optional (default: None)
        | Optional path to a PyTorch file containing gene embeddings (as a dict: `{gene: embedding}`).
        | If provided, only perturbations present in this file (plus `ctrl_name`) are retained.
    save_dir : str, optional (default: `"./intersection-outdir"`)
        | Directory where the processed AnnData files and metadata CSV will be saved.

    **Outputs:**

    The function saves the following to `save_dir`:
        | - `source.h5ad`: filtered and feature-aligned source AnnData.
        | - `target.h5ad`: filtered and feature-aligned target AnnData.
        | - `target_info.csv`: a table listing each perturbation in the target and its cell count.
    """
    if isinstance(adata_source, str):
        logger.info(f"Reading source adata from {adata_source}")
        adata_source = sc.read_h5ad(adata_source)
    else:
        adata_source = adata_source.copy()
    
    if isinstance(adata_target, str):
        logger.info(f"Reading target adata from {adata_target}")
        adata_target = sc.read_h5ad(adata_target)
    else:
        adata_target = adata_target.copy()

    # Intersect perturbations and features
    if feature_names_col is not None:
        source_features = adata_source.var[feature_names_col].unique().tolist()
        target_features = adata_target.var[feature_names_col].unique().tolist()
    else:
        source_features = adata_source.var_names.unique().tolist()
        target_features = adata_target.var_names.unique().tolist()
        
    source_perts = adata_source.obs[pert_col].unique().tolist()
    target_perts = adata_target.obs[pert_col].unique().tolist()

    if gene_embs_file is not None:
        gene_embs = torch.load(gene_embs_file, weights_only=False)
        perts_in_gene_embs = list(gene_embs.keys())
        perts_in_gene_embs.append(ctrl_name)
        perts_to_keep = list(set(source_perts) & set(target_perts) & set(perts_in_gene_embs))
    else:
        perts_to_keep = list(set(source_perts) & set(target_perts))
    
    features_to_keep = list(set(source_features) & set(target_features))

    logger.info(f"Number of overlapping perturbations: {len(perts_to_keep)}")
    logger.info(f"Number of overlapping features: {len(features_to_keep)}")
    
    assert ctrl_name in perts_to_keep, "The provided adata does not appear to contain control cells. Please check it."

    # Process source adata
    logger.info("------------------------------ Processing source adata ------------------------------")
    source_preprocessor = UniformFeatureForAnnData(
        input_h5ad=adata_source,
        feature_names_col=feature_names_col,
        duplicated_features_handling=duplicated_features_handling
    )
    processed_adata_source = source_preprocessor(target_feature_names=features_to_keep)
    processed_adata_source.obs[pert_col] = adata_source.obs[pert_col].values
    if feature_names_col is not None:
        processed_adata_source.var[feature_names_col] = processed_adata_source.var_names.values
    processed_adata_source = processed_adata_source[processed_adata_source.obs[pert_col].isin(perts_to_keep)]
    processed_adata_source = processed_adata_source[:, processed_adata_source.var_names.isin(features_to_keep)]
    logger.info("-------------------------------------------------------------------------------------")

    # Process target adata
    logger.info("------------------------------ Processing target adata ------------------------------")
    target_preprocessor = UniformFeatureForAnnData(
        input_h5ad=adata_target,
        feature_names_col=feature_names_col,
        duplicated_features_handling=duplicated_features_handling
    )
    processed_adata_target = target_preprocessor(target_feature_names=features_to_keep)
    processed_adata_target.obs[pert_col] = adata_target.obs[pert_col].values
    if feature_names_col is not None:
        processed_adata_target.var[feature_names_col] = processed_adata_target.var_names.values
    processed_adata_target = processed_adata_target[processed_adata_target.obs[pert_col].isin(perts_to_keep)]
    processed_adata_target = processed_adata_target[:, processed_adata_target.var_names.isin(features_to_keep)]
    logger.info("-------------------------------------------------------------------------------------")

    # Double Check
    assert (processed_adata_source.var_names == processed_adata_target.var_names).all()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    processed_adata_source.write_h5ad(os.path.join(save_dir, "source.h5ad"))
    processed_adata_target.write_h5ad(os.path.join(save_dir, "target.h5ad"))

    # Target Info CSV
    n_cells = []
    for g in perts_to_keep:
        n_cells.append(processed_adata_target[processed_adata_target.obs[pert_col] == g].shape[0])
    pd.DataFrame({"target_gene": perts_to_keep, "n_cells": n_cells}).to_csv(os.path.join(save_dir, "target_info.csv"), index=False)


def load_gene_names_engine(
    fp: Optional[str] = None
) -> Optional[list[str]]:
    r"""
    Load a list of gene names from a plain-text or CSV file.

    Supports two common formats:
    1. Single-column file (TXT/CSV): each line contains one gene name.
    2. Multi-column CSV with header: must include a column named `"target_gene"`; other columns (e.g., `"n_cells"`) are ignored.

    **Parameters:**

    fp : str or None, optional (default: None)
        | Path to the input file. If `None`, returns `None`.

    **Returns:**

    list[str] or None
        | A deduplicated list of gene names as strings. Returns `None` if `fp` is `None`.
    """
    if fp is None:
        return 
    
    # Read the file without assuming header first
    df = pd.read_csv(fp, header=None)

    if df.shape[1] == 1:
        # Case 1: single column → treat all entries as gene names
        gene_names = df.iloc[:, 0].dropna().astype(str).tolist()
    elif df.shape[1] >= 2:
        # Case 2: multiple columns → check if first row looks like a header
        first_row = df.iloc[0]
        if "target_gene" in first_row.values:
            # Re-read with header
            df = pd.read_csv(fp)
            gene_names = df["target_gene"].dropna().astype(str).tolist()
        else:
            # No header: assume first column is gene names
            gene_names = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        raise ValueError(f"Unsupported file format in {fp}: expected 1 or more columns, got {df.shape[1]}.")

    # Remove duplicates
    gene_names = list(set(gene_names))

    return gene_names


def load_gene_embs(
    gene_embs_file: str,
    perts_to_emb: Optional[list] = None  # All the genes you want to embed
) -> dict[str, torch.Tensor]:
    r"""
    Load gene embeddings from a PyTorch file and ensure coverage for a given set of perturbations.
    Handles single-gene and multi-gene perturbations. Missing embeddings are replaced by zero vectors
    for absent genes, and multi-gene perturbations are computed as the mean of all constituent genes
    (including zeros for missing genes).

    **Parameters:**

    gene_embs_file : str
        | Path to a `.pt` or `.pth` file containing a dictionary `{gene_name: embedding_tensor}`.
    perts_to_emb : list[str] or None, optional (default: None)
        | List of perturbation (gene) names that must be present in the output embedding dictionary.
        | Multi-gene perturbations must be denoted with '+' (e.g., `"GeneA+GeneB"`).
        | If `None`, simply returns all embeddings from the file without modification.

    **Returns:**

    dict[str, torch.Tensor]
        | A dictionary mapping gene names to their embedding tensors.
        | Guaranteed to include all entries in `perts_to_emb` (if not `None`).
    """
    # Load embeddings
    gene_embs = torch.load(gene_embs_file, weights_only=False)

    # Handle alias for TAZ
    if "TAFAZZIN" in gene_embs and "TAZ" not in gene_embs:
        gene_embs["TAZ"] = gene_embs["TAFAZZIN"]
    
    # Record origin genes in `gene_embs`
    origin_genes_in_gene_embs = set(gene_embs.keys())
    
    if perts_to_emb is None:
        return gene_embs

    zero_vector = torch.zeros_like(gene_embs[next(iter(gene_embs))])

    # Quickly count how many are single-gene perturbations and how many are multi-gene perturbations.
    single_pert, multiple_perts = 0, 0
    for pert in perts_to_emb:
        if "+" in pert:
            multiple_perts += 1
        else:
            single_pert += 1

    logger.info(
        f"Processing {len(perts_to_emb)} perturbations, including "
        f"{single_pert} single-gene perturbations and {multiple_perts} multi-gene perturbations.\n"
        f"Missing perturbations will be filled with zero vectors."
    )

    from collections import defaultdict
    missing_single_pert, missing_multiple_perts = [], defaultdict(list)
    for pert in perts_to_emb:
        if pert in origin_genes_in_gene_embs:
            continue

        if '+' in pert:  # multi-gene perturbation
            genes = [g.strip() for g in pert.split('+')]
            gene_vectors = []
            for g in genes:
                if g in origin_genes_in_gene_embs:
                    gene_vectors.append(gene_embs[g])
                else:
                    gene_vectors.append(zero_vector)
                    missing_multiple_perts[pert].append(g)
            gene_embs[pert] = torch.stack(gene_vectors, dim=0).mean(dim=0)  # mean pooling for multi-gene perturbation
        else:  # single-gene perturbation
            gene_embs[pert] = zero_vector
            missing_single_pert.append(pert)
    
    # logging
    multi_missing_info = [
        f"{pert} [missing: {', '.join(genes)}]" for pert, genes in missing_multiple_perts.items()
    ]

    logger.info(
        f"Single-gene perturbations missing from gene embeddings file ({len(missing_single_pert)}): "
        f"{', '.join(missing_single_pert)}.\n"
        f"Multi-gene perturbations with missing genes ({len(missing_multiple_perts)}): "
        f"{', '.join(multi_missing_info)}."
    )

    return gene_embs


def check_adata_format_consistent_with_epiagent(adata: AnnData):
    r"""
    Validate that the input AnnData conforms to the expected format for EpiAgent.

    Specifically checks:
    1. That the number of features (columns in `X`) equals the fixed cCRE count (1,355,445).
    2. That the feature matrix `X` contains continuous (floating-point) values, as required after TF-IDF transformation.

    **Parameters:**

    adata : AnnData
        | Input AnnData object to validate.
    """
    num_cCREs = 1355445

    # Step 1: Check the number of features in adata.X
    if adata.shape[1] != num_cCREs:
        raise ValueError(f"Feature dimensions are not {num_cCREs}. Please ensure you are using EpiAgent-required cCREs.")

    # Step 2: Check if the data is continuous (TF-IDF applied)
    if not np.issubdtype(adata.X.dtype, np.floating):
        raise ValueError("Feature values are not continuous. Please apply the global_TFIDF function before tokenization.")


def select_topk_ccres(
    adata: AnnData,
    topk_ccres: int = 50000
) -> AnnData:
    r"""
    Select the top-k cCREs (peaks) with highest total signal across all cells.

    The selection is based on the column-wise sum of `adata.X`. A stable sort is used to ensure reproducibility
    when ties occur in peak sums.

    **Parameters:**

    adata : AnnData
        | Input AnnData object containing cCRE-level features in `X`.
    topk_ccres : int, optional (default: 50000)
        | Number of top cCREs to retain.

    **Returns:**

    AnnData
        | A subset of the input `adata` containing only the top `topk_ccres` cCREs, in descending order of total signal.
    
    NOTE: This selection procedure is identical to that used in EpiAgent, with the addition of a stable sort to ensure that the top-k cCREs are always deterministically selected.
    """
    peak_sum = np.sum(adata.X, axis=0)
    peak_sum = np.array(peak_sum).reshape(-1)
    peak_sum_sortidx = np.argsort(peak_sum, kind="stable")[::-1]
    peak_sum_sortidx = peak_sum_sortidx[:topk_ccres]  # Ensure that the same top-k ccres are stably selected
    adata = adata[:, peak_sum_sortidx]
    return adata


def preprocess_atac_perturbation_adata_consistent_with_epiagent(
    adata: AnnData,
    topk_ccres: int = 50000,
    pert_col: str = "perturbation"
) -> AnnData:
    r"""
    Preprocess an ATAC-seq perturbation AnnData to be compatible with EpiAgent.

    This function performs three steps:
    1. Validates that the input matches EpiAgent's expected cCRE dimensionality and data type.
    2. Selects the top `topk_ccres` cCREs by total signal across cells.
    3. Removes cells with missing (NaN) values in the perturbation column.

    **Parameters:**

    adata : AnnData
        | Input ATAC-seq AnnData object, expected to contain ~1.36M cCREs and TF-IDF-transformed continuous values.
        | Ensure that the AnnData object meets the requirements of EpiAgent.
    topk_ccres : int, optional (default: 50000)
        | Number of most active cCREs to retain.
    pert_col : str, optional (default: `"perturbation"`)
        | Column name in `adata.obs` indicating the perturbation condition.

    **Returns:**

    AnnData
        | Preprocessed AnnData ready for use with EpiAgent: filtered to top cCREs and cleaned of NaN perturbations.
    """
    logger.info("------------------------- Preprocessing -------------------------")
    logger.info(f"AnnData shape before preprocessing: {adata.shape}")

    # Step1: Check adata format (consistent with epiagent)
    logger.info("Check whether the adata format is compatible with EpiAgent ...")
    check_adata_format_consistent_with_epiagent(adata)

    # Step2: Select topk ccres (consistent with epiagent)
    logger.info(f"Select top {topk_ccres} ccres ...")
    adata = select_topk_ccres(adata, topk_ccres)

    # Step 3: Remove cells with nan perturbations (consistent with epiagent)
    logger.info("Remove cells with nan perturbations ...")
    adata = adata[~adata.obs[pert_col].isna()]

    logger.info(f"AnnData shape after preprocessing: {adata.shape}")
    logger.info("------------------------------ Done ------------------------------")
    return adata


def preprocess_rna_perturbation_adata(
    adata: AnnData,
    target_sum: float = 1e4,
    pert_col: str = "target_gene",
    skip_raw_counts_check: bool = False
) -> AnnData:
    r"""
    Preprocess an RNA-seq perturbation AnnData for downstream analysis.

    This function:
    1. Removes cells with missing perturbation labels in `obs[pert_col]`.
    2. Applies library-size normalization (`normalize_total`) followed by `log1p`, 
    unless the data appears already log-normalized (detected heuristically).

    **Parameters:**

    adata : AnnData
        | Input RNA-seq AnnData object containing raw or normalized counts.
    target_sum : float, optional (default: 1e4)
        | Target total count per cell for library-size normalization.
    pert_col : str, optional (default: `"target_gene"`)
        | Column name in `adata.obs` indicating the perturbed gene or condition.
    skip_raw_counts_check : bool, optional (default: False)
        | If `True`, skips the heuristic check for log-normalization and **always applies**
        | library-size normalization (`sc.pp.normalize_total`) followed by `log1p`.
        | Use this only when you are certain the input contains **raw, unnormalized counts**.
        |
        | If `False` (default), the function checks whether the data appears to be already
        | log-normalized (e.g., via `log1p` of normalized counts). If so, normalization and
        | log-transformation are **skipped**; otherwise, they are applied.

    **Returns:**

    AnnData
        | Cleaned and normalized AnnData, with NaN-labeled cells removed and expression log-transformed.
    """
    logger.info("------------------------- Preprocessing -------------------------")
    logger.info(f"AnnData shape before preprocessing: {adata.shape}")

    # Step1: Remove cells with missing perturbation labels
    logger.info("Remove cells with missing perturbation labels ...")
    if adata.obs[pert_col].isnull().any():
        num_before = adata.n_obs
        adata = adata[adata.obs[pert_col].notnull()]
        num_after = adata.n_obs
        logger.info(f"Filtered out {num_before - num_after} cells with NaN in adata.obs '{pert_col}'.")
    else:
        logger.info("No missing perturbation labels found.")

    # Step2: Doing normalize_total and log1p
    logger.info(f"Performing normalize_total ({target_sum}) and log1p ...")
    if not skip_raw_counts_check:
        if guess_is_lognorm(adata):
            logger.warning(
                "AnnData appears to be log-normalized; skipping normalization and log1p."
            )
        else:
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)
    else:
        logger.warning("Skipping raw counts check, directly applying normalization and log1p ...")
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)

    logger.info(f"AnnData shape after preprocessing: {adata.shape}")
    logger.info("------------------------------ Done ------------------------------")
    return adata


def pearson_delta_on_topk_de(
    data: PerturbationAnndataPair,  # evaluator.anndata_pair
    de_real: str,
    topk: int = 20
) -> tuple[dict[str, float], float]:
    r"""
    Compute Pearson correlation between predicted and real perturbation effects on the top-k differentially expressed (DE) genes.
    Recommended to use in conjunction with `cell-eval`. (https://github.com/ArcInstitute/cell-eval)

    For each perturbation:
    1. Load its top-k DE genes from a precomputed CSV (`de_real`) ranked by `FDR`.
    2. Compute the bulk profile for the perturbation (mean expression across its cells) and for the control group.
    3. Derive the perturbation effect as: `Δ = bulk_perturbation - bulk_control`, separately for real and predicted data.
    4. Compute Pearson correlation between the real and predicted `Δ` vectors over the top-k DE genes.
    Returns per-perturbation correlations and their average.

    **Parameters:**

    data : PerturbationAnndataPair
        | `evaluator.anndata_pair`, see the example for details.
    de_real : str
        | DE CSV file generated by cell-eval from the real AnnData.
    topk : int, optional (default: 20)
        | Number of top DE genes (lowest FDR) to use for correlation per perturbation.

    **Returns:**

    tuple[dict[str, float], float]
        | - First element: a dictionary mapping perturbation names (`str`) to Pearson correlation coefficients (`float`).
        | - Second element: the mean of all per-perturbation correlations.

    **Example:**

    >>> from cell_eval import MetricsEvaluator

    >>> # Initialize evaluator
    >>> evaluator = MetricsEvaluator(...)

    >>> pearson_delta_on_topk_de, pearson_delta_on_topk_de_mean = pearson_delta_on_topk_de(
    ...     data=evaluator.anndata_pair,
    ...     de_real="de_results.csv",  # pre-computed DE results from cell-eval
    ...     topk=20
    ... )
    
    >>> print(f"Mean Pearson Delta on top-20 DE genes: {pearson_delta_on_topk_de_mean:.4f}")
    """
    de_real_df = pd.read_csv(de_real)
    topk_de_cache = {}
    for gp, df in de_real_df.groupby("target"):
        df = df.sort_values(by="fdr", ascending=True)
        topk_de = df["feature"][:topk].astype(str).tolist()
        topk_de_cache[gp] = topk_de

    res = {}
    for bulk_array in data.iter_bulk_arrays(embed_key=None):
        x = bulk_array.perturbation_effect(which="pred", abs=False)
        y = bulk_array.perturbation_effect(which="real", abs=False)

        de_genes = topk_de_cache[bulk_array.key]
        de_genes_indices = np.where(np.isin(data.genes, de_genes))[0]
        x_de, y_de = x[de_genes_indices], y[de_genes_indices]
        result = pearsonr(x_de, y_de)
        if isinstance(result, tuple):
            result = result[0]

        res[bulk_array.key] = float(result)

    return res, sum(res.values()) / len(res)


@overload
def direction_match_on_topk_de(
    data: PerturbationAnndataPair,
    de_real: str,
    topk: int = ...,
    separate_up_down_regulated: bool = False,
) -> tuple[dict[str, float], float]: ...


@overload
def direction_match_on_topk_de(
    data: PerturbationAnndataPair,
    de_real: str,
    topk: int = ...,
    separate_up_down_regulated: bool = True,
) -> tuple[dict[str, float], dict[str, float], float, float]: ...


def direction_match_on_topk_de(
    data: PerturbationAnndataPair,  # evaluator.anndata_pair
    de_real: str,
    topk: int = 100,
    separate_up_down_regulated: bool = True
) -> Union[tuple[dict[str, float], float], tuple[dict[str, float], dict[str, float], float, float]]:
    r"""
    Compute direction match accuracy (sign agreement) between predicted and real perturbation effects on top-k differentially expressed (DE) genes.
    Recommended to use in conjunction with `cell-eval`. (https://github.com/ArcInstitute/cell-eval)

    For each perturbation:
    1. Load its top-k DE genes from a precomputed CSV (`de_real`) ranked by `FDR`.
    2. Optionally split them into up-regulated (`percent_change > 0`) and down-regulated (`percent_change < 0`) sets.
    3. Compute perturbation effect as: `Δ = bulk_perturbation - bulk_control` for both real and predicted data.
    4. Measure the fraction of genes where the sign of `Δ_pred` matches the sign of `Δ_real`.

    Returns either: A single accuracy per perturbation (if `separate_up_down_regulated=False`), or
    Separate accuracies for up- and down-regulated genes (if `True`).

    **Parameters:**

    data : PerturbationAnndataPair
        | `evaluator.anndata_pair`, see the example for details.
    de_real : str
        | DE CSV file generated by `cell-eval` from the real AnnData.
    topk : int, optional (default: 100)
        | Number of top DE genes (lowest FDR) to consider in each direction (or overall if not separated).
    separate_up_down_regulated : bool, optional (default: True)
        | If `True`, compute direction accuracy separately for up- and down-regulated genes.
        | If `False`, compute a single accuracy over the top-k DE genes regardless of regulation direction.

    **Returns:**

    If `separate_up_down_regulated=False`:
        tuple[dict[str, float], float]
            | - Per-perturbation direction match accuracy (fraction of genes with matching sign).
            | - Mean accuracy across all perturbations.

    If `separate_up_down_regulated=True`:
        tuple[dict[str, float], dict[str, float], float, float]
            | - Per-perturbation accuracy on up-regulated genes.
            | - Per-perturbation accuracy on down-regulated genes.
            | - Mean accuracy on up-regulated genes.
            | - Mean accuracy on down-regulated genes.

    **Example:**

    >>> from cell_eval import MetricsEvaluator

    >>> # Initialize evaluator
    >>> evaluator = MetricsEvaluator(...)

    >>> # Using separate up/down evaluation (default)
    >>> up_acc, down_acc, mean_up, mean_down = direction_match_on_topk_de(
    ...     data=evaluator.anndata_pair,
    ...     de_real="de_results.csv",  # pre-computed DE results from cell-eval
    ...     topk=100,
    ...     separate_up_down_regulated=True
    ... )
    >>> print(f"Up-regulated direction accuracy: {mean_up:.4f}")
    >>> print(f"Down-regulated direction accuracy: {mean_down:.4f}")

    >>> # Or using combined evaluation
    >>> acc, mean_acc = direction_match_on_topk_de(
    ...     data=evaluator.anndata_pair,
    ...     de_real="de_results.csv",
    ...     topk=100,
    ...     separate_up_down_regulated=False
    ... )
    >>> print(f"Overall direction accuracy: {mean_acc:.4f}")
    """
    de_real_df = pd.read_csv(de_real)
    topk_de_cache = {}

    # Step 1: Build cache of top-K DE features per perturbation
    for gp, df in de_real_df.groupby("target"):
        df = df.sort_values(by="fdr", ascending=True)
        if separate_up_down_regulated:
            up = df[df["percent_change"] > 0]["feature"][:topk].astype(str).tolist()  # up regulated topk
            down = df[df["percent_change"] < 0]["feature"][:topk].astype(str).tolist()  # down regulated topk
            topk_de_cache[gp] = [up, down]
        else:
            topk_de_cache[gp] = df["feature"][:topk].astype(str).tolist()  # mix topk

    def _compute_direction_accuracy(
        x: np.ndarray,  # pred bulk array (perturbation effect)
        y: np.ndarray,  # real bulk array (perturbation effect)
        de_genes_list: list[str]
    ) -> float:
        de_genes_indices = np.where(np.isin(data.genes, de_genes_list))[0]
        x_de, y_de = x[de_genes_indices], y[de_genes_indices]
        results = np.mean(np.sign(x_de) == np.sign(y_de))  # same_sign_ratio
        return float(results)

    # Step 2: Compute direction accuracy per perturbation
    if not separate_up_down_regulated:
        res = {}
        for bulk_array in data.iter_bulk_arrays(embed_key=None):
            x = bulk_array.perturbation_effect(which="pred", abs=False)
            y = bulk_array.perturbation_effect(which="real", abs=False)
            de_genes_list = topk_de_cache[bulk_array.key]
            results = _compute_direction_accuracy(x, y, de_genes_list)
            res[bulk_array.key] = results
        return res, sum(res.values()) / len(res)
    else:
        up_res, down_res = {}, {}
        for bulk_array in data.iter_bulk_arrays(embed_key=None):
            x = bulk_array.perturbation_effect(which="pred", abs=False)
            y = bulk_array.perturbation_effect(which="real", abs=False)
            up_de_genes_list, down_de_genes_list = topk_de_cache[bulk_array.key]
            up_results = _compute_direction_accuracy(x, y, up_de_genes_list)
            down_results = _compute_direction_accuracy(x, y, down_de_genes_list)
            up_res[bulk_array.key], down_res[bulk_array.key] = up_results, down_results
        return up_res, down_res, sum(up_res.values()) / len(up_res), sum(down_res.values()) / len(down_res)


def pearson(
    data: PerturbationAnndataPair  # evaluator.anndata_pair
) -> tuple[dict[str, float], float]:
    r"""
    Compute Pearson correlation between predicted and real bulk expression profiles for each perturbation.
    Recommended to use in conjunction with `cell-eval`. (https://github.com/ArcInstitute/cell-eval)

    For each perturbation:
    1. Compute the bulk (mean) expression profile across all cells under that perturbation,
    separately for the predicted (`pert_pred`) and real (`pert_real`) adata.
    2. Calculate the Pearson correlation coefficient between these two bulk vectors.

    Returns per-perturbation correlations and their average.

    **Parameters:**

    data : PerturbationAnndataPair
        | `evaluator.anndata_pair`, see the example for details.

    **Returns:**

    tuple[dict[str, float], float]
        | - First element: a dictionary mapping perturbation names (`str`) to Pearson correlation coefficients (`float`).
        | - Second element: the mean of all per-perturbation correlations.

    **Example:**

    >>> from cell_eval import MetricsEvaluator

    >>> # Initialize evaluator
    >>> evaluator = MetricsEvaluator(...)

    >>> pert_corr, mean_corr = pearson(data=evaluator.anndata_pair)
    >>> print(f"Mean Pearson correlation across perturbations: {mean_corr:.4f}")
    """
    res = {}
    for bulk_array in data.iter_bulk_arrays(embed_key=None):
        x = bulk_array.pert_pred
        y = bulk_array.pert_real

        result = pearsonr(x, y)
        if isinstance(result, tuple):
            result = result[0]
        
        res[bulk_array.key] = float(result)

    return res, sum(res.values()) / len(res)
    

def edistance(
    adata: Union[str, AnnData],
    control_pert: str = "non-targeting",
    pert_col: str = "target_gene",
    metric: str = "euclidean",
    **kwargs,
) -> dict[str, float]:
    r"""
    Compute energy distance (E-distance) between each perturbation and the control group.

    The E-distance is defined as: `2 * E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]`,  
    where `X` is the perturbation, `Y` is the control, and `X'`, `Y'` are independent copies.  
    This implementation reuses the precomputed within-control distance (`sigma_y`) for efficiency.

    **Parameters:**

    adata : str or AnnData
        | Path to H5AD file or in-memory AnnData object.
    control_pert : str, optional (default: `"non-targeting"`)
        | Value in `adata.obs[pert_col]` that identifies the control group.
    pert_col : str, optional (default: `"target_gene"`)
        | Column name in `adata.obs` indicating the perturbation condition.
    metric : str, optional (default: `"euclidean"`)
        | Distance metric passed to `sklearn.metrics.pairwise_distances`.
        | Supported values include `"euclidean"`, `"cosine"`, etc.
    **kwargs : dict
        | Additional keyword arguments passed to `pairwise_distances`.

    **Returns:**

    dict[str, float]
        | Mapping from perturbation name (`str`) to its E-distance from the control group (`float`).
    """
    if isinstance(adata, str):
        logger.info(f"Reading anndata from {adata}")
        adata = ad.read_h5ad(adata)

    if sp.issparse(adata.X):
        logger.info("Converting sparse matrix to dense array")
        adata.X = adata.X.toarray()

    def _edistance(
        x: np.ndarray,
        y: np.ndarray,
        metric: str = "euclidean",
        precomp_sigma_y: Optional[float] = None,
        **kwargs,
    ) -> float:
        sigma_x = skm.pairwise_distances(x, metric=metric, **kwargs).mean()
        sigma_y = (
            precomp_sigma_y
            if precomp_sigma_y is not None
            else skm.pairwise_distances(y, metric=metric, **kwargs).mean()
        )
        delta = skm.pairwise_distances(x, y, metric=metric, **kwargs).mean()
        return float(2 * delta - sigma_x - sigma_y)

    # Precompute sigma for control data (reused by all perturbations)
    logger.info("Precomputing sigma for control data")
    ctrl_matrix = adata[adata.obs[pert_col] == control_pert].X
    precomp_sigma = skm.pairwise_distances(
        ctrl_matrix, metric=metric, **kwargs
    ).mean()

    perts = adata.obs[pert_col].unique().tolist()
    perts.remove(control_pert)
    results = {}
    logger.info("Computing edistance for each perturbation")
    for p in tqdm(perts):
        pert_matrix = adata[adata.obs[pert_col] == p].X
        results[p] = _edistance(
            pert_matrix,
            ctrl_matrix,
            precomp_sigma_y=precomp_sigma,
            metric=metric,
            **kwargs,
        )

    return results


def build_de(
    adata: Union[str, AnnData],
    control_pert: str = "non-targeting",
    pert_col: str = "target_gene",
    de_method: str = "wilcoxon",
    num_threads: int = 1,
    batch_size: int = 100,
    outdir: Optional[str] = None,
    allow_discrete: bool = False,
    pdex_kwargs: Optional[dict[str, Any]] = None,
    check_log_norm: bool = True
) -> pl.DataFrame:
    r"""
    Compute differential expression (DE) for all perturbations against a control group using scalable parallelized backend.
    """
    if isinstance(adata, str):
        logger.info(f"Reading real anndata from {adata}")
        adata = ad.read_h5ad(adata)
    
    if check_log_norm:
        assert guess_is_lognorm(adata), (
            "AnnData appears to be not log-normalized. "
            "Please run normalization and log1p transformation before proceeding."
        )

    logger.info("Computing DE")
    pdex_kwargs = _build_pdex_kwargs(
        reference=control_pert,
        groupby_key=pert_col,
        num_workers=num_threads,
        metric=de_method,
        batch_size=batch_size,
        allow_discrete=allow_discrete,
        pdex_kwargs=pdex_kwargs or {},
    )

    logger.info(f"Using the following pdex kwargs: {pdex_kwargs}")
    frame = parallel_differential_expression(adata=adata, **pdex_kwargs)

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        full_path = os.path.join(outdir, "de.csv")
        logger.info(f"Writing DE results to: {full_path}")
        frame.write_csv(full_path)
        
    return frame