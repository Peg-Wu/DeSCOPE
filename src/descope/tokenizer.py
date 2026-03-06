import os
import logging
import scanpy as sc
from tqdm.auto import tqdm
from typing import Union, Optional
from collections import defaultdict
from datasets import Dataset, concatenate_datasets
from .utils import (
    DuplicatedFeatureHandling,
    UniformFeatureForAnnData,
    check_adata_format_consistent_with_epiagent, 
    preprocess_atac_perturbation_adata_consistent_with_epiagent,
    preprocess_rna_perturbation_adata
)

logger = logging.getLogger(__name__)


def _filter_perturbations(
    adata: sc.AnnData,
    pert_col: str = "perturbation",
    ctrl_name: str = "control",
    perts_to_include: Optional[list[str]] = None,
    perts_to_exclude: Optional[list[str]] = None
) -> sc.AnnData:
    """Filter an AnnData object to include or exclude specific perturbations."""
    logger.info("Filtering perturbations ...")
    if perts_to_include is not None and perts_to_exclude is not None:
        raise ValueError("`perts_to_include` and `perts_to_exclude` cannot be specified at the same time.")
    elif perts_to_include is None and perts_to_exclude is None:
        logger.info("No perturbation filtering specified. Using all perturbations.")
    elif perts_to_include is not None and perts_to_exclude is None:
        if ctrl_name not in perts_to_include:
            perts_to_include.append(ctrl_name)
        logger.info(f"Filtering perturbations to include: {perts_to_include} ...")
        adata = adata[adata.obs[pert_col].isin(perts_to_include)]
    elif perts_to_exclude is not None and perts_to_include is None:
        if ctrl_name in perts_to_exclude:
            perts_to_exclude.remove(ctrl_name)
        logger.info(f"Filtering perturbations to exclude: {perts_to_exclude} ...")
        adata = adata[~adata.obs[pert_col].isin(perts_to_exclude)]

    # Double Check control cells are included
    assert ctrl_name in adata.obs[pert_col].unique().tolist()
    return adata


def _merge_list_of_dicts(list_of_dicts: list[dict]) -> dict:
    merged_dic = defaultdict(list)
    for dic in list_of_dicts:
        for k, v in dic.items():
            merged_dic[k].append(v)
    return merged_dic


def tokenize_adata_to_hf_dataset(
    adata: sc.AnnData,
    cell_line_name: str,
    pert_col: str = "gene",
    chunk_size: int = 20000
) -> Dataset:
    r"""
    Convert an AnnData object into a Hugging Face Dataset for downstream modeling.

    This function processes single-cell gene expression data stored in an AnnData object
    by extracting expression vectors and associated perturbation labels, then packages
    them into a Hugging Face `Dataset`. To manage memory usage for large datasets,
    the conversion is performed in chunks.

    **Parameters**

    adata : sc.AnnData
        | Single-cell dataset in AnnData format containing gene expression matrix (in `.X`)
        | and perturbation annotations (in `.obs[pert_col]`).
    cell_line_name : str
        | Name of the cell line or cell type associated with all cells in the AnnData object.
        | This will be added as a constant metadata field (`"celltype"`) in the output dataset.
    pert_col : str, default="gene"
        | Column name in `adata.obs` that contains the perturbation labels (e.g., gene names)
    chunk_size : int, default=20000
        | Number of cells to process per chunk during dataset construction.
        | Helps reduce memory overhead when handling large AnnData objects.

    **Returns**

    ds : datasets.Dataset
        | A Hugging Face Dataset with the following columns:
        | - `"labels"`: Gene expression vector for each cell (as a list of floats).
        | - `"pert_gene"`: Perturbation label (e.g., gene name or "control").
        | - `"celltype"`: Constant string indicating the cell line name provided as input.

    **Notes**

    - If `adata.X` is sparse (e.g., scipy sparse matrix), it is converted to a dense array
      using `.toarray()` before processing.
    - The resulting dataset is suitable for use with Hugging Face Transformers or other
      deep learning pipelines that expect dictionary-like batched inputs.
    """
    cell_dicts: list[dict] = [
        {"labels": labels, "pert_gene": pert_gene, "celltype": cell_line_name} 
        for labels, pert_gene in zip(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
            adata.obs[pert_col].tolist()
        )
    ]
    all_chunks: list[Dataset] = []
    for i in tqdm(range(0, len(cell_dicts), chunk_size), desc="Split Dataset Into Chunks"):
        chunk: list[dict] = cell_dicts[i:i+chunk_size]
        chunk: dict = _merge_list_of_dicts(chunk)
        chunk_ds = Dataset.from_dict(chunk)
        all_chunks.append(chunk_ds)
    ds = concatenate_datasets(all_chunks)

    return ds


def tokenize_adata_to_hf_dataset_for_atac(
    adata: Union[str, sc.AnnData],
    cell_line_name: str,
    perts_to_include: list[str] = None,
    perts_to_exclude: list[str] = None,
    topk_ccres: int = 50000,
    pert_col: str = "perturbation",
    ctrl_name: str = "control",
    save_dir: str = "./tokenized_dataset",
    chunk_size: int = 20000
) -> Dataset:
    r"""
    Preprocess and tokenize ATAC-seq perturbation data into a Hugging Face Dataset.

    **Parameters**

    adata : str or sc.AnnData
        | Input AnnData (or path to .h5ad) with ATAC profiles and perturbation labels.
    cell_line_name : str
        | Cell line name; saved as `"celltype"` in the dataset.
    perts_to_include / perts_to_exclude : list of str, optional
        | Mutually exclusive filters for perturbations. Only one may be specified.
        | If both are None, all perturbations are retained.
        | The control condition (ctrl_name) is always preserved regardless of the filter.
    topk_ccres : int, default=50000
        | Number of top variable cCREs to retain.
    pert_col : str, default="perturbation"
        | Column in `adata.obs` storing perturbation names.
    ctrl_name : str, default="control"
        | Label for control cells.
    save_dir : str, default="./tokenized_dataset"
        | Output directory for the saved dataset.
    chunk_size : int, default=20000
        | Number of cells to process per chunk during dataset construction.
        | Helps reduce memory overhead when handling large AnnData objects.

    **Returns**

    datasets.Dataset
        | Tokenized dataset saved to `save_dir`, with columns: `"labels"`, `"pert_gene"`, `"celltype"`.
    """
    if isinstance(adata, str):
        logger.info(f"Reading adata from {adata} ...")
        adata = sc.read_h5ad(adata)
    else:
        adata = adata.copy()
    
    preprocessed_adata = preprocess_atac_perturbation_adata_consistent_with_epiagent(
        adata=adata,
        topk_ccres=topk_ccres,
        pert_col=pert_col
    )

    preprocessed_adata = _filter_perturbations(
        adata=preprocessed_adata,
        pert_col=pert_col,
        ctrl_name=ctrl_name,
        perts_to_include=perts_to_include,
        perts_to_exclude=perts_to_exclude,
    )

    logger.info("Tokenizing adata to huggingface dataset ...")
    tokenized_dataset = tokenize_adata_to_hf_dataset(
        adata=preprocessed_adata,
        cell_line_name=cell_line_name,
        pert_col=pert_col,
        chunk_size=chunk_size
    )

    tokenized_dataset.save_to_disk(save_dir)
    return tokenized_dataset


def tokenize_adata_to_hf_dataset_for_rna(
    adata: Union[str, sc.AnnData],
    cell_line_name: str,
    perts_to_include: list[str] = None,
    perts_to_exclude: list[str] = None,
    target_sum: float = 1e4,
    pert_col: str = "gene",
    ctrl_name: str = "non-targeting",
    skip_raw_counts_check: bool = False,
    save_dir: str = "./tokenized_dataset",
    chunk_size: int = 20000
) -> Dataset:
    r"""
    Preprocess and tokenize scRNA-seq perturbation data into a Hugging Face Dataset.

    **Parameters**

    adata : str or sc.AnnData
        | Input AnnData (or path to .h5ad) with raw gene counts and perturbation labels.
    cell_line_name : str
        | Cell line name; stored as `"celltype"` in the output dataset.
    perts_to_include / perts_to_exclude : list of str, optional
        | Mutually exclusive filters for perturbations. Only one may be specified.
        | If both are None, all perturbations are retained.
        | The control condition (ctrl_name) is always preserved regardless of the filter.
    target_sum : float, default=1e4
        | Total count per cell after normalization (CPM-like scaling).
    pert_col : str, default="gene"
        | Column in `adata.obs` containing perturbation identifiers.
    ctrl_name : str, default="non-targeting"
        | Label for control cells.
    skip_raw_counts_check : bool, default=False
        | Skip assertion that input counts are integers (use only if data is pre-validated).
    save_dir : str, default="./tokenized_dataset"
        | Directory to save the resulting dataset.
    chunk_size : int, default=20000
        | Number of cells to process per chunk during dataset construction.
        | Helps reduce memory overhead when handling large AnnData objects.

    **Returns**

    datasets.Dataset
        | Tokenized dataset saved to `save_dir`, with columns: `"labels"`, `"pert_gene"`, `"celltype"`.
    """
    if isinstance(adata, str):
        logger.info(f"Reading adata from {adata} ...")
        adata = sc.read_h5ad(adata)
    else:
        adata = adata.copy()
    
    preprocessed_adata = preprocess_rna_perturbation_adata(
        adata=adata,
        target_sum=target_sum,
        pert_col=pert_col,
        skip_raw_counts_check=skip_raw_counts_check
    )

    preprocessed_adata = _filter_perturbations(
        adata=preprocessed_adata,
        pert_col=pert_col,
        ctrl_name=ctrl_name,
        perts_to_include=perts_to_include,
        perts_to_exclude=perts_to_exclude,
    )

    logger.info("Tokenizing adata to huggingface dataset ...")
    tokenized_dataset = tokenize_adata_to_hf_dataset(
        adata=preprocessed_adata,
        cell_line_name=cell_line_name,
        pert_col=pert_col,
        chunk_size=chunk_size
    )

    tokenized_dataset.save_to_disk(save_dir)
    return tokenized_dataset
    

class TokenizerForATAC:
    r"""
    A tokenizer class specifically designed for ATAC-seq data to preprocess and tokenize datasets for pretraining.

    **Example:**

    >>> from descope.tokenizer import TokenizerForATAC

    >>> tokenizer = TokenizerForATAC(
    ...     cell_line_ft="path/to/finetune_data.h5ad",
    ...     topk_ccres=50000,
    ...     pert_col="perturbation"
    ... )

    >>> tokenizer.tokenize(
    ...     cell_line_pt=["path/to/pretrain_data1.h5ad", "path/to/pretrain_data2.h5ad"],
    ...     cell_line_name=["cell_line_pretrain1", "cell_line_pretrain2"],
    ...     pert_col=["perturbation1", "perturbation2"],
    ...     save_dir="./tokenized_dataset",
    ...     apply_pert_gene_filter=False,
    ...     chunk_size=20000
    ... )
    """
    def __init__(
        self,
        cell_line_ft: Union[str, sc.AnnData],
        topk_ccres: int = 50000,
        pert_col: str = "perturbation"
    ):
        if isinstance(cell_line_ft, str):
            logger.info(f"Reading cell line (finetune) from {cell_line_ft} ...")
            cell_line_ft = sc.read_h5ad(cell_line_ft)
        else:
            cell_line_ft = cell_line_ft.copy()

        self.topk_ccres = topk_ccres
        self.pert_col = pert_col

        # Other cell lines used for pretraining should retain the same topk-ccres and perturbation genes as cell line (finetune).
        self.topk_ccres_list, self.pert_genes_list = self._get_inputs_and_perts_from_adata_ft(cell_line_ft)

    def _get_inputs_and_perts_from_adata_ft(self, cell_line_ft: sc.AnnData) -> tuple[list[str], list[str]]:
        logger.info("Getting unified input topk-ccres and perturbation genes from cell line (finetune) ...")
        logger.info("Other cell lines used for pretraining should retain the same topk-ccres and perturbation genes as cell line (finetune).")
        cell_line_ft_processed = preprocess_atac_perturbation_adata_consistent_with_epiagent(
            adata=cell_line_ft, 
            topk_ccres=self.topk_ccres, 
            pert_col=self.pert_col
        )
        topk_ccres_list = cell_line_ft_processed.var_names.tolist()
        pert_genes_list = cell_line_ft_processed.obs[self.pert_col].unique().tolist()

        return topk_ccres_list, pert_genes_list
    
    def _tokenize(
        self,
        cell_line_pt: Union[str, sc.AnnData],
        cell_line_name: str,
        pert_col: Optional[str] = None,
        save_dir: str = "./tokenized_datasets",
        apply_pert_gene_filter: bool = True,
        chunk_size: int = 20000
    ):
        if isinstance(cell_line_pt, str):
            logger.info(f"Reading cell line (pretrain) from {cell_line_pt}, cell line name: {cell_line_name} ...")
            cell_line_pt = sc.read_h5ad(cell_line_pt)
        else:
            cell_line_pt = cell_line_pt.copy()

        pert_col = self.pert_col if pert_col is None else pert_col
        
        check_adata_format_consistent_with_epiagent(cell_line_pt)
        logger.info("Filtering cell line (pretrain) to keep the same topk ccres and perturbation genes as cell line (finetune) ...")
        cell_line_pt = cell_line_pt[:, self.topk_ccres_list]  # keep the same topk ccres as cell_line_ft
        if apply_pert_gene_filter:
            cell_line_pt = cell_line_pt[cell_line_pt.obs[pert_col].isin(self.pert_genes_list)]  # keep the same perturbation genes as cell_line_ft
        else:
            logger.info("apply_pert_gene_filter is False, so we will keep all perturbation genes in the cell line (pretrain).")
            cell_line_pt = cell_line_pt[~cell_line_pt.obs[pert_col].isna()]  # filter out cells with nan perturbation

        # Tokenize to huggingface dataset
        logger.info("Tokenizing cell line (pretrain) to huggingface dataset ...")
        ds = tokenize_adata_to_hf_dataset(cell_line_pt, cell_line_name, pert_col, chunk_size)
        ds.save_to_disk(os.path.join(save_dir, cell_line_name))

    def tokenize(
        self,
        cell_line_pt: list[Union[str, sc.AnnData]],
        cell_line_name: list[str],
        pert_col: Optional[list[str]] = None,
        save_dir: str = "./tokenized_datasets",
        apply_pert_gene_filter: bool = True,
        chunk_size: int = 20000
    ):
        r"""Tokenizes multiple pretraining ATAC-seq datasets into Hugging Face Datasets format.

        **Parameters:**

        cell_line_pt : list[Union[str, sc.AnnData]]
            | A list of paths to `.h5ad` files or AnnData objects representing pretraining cell lines.
        cell_line_name : list[str]
            | A list of names corresponding to each pretraining dataset; used as subdirectory names when saving tokenized data.
        pert_col : list[str] or None, optional (default: None)
            | A list of column names in the `.obs` attribute of each AnnData object indicating the perturbation labels.
            | If None, the `pert_col` specified during tokenizer initialization is used for all datasets.
        save_dir : str, optional (default: "./tokenized_datasets")
            | Directory path where the tokenized datasets will be saved.
        apply_pert_gene_filter : bool, optional (default: True)
            | Whether to filter out cells with perturbations not present in the finetune dataset.
            | If True, only cells with perturbations in `self.pert_genes_list` are retained.
        chunk_size : int, default=20000
            | Number of cells to process per chunk during dataset construction.
            | Helps reduce memory overhead when handling large AnnData objects.
        """
        pert_col = [self.pert_col] * len(cell_line_pt) if pert_col is None else pert_col
        assert len(cell_line_pt) == len(cell_line_name) == len(pert_col), "The length of cell_line_pt, cell_line_name, pert_col should be the same."
        
        pbar = tqdm(range(len(cell_line_pt)), desc="Tokenization")
        for cell_line_pt_i, cell_line_name_i, pert_col_i in zip(cell_line_pt, cell_line_name, pert_col):
            self._tokenize(
                cell_line_pt=cell_line_pt_i,
                cell_line_name=cell_line_name_i,
                pert_col=pert_col_i,
                save_dir=save_dir,
                apply_pert_gene_filter=apply_pert_gene_filter,
                chunk_size=chunk_size
            )
            pbar.update(1)


class TokenizerForRNA:
    r"""
    A tokenizer class specifically designed for RNA-seq (scRNA-seq) perturbation data to preprocess and tokenize datasets for pretraining.

    **Example:**
    
    >>> from descope.tokenizer import TokenizerForRNA
    >>> from descope.utils import DuplicatedFeatureHandling

    >>> tokenizer = TokenizerForRNA(
    ...     cell_line_ft="path/to/finetune_data.h5ad",
    ...     target_sum=1e4,
    ...     pert_col="gene",
    ...     gene_names_col="gene_symbols"
    ... )

    >>> tokenizer.tokenize(
    ...     cell_line_pt=["path/to/pretrain_data1.h5ad", "path/to/pretrain_data2.h5ad"],
    ...     cell_line_name=["cell_line_pretrain1", "cell_line_pretrain2"],
    ...     pert_col=["gene", "gene"],
    ...     gene_names_col=["gene_name", "gene_name"],
    ...     save_dir="./tokenized_dataset",
    ...     apply_pert_gene_filter=False,
    ...     duplicated_features_handling=DuplicatedFeatureHandling.mean_pooling,
    ...     skip_raw_counts_check=True,
    ...     chunk_size=20000
    ... )
    """
    def __init__(
        self,
        cell_line_ft: Union[str, sc.AnnData],
        target_sum: float = 1e4,
        pert_col: str = "gene",
        gene_names_col: Optional[str] = None
    ):
        if isinstance(cell_line_ft, str):
            logger.info(f"Reading cell line (finetune) from {cell_line_ft} ...")
            cell_line_ft = sc.read_h5ad(cell_line_ft)
        else:
            cell_line_ft = cell_line_ft.copy()

        self.target_sum = target_sum
        self.pert_col = pert_col

        if gene_names_col is not None:
            logger.info(f"Set cell_line_ft.var_names from cell_line_ft.var['{gene_names_col}'].")
            cell_line_ft.var_names = cell_line_ft.var[gene_names_col].astype(str)
        else:
            logger.warning("Using cell_line_ft.var_names as gene names.")

        # Other cell lines used for pretraining should retain the same input genes and perturbation genes as cell line (finetune).
        self.input_genes_list, self.pert_genes_list = self._get_inputs_and_perts_from_adata_ft(cell_line_ft)
    
    def _get_inputs_and_perts_from_adata_ft(self, cell_line_ft: sc.AnnData) -> tuple[list[str], list[str]]:
        logger.info("Getting unified input genes and perturbation genes from cell line (finetune) ...")
        logger.info("Other cell lines used for pretraining should retain the same input genes and perturbation genes as cell line (finetune).")
        cell_line_ft = cell_line_ft[cell_line_ft.obs[self.pert_col].notnull()]
        input_genes_list = cell_line_ft.var_names.tolist()
        pert_genes_list = cell_line_ft.obs[self.pert_col].unique().tolist()

        return input_genes_list, pert_genes_list
    
    def _tokenize(
        self,
        cell_line_pt: Union[str, sc.AnnData],
        cell_line_name: str,
        pert_col: Optional[str] = None,
        gene_names_col: Optional[str] = None,
        save_dir: str = "./tokenized_datasets",
        apply_pert_gene_filter: bool = True,
        duplicated_features_handling: DuplicatedFeatureHandling = DuplicatedFeatureHandling.max_pooling,
        skip_raw_counts_check: bool = False,
        chunk_size: int = 20000
    ):
        # Cache perturbation genes in cell line (pretrain)
        logger.info(f"Tokenizing cell line {cell_line_name} ...")
        pert_col = self.pert_col if pert_col is None else pert_col
        if isinstance(cell_line_pt, str):
            pert_genes = sc.read_h5ad(cell_line_pt, backed="r").obs[pert_col].values
        else:
            pert_genes = cell_line_pt.obs[pert_col].values
        
        uniform_toolkit = UniformFeatureForAnnData(
            input_h5ad=cell_line_pt,
            feature_names_col=gene_names_col,
            duplicated_features_handling=duplicated_features_handling,
        )
        cell_line_pt = uniform_toolkit(target_feature_names=self.input_genes_list)  # keep the same inputgenes as cell_line_ft
        cell_line_pt.obs[pert_col] = pert_genes
        
        # normalize_total and log1p
        cell_line_pt = preprocess_rna_perturbation_adata(
            adata=cell_line_pt,
            target_sum=self.target_sum,
            pert_col=pert_col,
            skip_raw_counts_check=skip_raw_counts_check
        )

        logger.info("Filtering cell line (pretrain) to keep the same input_genes and perturbation genes as cell line (finetune) ...")
        if apply_pert_gene_filter:
            cell_line_pt = cell_line_pt[cell_line_pt.obs[pert_col].isin(self.pert_genes_list)]  # keep the same perturbation genes as cell_line_ft
        else:
            logger.info("apply_pert_gene_filter is False, so we will keep all perturbation genes in the cell line (pretrain).")

        # Tokenize to huggingface dataset
        logger.info("Tokenizing cell line (pretrain) to huggingface dataset ...")
        ds = tokenize_adata_to_hf_dataset(cell_line_pt, cell_line_name, pert_col, chunk_size)
        ds.save_to_disk(os.path.join(save_dir, cell_line_name))
    
    def tokenize(
        self,
        cell_line_pt: list[Union[str, sc.AnnData]],
        cell_line_name: list[str],
        pert_col: Optional[list[str]] = None,
        gene_names_col: Optional[list[str]] = None,
        save_dir: str = "./tokenized_datasets",
        apply_pert_gene_filter: bool = True,
        duplicated_features_handling: DuplicatedFeatureHandling = DuplicatedFeatureHandling.max_pooling,
        skip_raw_counts_check: bool = False,
        chunk_size: int = 20000
    ):
        r"""Tokenizes multiple pretraining scRNA-seq datasets into Hugging Face Datasets format.

        **Parameters:**

        cell_line_pt : list[Union[str, sc.AnnData]]
            | A list of paths to `.h5ad` files or AnnData objects representing pretraining cell lines.
        cell_line_name : list[str]
            | A list of names corresponding to each pretraining dataset; used as subdirectory names when saving tokenized data.
        pert_col : list[str] or None, optional (default: None)
            | A list of column names in the `.obs` attribute of each AnnData object indicating the perturbation labels.
            | If None, the `pert_col` specified during tokenizer initialization is used for all datasets.
        gene_names_col : list[str] or None, optional (default: None)
            | A list of column names in the `.var` attribute of each AnnData object that contain gene symbols.
            | Used to standardize gene names before alignment. If None, assumes `var_names` are already gene symbols.
        save_dir : str, optional (default: "./tokenized_datasets")
            | Directory path where the tokenized datasets will be saved.
        apply_pert_gene_filter : bool, optional (default: True)
            | Whether to filter out cells with perturbations not present in the finetune dataset.
            | If True, only cells with perturbations in `self.pert_genes_list` are retained.
        duplicated_features_handling : DuplicatedFeatureHandling, optional (default: max_pooling)
            | Strategy for handling duplicated gene names.
        skip_raw_counts_check : bool, optional (default: False)
            | Whether to skip the raw counts check during preprocessing.
            | If you are sure that your data is raw counts, you can set this to True to skip the check.
        chunk_size : int, default=20000
            | Number of cells to process per chunk during dataset construction.
            | Helps reduce memory overhead when handling large AnnData objects.
        """
        pert_col = [self.pert_col] * len(cell_line_pt) if pert_col is None else pert_col
        gene_names_col = [None] * len(cell_line_pt) if gene_names_col is None else gene_names_col

        assert (
            len(cell_line_pt) == len(cell_line_name) == len(pert_col) == len(gene_names_col)
        ), "The length of cell_line_pt, cell_line_name, pert_col, and gene_names_col should be the same."

        pbar = tqdm(range(len(cell_line_pt)), desc="Tokenization")
        for cell_line_pt_i, cell_line_name_i, pert_col_i, gene_names_col_i in zip(cell_line_pt, cell_line_name, pert_col, gene_names_col):
            self._tokenize(
                cell_line_pt=cell_line_pt_i,
                cell_line_name=cell_line_name_i,
                pert_col=pert_col_i,
                gene_names_col=gene_names_col_i,
                save_dir=save_dir,
                apply_pert_gene_filter=apply_pert_gene_filter,
                duplicated_features_handling=duplicated_features_handling,
                skip_raw_counts_check=skip_raw_counts_check,
                chunk_size=chunk_size
            )
            pbar.update(1)
