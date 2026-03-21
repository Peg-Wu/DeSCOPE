import logging
import numpy as np
import pandas as pd
from enum import Enum
from tqdm.auto import tqdm
from anndata import AnnData
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class MeanBaselineType(Enum):
    PERTURBED_MEAN = "perturbed_mean"
    MATCHING_MEAN = "matching_mean"


def mean_baseline(
    train_adata: AnnData,
    test_perts: list[str],
    pert_col: str = "target_gene",
    ctrl_name: str = "non-targeting",
    feature_names_col: Optional[str] = None,
    mean_type: MeanBaselineType = MeanBaselineType.PERTURBED_MEAN,
    preprocess_adata_func: Optional[Callable[..., AnnData]] = None,
    **preprocess_adata_func_kwargs
) -> AnnData:
    r"""perturbed_mean & matching_mean baseline for perturbation response prediction: https://www.nature.com/articles/s41587-025-02777-8"""
    # Preprocess adata if `preprocess_adata_func` is provided
    if preprocess_adata_func is None:
        logger.warning("Preprocessing function missing. Using raw data for mean calculation.")
        preprocessed_adata = train_adata
    else:
        preprocessed_adata = preprocess_adata_func(train_adata, **preprocess_adata_func_kwargs)
    
    # Remove `ctrl_name` from `test_perts` if present
    if ctrl_name in test_perts:
        test_perts.remove(ctrl_name)
    
    # Check control cells exist in adata
    perts_in_train_adata = train_adata.obs[pert_col].unique().tolist()
    assert ctrl_name in perts_in_train_adata, (
        f"Control cells '{ctrl_name}' not found in adata.obs['{pert_col}']"
    )
    perts_in_train_adata.remove(ctrl_name)
    
    # Convert adata.X to dense matrix
    if hasattr(preprocessed_adata.X, "toarray"):
        logger.info("Converting adata.X to dense matrix.")
        preprocessed_adata.X = preprocessed_adata.X.toarray()
    
    # Extract feature names (genes, ccres, ...)
    if feature_names_col is not None:
        logger.info(f"Extracting feature names from adata.var['{feature_names_col}'].")
        features = preprocessed_adata.var[feature_names_col].values
    else:
        logger.info("Extracting feature names from adata.var_names.")
        features = preprocessed_adata.var_names.values
    
    # Extract X_ctrl. & Calculate perturbed_mean.
    X_ctrl = preprocessed_adata[preprocessed_adata.obs[pert_col] == ctrl_name].X
    perturbed_mean = preprocessed_adata[preprocessed_adata.obs[pert_col] != ctrl_name].X.mean(axis=0, keepdims=True)

    # Calculate perturbed mean or matching mean for each perturbation
    if mean_type == MeanBaselineType.PERTURBED_MEAN:
        logger.info("Calculating perturbed_mean baseline.")
        X_pert = perturbed_mean.repeat(len(test_perts), axis=0)
    elif mean_type == MeanBaselineType.MATCHING_MEAN:
        logger.info("Calculating matching_mean baseline.")
        X_pert = []
        for pert in tqdm(test_perts):
            if "+" in pert:  # multi-gene perturbations
                gene_list = [g.strip() for g in pert.split("+")]
                # gene present in training adata, use its mean; gene not present in training adata, use perturbed mean
                X_pert.append(np.concatenate([
                    preprocessed_adata[preprocessed_adata.obs[pert_col] == g].X.mean(axis=0, keepdims=True)
                    if g in perts_in_train_adata
                    else perturbed_mean
                    for g in gene_list
                ], axis=0).mean(axis=0, keepdims=True))
            else:  # single-gene perturbation -> slightly different from perturbed mean baseline for gene present in training adata.
                # gene present in training adata, use its mean; gene not present in training adata, use perturbed mean
                X_pert.append(
                    preprocessed_adata[preprocessed_adata.obs[pert_col] == pert].X.mean(axis=0, keepdims=True)
                    if pert in perts_in_train_adata
                    else perturbed_mean
                )
        X_pert = np.concatenate(X_pert, axis=0)
    
    # Organize results into anndata
    logger.info("Organizing results into AnnData.")
    result_adata = AnnData(
        X=np.concatenate([X_pert, X_ctrl], axis=0), 
        obs={pert_col: test_perts + [ctrl_name] * X_ctrl.shape[0]},
        var=pd.DataFrame(index=features)
    )
    return result_adata