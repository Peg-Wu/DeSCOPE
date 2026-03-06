import torch
import random
import logging
import cell_eval
import numpy as np
import pandas as pd
import scanpy as sc
import polars as pl
import anndata as ad

from pathlib import Path
from tqdm.auto import tqdm
from typing import Union, Optional
from abc import ABC, abstractmethod
from easydict import EasyDict as edict
from cell_eval import MetricsEvaluator
from cell_eval._types import MetricType
from transformers import PreTrainedModel
from cell_eval.metrics import metrics_registry
from cell_eval._evaluator import PerturbationAnndataPair
from .utils import (
    pearson, 
    pearson_delta_on_topk_de,
    direction_match_on_topk_de,
    edistance, 
    load_gene_embs,
    preprocess_atac_perturbation_adata_consistent_with_epiagent,
    preprocess_rna_perturbation_adata
)
from .models.modeling_descope import DeSCOPEForATAC, DeSCOPEForRNA

logger = logging.getLogger(__name__)


def _build_anndata_pair_NOCHECK(
    real: Union[sc.AnnData, str],
    pred: Union[sc.AnnData, str],
    control_pert: str,
    pert_col: str,
    allow_discrete: bool = False,
) -> PerturbationAnndataPair:
    r"""
    Builds a PerturbationAnndataPair without performing cell-eval's default data validation,
    particularly the check for log1p-normalized expression values.
    
    This is useful when the model outputs raw or non-log-transformed predictions (e.g., negative values),
    and the user wishes to bypass format enforcement to compute metrics directly on the given scale.
    """
    logger.warning("-------------------------------------------------")
    logger.warning("Running cell-eval without data format validation.")
    logger.warning("-------------------------------------------------")

    if isinstance(real, str):
        logger.info(f"Reading real anndata from {real}")
        real = ad.read_h5ad(real)
    if isinstance(pred, str):
        logger.info(f"Reading pred anndata from {pred}")
        pred = ad.read_h5ad(pred)

    # Validate that the input is normalized and log-transformed
    # _convert_to_normlog(real, which="real", allow_discrete=allow_discrete)
    # _convert_to_normlog(pred, which="pred", allow_discrete=allow_discrete)

    # Build the anndata pair
    return PerturbationAnndataPair(
        real=real, pred=pred, control_pert=control_pert, pert_col=pert_col
    )


class CellEvalMixin:
    r"""
    A mixin class designed to extend evaluation capabilities for single-cell perturbation prediction tasks. 
    This class provides methods to compute various metrics on predicted and real AnnData objects, including differential expression (DE) analysis etc.

    **Example:**

    >>> from descope.inference import CellEvalMixin

    >>> evalmixin = CellEvalMixin()
    >>> print(evalmixin.all_metrics)  # list all metrics in cell-eval

    >>> results, agg_results, evaluator = evalmixin.compute_metrics(...)

    >>> # Compute extra metrics
    >>> evalmixin.extra_metrics_func.pearson(evaluator.anndata_pair, ...)  # pearson
    >>> evalmixin.extra_metrics_func.pearson_delta_on_topk_de(evaluator.anndata_pair, ...)  # pearson_delta_on_topk_de
    >>> evalmixin.extra_metrics_func.direction_match_on_topk_de(evaluator.anndata_pair, ...)  # direction_match_on_topk_de
    >>> evalmixin.extra_metrics_func.edistance(...)  # edistance
    """
    def __init__(self):
        self.extra_metrics_func = {
            "pearson": pearson,
            "pearson_delta_on_topk_de": pearson_delta_on_topk_de,
            "direction_match_on_topk_de": direction_match_on_topk_de,
            "edistance": edistance
        }
        self.extra_metrics_func = edict(self.extra_metrics_func)

    @property
    def all_metrics(self) -> list:
        # Includes only the built-in metrics provided by cell-eval.
        return metrics_registry.list_metrics()

    @property
    def all_de_metrics(self) -> list:
        # Includes only the built-in metrics provided by cell-eval.
        return metrics_registry.list_metrics(MetricType.DE)
    
    @property
    def all_anndata_pair_metrics(self) -> list:
        # Includes only the built-in metrics provided by cell-eval.
        return metrics_registry.list_metrics(MetricType.ANNDATA_PAIR)
    
    @property
    def extra_metrics(self) -> list:
        return list(self.extra_metrics_func.keys())

    def check_whether_skip_de_metrics(
        self,
        metrics_to_calculate: list
    ) -> bool:
        de_metrics = self.all_de_metrics
        skip_de = True if len(set(metrics_to_calculate) & set(de_metrics)) == 0 else False
        return skip_de

    def _compute_metrics(
        self,
        adata_pred: Union[str, sc.AnnData],
        adata_real: Union[str, sc.AnnData],
        metrics_to_calculate: list,
        de_pred: Optional[str] = None,
        de_real: Optional[str] = None,  # If available, provide DE results to speed up computation.
        control_pert: str = "non-targeting",
        pert_col: str = "target_gene",
        de_method: str = "wilcoxon",
        num_threads: int = -1,
        outdir: str = "./cell-eval-outdir",
    ) -> tuple[pl.DataFrame, pl.DataFrame, MetricsEvaluator]:
        evaluator = MetricsEvaluator(
            adata_pred=adata_pred,
            adata_real=adata_real,
            de_pred=de_pred,
            de_real=de_real,
            control_pert=control_pert,
            pert_col=pert_col,
            de_method=de_method,
            num_threads=num_threads,
            outdir=outdir,
            skip_de=self.check_whether_skip_de_metrics(metrics_to_calculate)
        )

        (results, agg_results) = evaluator.compute(
            profile="full",
            skip_metrics=list(set(self.all_metrics) - set(metrics_to_calculate))
        )

        # Return evaluator for calculating pearson and pearson_delta_on_topk_de
        return results, agg_results, evaluator

    def compute_metrics(
        self,
        adata_pred: Union[str, sc.AnnData],
        adata_real: Union[str, sc.AnnData],
        metrics_to_calculate: list,
        de_pred: Optional[str] = None,
        de_real: Optional[str] = None,  # If available, provide DE results to speed up computation.
        control_pert: str = "non-targeting",
        pert_col: str = "target_gene",
        de_method: str = "wilcoxon",
        num_threads: int = -1,
        outdir: str = "./cell-eval-outdir",
    ) -> tuple[pl.DataFrame, pl.DataFrame, MetricsEvaluator]:
        # ATAC: 
        # Since the prediction target is TF-IDF, there is no need to perform normalization or log1p checks.

        # RNA: 
        # As some methods allow the model to output negative values, 
        # normalization and log1p checks should also be disabled; 
        # however, note that this step needs to be handled manually.

        # If you want to validate the adata format, please call the `self._compute_metrics` method.

        cell_eval._evaluator._build_anndata_pair = _build_anndata_pair_NOCHECK

        return self._compute_metrics(
            adata_pred=adata_pred,
            adata_real=adata_real,  # self.adata_test_ground_truth
            metrics_to_calculate=metrics_to_calculate,
            de_pred=de_pred,
            de_real=de_real,
            control_pert=control_pert,
            pert_col=pert_col,
            de_method=de_method,
            num_threads=num_threads,
            outdir=outdir,
        )


class BaseInference(CellEvalMixin, ABC):
    def __init__(
        self,
        test_csv_template_fp: str,
        adata: Union[str, sc.AnnData],  # where you select control cells for inference
        pretrained_model_name_or_path: str,
        pert_col: str = "perturbation",
        ctrl_name: str = "control"
    ):
        super().__init__()
        self.pert_col = pert_col
        self.ctrl_name = ctrl_name

        # Load test csv template
        self.test_csv_template = pd.read_csv(test_csv_template_fp)
        self._check_test_csv_template(self.test_csv_template)

        # Add test_genes attribute (involve control)
        self.test_genes = self.test_csv_template["target_gene"].unique().tolist()
        if ctrl_name not in self.test_genes:
            self.test_genes.append(ctrl_name)

        # Load model (abstract)
        self.model = self.load_model(pretrained_model_name_or_path)

        # Load and prepare anndata (abstract) -> Return (adata_test_ground_truth & adata_ctrl)
        if isinstance(adata, str):
            logger.info(f"Read AnnData from {adata} ...")
            adata = sc.read_h5ad(adata)
        adata = self.preprocess_adata(adata)
        self.adata_ctrl = self._prepare_control_adata(adata)
        self.adata_test_ground_truth = self._prepare_test_adata(adata)
        self.adata_pred = None  # Placeholder

        # Ensure control row exists in template
        if ctrl_name not in self.test_csv_template["target_gene"].unique():
            self.test_csv_template = pd.concat(
                [
                    self.test_csv_template,
                    pd.DataFrame({"target_gene": [ctrl_name], "n_cells": len(self.adata_ctrl)})
                ],
                ignore_index=True
            )
    
    @staticmethod
    def write_h5ad(adata: sc.AnnData, save_path: str):
        output_dir = Path(save_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(save_path)

    @staticmethod
    def _check_test_csv_template(test_csv_template: pd.DataFrame):
        columns = test_csv_template.columns
        required_columns = ["target_gene", "n_cells"]
        missing_columns = [c for c in required_columns if c not in columns]

        if missing_columns:
            raise ValueError(f"Missing required columns in test csv template: {missing_columns}")
    
    def _prepare_control_adata(self, adata: sc.AnnData) -> sc.AnnData:
        if self.ctrl_name not in adata.obs[self.pert_col].unique().tolist():
            raise ValueError(
                "No control cells detected in the input adata; inference cannot be performed."
            )
        else:
            adata_ctrl = adata[adata.obs[self.pert_col] == self.ctrl_name]
            logger.info("self.adata_ctrl registered successfully.")
        return adata_ctrl
    
    def _prepare_test_adata(self, adata: sc.AnnData) -> sc.AnnData:
        pert_genes_in_adata = adata.obs[self.pert_col].unique().tolist()
        missing = 0
        for g in self.test_genes:
            if g not in pert_genes_in_adata:
                missing += 1
        if missing > 0:
            logger.warning(
                "The input adata does not contain all the test perturbation genes; "
                "`self.adata_test_ground_truth` has been set to None; "
                "You need to manually provide the ground truth when computing metrics."
            )
            return
        else:
            adata_test_ground_truth = adata[adata.obs[self.pert_col].isin(self.test_genes)]
            logger.info("self.adata_test_ground_truth registered successfully.")
            return adata_test_ground_truth
    
    @torch.no_grad()
    def inference(
        self,
        gene_embs_file: str,
        device: torch.device,
        batch_size: int = 128,
        use_generated_control_cells: bool = False,
        seed: int = 42
    ):
        gene_embs = load_gene_embs(gene_embs_file, perts_to_emb=self.test_genes)

        if hasattr(self.adata_ctrl.X, "toarray"):
            X_ctrl = self.adata_ctrl.X.toarray()
        else:
            X_ctrl = self.adata_ctrl.X

        self.model = self.model.to(device)
        self.model.eval()
        random.seed(seed)
        pbar = tqdm(total=len(self.test_csv_template), desc="Inference")
        results = []
        for _, row in self.test_csv_template.iterrows():
            target_gene = row["target_gene"]
            n_cells = row["n_cells"]
            gen_n_cells = 0
            while True:
                selected_ctrl_cells_indices = random.choices(range(len(X_ctrl)), k=batch_size)
                selected_ctrl_cells_X = torch.tensor(
                    X_ctrl[selected_ctrl_cells_indices], dtype=torch.float32, device=device
                )
                pert_gene_emb = gene_embs[target_gene].unsqueeze(0).repeat(
                    len(selected_ctrl_cells_X), 1
                ).to(torch.float32).to(device)
                logits = self.model.inference(
                    selected_ctrl_cells_X,  # Main Input Name
                    pert_gene_emb=pert_gene_emb,
                ).logits.cpu().numpy()
                if np.any(np.isnan(logits)):  # Remove cells contain NaN.
                    print("Detected NaN, drop current batch.")
                    continue
                gen_n_cells += len(logits)
                if gen_n_cells <= n_cells:
                    results.append(logits)
                else:
                    results.append(logits[:-(gen_n_cells - n_cells)])
                    pbar.update(1)
                    break
        results = np.concatenate(results, axis=0)
        torch.cuda.empty_cache()

        # Organize final results
        self.adata_pred = self._organize_final_results(
            results, use_generated_control_cells
        )
        logger.info("Predicted results have been registered to self.adata_pred.")
    
    def _organize_final_results(
        self, 
        results: np.ndarray,
        use_generated_control_cells: bool = False
    ) -> sc.AnnData:
        target_gene = self.test_csv_template.apply(
            lambda r: [r["target_gene"]] * r["n_cells"], axis=1
        ).values.tolist()
        target_gene = sum(target_gene, [])

        predicted_adata = ad.AnnData(X=results)
        predicted_adata.obs[self.pert_col] = target_gene
        predicted_adata.var_names = self.adata_ctrl.var_names

        # If not use_generated_control_cells, replace generated control cells with real control cells.
        if not use_generated_control_cells:
            logger.info("You are using real control cells in the final predicted_adata.")
            predicted_adata = predicted_adata[predicted_adata.obs[self.pert_col] != self.ctrl_name]
            predicted_adata = sc.concat([predicted_adata, self.adata_ctrl], axis=0)
        # predicted_adata.X[predicted_adata.X < np.log1p(1)] = 0  # cutoff small values
        return predicted_adata
    
    @abstractmethod
    def load_model(self, pretrained_model_name_or_path: str) -> PreTrainedModel:
        raise NotImplementedError()
    
    @abstractmethod
    def preprocess_adata(self, adata: sc.AnnData) -> sc.AnnData:
        raise NotImplementedError()


class InferenceForATAC(BaseInference):
    def preprocess_adata(self, adata: sc.AnnData) -> sc.AnnData:
        # Extract topk ccres from model.config.input_length
        topk_ccres = self.model.config.input_length

        return preprocess_atac_perturbation_adata_consistent_with_epiagent(
            adata, topk_ccres, self.pert_col
        )

    def load_model(self, pretrained_model_name_or_path: str) -> PreTrainedModel:
        return DeSCOPEForATAC.from_pretrained(pretrained_model_name_or_path)


class InferenceForRNA(BaseInference):
    def __init__(
        self,
        test_csv_template_fp: str,
        adata: Union[str, sc.AnnData],  # where you select control cells for inference
        pretrained_model_name_or_path: str,
        pert_col: str = "target_gene",
        ctrl_name: str = "non-targeting",
        target_sum: float = 1e4,
        skip_raw_counts_check: bool = False
    ):
        self.target_sum = target_sum
        self.skip_raw_counts_check = skip_raw_counts_check
        super().__init__(
            test_csv_template_fp=test_csv_template_fp,
            adata=adata,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            pert_col=pert_col,
            ctrl_name=ctrl_name
        )

    def preprocess_adata(self, adata: sc.AnnData) -> sc.AnnData:
        return preprocess_rna_perturbation_adata(
            adata=adata, 
            target_sum=self.target_sum, 
            pert_col=self.pert_col,
            skip_raw_counts_check=self.skip_raw_counts_check
        )

    def load_model(self, pretrained_model_name_or_path: str) -> PreTrainedModel:
        return DeSCOPEForRNA.from_pretrained(pretrained_model_name_or_path)