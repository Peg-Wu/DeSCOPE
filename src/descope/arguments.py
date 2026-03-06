from typing import Optional
from wppkg import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class DeSCOPETrainingArguments(TrainingArguments):
    alpha_mse_loss: float = field(
        default=1.0, metadata={
            "help": "The weight (or coefficient) for the MSE loss."
        }
    )
    alpha_kl_loss: float = field(
        default=1.0, metadata={
            "help": "The weight (or coefficient) for the KL loss."
        }
    )
    pretrained_model_name_or_path: Optional[str] = field(
        default=None, metadata={
            "help": "Path to a pretrained model."
        }
    )


@dataclass
class DeSCOPEDataArguments:
    tokenized_datasets_dir: str = field(
        default="./tokenized_dataset/K562", metadata={
            "help": "Path to the tokenized huggingface datasets."
        }
    )
    keep_in_memory: bool = field(
        default=False, metadata={
            "help": "Whether to keep the datasets in memory."
        }
    )
    ctrl_name: str = field(
        default="control", metadata={
            "help": "The name in huggingface datasets that represents control cells."
        }
    )
    gene_embs_file: str = field(
        default="./ESM2_pert_features.pt", metadata={
            "help": "Path to the gene embedding file."
        }
    )


@dataclass
class DeSCOPEModelArguments:
    hidden_act: str = field(
        default="gelu", metadata={
            "help": "The activation function for the hidden layers."
        }
    )
    hidden_size: int = field(
        default=672, metadata={
            "help": "The hidden size of the model."
        }
    )
    hidden_dropout: float = field(
        default=0, metadata={
            "help": "The dropout rate for the hidden layers."
        }
    )
    pert_gene_encoder_layers: int = field(
        default=1, metadata={
            "help": "The number of layers in the perturbation gene encoder."
        }
    )
    variational_encoder_layers: int = field(
        default=4, metadata={
            "help": "The number of layers in the variational encoder."
        }
    )
    variational_decoder_layers: int = field(
        default=4, metadata={
            "help": "The number of layers in the variational decoder."
        }
    )
    add_layernorm: bool = field(
        default=True, metadata={
            "help": "Whether to add layer normalization to the hidden layers."
        }
    )