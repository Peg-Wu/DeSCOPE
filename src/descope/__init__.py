__version__ = "0.0.1"

from .logging import (
    set_verbosity_info,
    set_verbosity_warning
)

set_verbosity_info()

from .models.configuration_descope import DeSCOPEConfig
from .models.modeling_descope import DeSCOPEForATAC, DeSCOPEForRNA
from .dataset import HFDatasetForATAC, HFDatasetForRNA
from .arguments import DeSCOPEDataArguments, DeSCOPEModelArguments, DeSCOPETrainingArguments

from .tokenizer import (
    TokenizerForATAC, TokenizerForRNA,
    tokenize_adata_to_hf_dataset_for_atac, 
    tokenize_adata_to_hf_dataset_for_rna
)
from .inference import InferenceForATAC, InferenceForRNA
from .trainer import DeSCOPETrainer

from .utils import (
    UniformFeatureForAnnData,
    intersect_adatas_for_celltype_transfer,
    load_gene_names_engine,
    load_gene_embs,
    preprocess_atac_perturbation_adata_consistent_with_epiagent,
    preprocess_rna_perturbation_adata,
    pearson_delta_on_topk_de,
    direction_match_on_topk_de,
    pearson,
    edistance,
    build_de
)


def welcome():
    logo = r"""
            ____      _____ __________  ____  ______
           / __ \___ / ___// ____/ __ \/ __ \/ ____/
          / / / / _ \\__ \/ /   / / / / /_/ / __/   
         / /_/ /  __/__/ / /___/ /_/ / ____/ /___   
        /_____/\___/____/\____/\____/_/   /_____/   
                                            
    """
    from . import __version__
    print(
        f"{logo}\n"
        f"--------------------------------------------------------------------\n"
        f"- DeSCOPE: Decoding Single-Cell Observations of Perturbed Expression\n"
        f"- Version: {__version__}\n"
        f"- Repository: https://github.com/Peg-Wu/DeSCOPE\n"
        f"- Documentation: https://descope.readthedocs.io\n"
        f"- Lab Website: https://wanglabtongji.github.io\n"
        f"- Authors:\n"
        f"\t- Pengpeng Wu\n"
        f"\t\t- Email: peg2_wu@163.com\n"
        f"\t\t- Github: https://github.com/Peg-Wu\n"
        f"\t\t- Homepage: https://peg-wu.github.io/\n"
        f"\t- Hailin Wei\n"
        f"\t\t- Email: hailinwei98@gmail.com\n"
        f"\t\t- Github: https://github.com/HailinWei98\n"
        f"\t- Yazi Li\n"
        f"\t\t- Email: liyazi23811@gmail.com\n"
        f"\t\t- Github: https://github.com/liyazi712\n"
        f"--------------------------------------------------------------------\n"
        f"🎉 Feel free to contact us if you have any questions or suggestions.\n"
        f"--------------------------------------------------------------------"
    )