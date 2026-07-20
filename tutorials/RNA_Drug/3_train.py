import os
import datasets
from descope import set_verbosity_warning
from descope.trainer import DeSCOPETrainer
from descope.dataset import HFDatasetForRNA
from transformers import HfArgumentParser, set_seed
from datasets import load_from_disk, concatenate_datasets
from descope.models.modeling_descope import DeSCOPEForRNA
from descope.models.configuration_descope import DeSCOPEConfig
from descope.arguments import (
    DeSCOPEDataArguments, 
    DeSCOPEModelArguments, 
    DeSCOPETrainingArguments
)


def main():
    parser = HfArgumentParser((DeSCOPEDataArguments, DeSCOPEModelArguments, DeSCOPETrainingArguments))
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()
    data_args: DeSCOPEDataArguments
    model_args: DeSCOPEModelArguments
    train_args: DeSCOPETrainingArguments

    # seed everything
    set_seed(train_args.seed)

    # suppress DeSCOPE's verbose logging to reduce console output
    set_verbosity_warning()
    datasets.logging.set_verbosity_warning()

    # create dataset
    try:
        hf_dataset = load_from_disk(data_args.tokenized_datasets_dir, keep_in_memory=data_args.keep_in_memory)
    except FileNotFoundError:
        hf_dataset_dir = [
            os.path.join(data_args.tokenized_datasets_dir, cellline) 
            for cellline in os.listdir(data_args.tokenized_datasets_dir)
        ]
        hf_dataset = concatenate_datasets([
            load_from_disk(cellline_dir, keep_in_memory=data_args.keep_in_memory)
            for cellline_dir in hf_dataset_dir
        ])

    train_ds = HFDatasetForRNA(
        hf_dataset=hf_dataset,
        ctrl_name=data_args.ctrl_name,
        gene_embs_file=data_args.gene_embs_file
    )

    # create model
    if train_args.pretrained_model_name_or_path is None:
        print("Training from scratch ...")
        config = DeSCOPEConfig(
            input_pert_gene_embedding_size=len(train_ds[0]["pert_gene_emb"]),
            input_length=len(train_ds[0]["labels"]),
            hidden_act=model_args.hidden_act,
            hidden_size=model_args.hidden_size,
            hidden_dropout=model_args.hidden_dropout,
            pert_gene_encoder_layers=model_args.pert_gene_encoder_layers,
            variational_encoder_layers=model_args.variational_encoder_layers,
            variational_decoder_layers=model_args.variational_decoder_layers,
            add_layernorm=model_args.add_layernorm
        )

        model = DeSCOPEForRNA(
            config=config,
            alpha_mse_loss=train_args.alpha_mse_loss,
            alpha_kl_loss=train_args.alpha_kl_loss
        )
    else:
        print(f"Loading pretrained model from {train_args.pretrained_model_name_or_path} ...")
        model = DeSCOPEForRNA.from_pretrained(
            train_args.pretrained_model_name_or_path,
            alpha_mse_loss=train_args.alpha_mse_loss,
            alpha_kl_loss=train_args.alpha_kl_loss
        )

    # train
    trainer = DeSCOPETrainer(
        args=train_args,
        model=model,
        train_dataset=train_ds,
        data_collator=train_ds.collate_fn
    )

    trainer.train()


if __name__ == "__main__":
    main()
