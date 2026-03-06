#!/bin/bash

output_dir="./results/descope/trainer_output/GM12878"
tokenized_datasets_dir="/fse/home/wupengpeng/perturbation_data_origin/ATAC/GM12878/tokenized_dataset"
keep_in_memory=False
ctrl_name="control"
gene_embs_file="/fse/home/wupengpeng/DeSCOPE/ESM2_pert_features.pt"


MODELPARAMS="
    --hidden_act=gelu \
    --hidden_size=672 \
    --hidden_dropout=0 \
    --pert_gene_encoder_layers=1 \
    --variational_encoder_layers=4 \
    --variational_decoder_layers=4 \
    --add_layernorm=True"


DATAPARAMS="
    --tokenized_datasets_dir=$tokenized_datasets_dir \
    --keep_in_memory=$keep_in_memory \
    --ctrl_name=$ctrl_name \
    --gene_embs_file=$gene_embs_file"


TRAINPARAMS="
    --seed=42 \
    --output_dir=$output_dir \
    --num_train_epochs=20 \
    --logging_steps=50 \
    --checkpointing_steps=epoch-10 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=1 \
    --max_grad_norm=1.0 \
    --learning_rate=1e-4 \
    --lr_scheduler_type=cosine \
    --weight_decay=1e-2 \
    --num_warmup_ratio=0.1 \
    --mixed_precision=bf16 \
    --with_tracking=True \
    --report_to=tensorboard \
    --dataloader_pin_memory=True \
    --dataloader_persistent_workers=True \
    --dataloader_num_workers=16 \
    --dataloader_prefetch_factor=2 \
    --alpha_mse_loss=1.0 \
    --alpha_kl_loss=1.0"


export CUDA_VISIBLE_DEVICES="0"
accelerate launch \
    --config_file="./accelerate_config.yaml" \
    --num_processes=1 \
    train.py \
    $TRAINPARAMS \
    $DATAPARAMS \
    $MODELPARAMS
