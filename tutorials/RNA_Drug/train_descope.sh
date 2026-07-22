#!/bin/bash

output_dir="./trainer_output_2k_hvgs"
tokenized_datasets_dir="/fse/home/wupengpeng/perturbation_datasets/tokenized_datasets/Tahoe_100M/2k_hvgs"
keep_in_memory=False
ctrl_name="DMSO_TF_0.0"
gene_embs_file="./metadata/tahoe100m_drug_dose_embed.pt"


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
    --num_train_epochs=10 \
    --logging_steps=200 \
    --checkpointing_steps=epoch-2 \
    --per_device_train_batch_size=256 \
    --gradient_accumulation_steps=1 \
    --max_grad_norm=1.0 \
    --learning_rate=1e-4 \
    --lr_scheduler_type=cosine \
    --weight_decay=1e-2 \
    --num_warmup_ratio=0.05 \
    --mixed_precision=bf16 \
    --with_tracking=True \
    --report_to=tensorboard \
    --dataloader_pin_memory=True \
    --dataloader_persistent_workers=True \
    --dataloader_num_workers=8 \
    --dataloader_prefetch_factor=2 \
    --alpha_mse_loss=1.0 \
    --alpha_kl_loss=1.0"


export CUDA_VISIBLE_DEVICES="4,5,6,7"
accelerate launch \
    --config_file="./accelerate_config.yaml" \
    --num_processes=4 \
    3_train.py \
    $TRAINPARAMS \
    $DATAPARAMS \
    $MODELPARAMS
