#!/bin/bash

# Pretrain related parameters
pretrain_output_dir="./results/descope_loo/trainer_output_pretrain/loo_K562_ESSENTIAL"
pretrain_tokenized_datasets_dir="/fse/home/wupengpeng/perturbation_data_origin/RNA/tokenized_dataset/loo_K562_ESSENTIAL"

# Finetune related parameters
finetune_output_dir="./results/descope_loo/trainer_output_finetune/loo_K562_ESSENTIAL"
finetune_tokenized_datasets_dir="/fse/home/wupengpeng/perturbation_data_origin/RNA/K562_ESSENTIAL/tokenized_dataset"

# Common parameters
keep_in_memory=False
ctrl_name="non-targeting"
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
    --keep_in_memory=$keep_in_memory \
    --ctrl_name=$ctrl_name \
    --gene_embs_file=$gene_embs_file"


TRAINPARAMS="
    --seed=42 \
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


# ===================
# Step 1: Pretraining
# ===================
export CUDA_VISIBLE_DEVICES="0"
echo "=================================================="
echo "[Stage 1 | Pretraining]"
echo "  Dataset (tokenized): $pretrain_tokenized_datasets_dir"
echo "  Output dir         : $pretrain_output_dir"
echo "  CUDA devices       : $CUDA_VISIBLE_DEVICES"
echo "=================================================="

accelerate launch \
    --config_file="./accelerate_config.yaml" \
    --num_processes=1 \
    train.py \
    --tokenized_datasets_dir=$pretrain_tokenized_datasets_dir \
    --output_dir=$pretrain_output_dir \
    $TRAINPARAMS \
    $DATAPARAMS \
    $MODELPARAMS


# ==================
# Step 2: Finetuning
# ==================
export CUDA_VISIBLE_DEVICES="0"
echo "=================================================="
echo "[Stage 2 | Finetuning]"
echo "  Dataset (tokenized): $finetune_tokenized_datasets_dir"
echo "  Output dir         : $finetune_output_dir"
echo "  CUDA devices       : $CUDA_VISIBLE_DEVICES"
echo "=================================================="

accelerate launch \
    --config_file="./accelerate_config.yaml" \
    --num_processes=1 \
    train.py \
    --pretrained_model_name_or_path="$pretrain_output_dir/last_model" \
    --tokenized_datasets_dir=$finetune_tokenized_datasets_dir \
    --output_dir=$finetune_output_dir \
    $TRAINPARAMS \
    $DATAPARAMS \
    $MODELPARAMS