#!/usr/bin/env zsh

export NAME=blaidd &&

python /home/kade/diffusion/sd-scripts/lumina_train_network.py \
  --resolution 1024,1024 \
  --train_data_dir=/home/kade/toolkit/diffusion/datasets/blaidd/ \
  --caption_extension=".txt" \
  --pretrained_model_name_or_path=/home/kade/diffusion/comfy/models/checkpoints/lumina_2_model_bf16.safetensors \
  --gemma2=/home/kade/diffusion/comfy/models/text_encoders/gemma_2_2b_fp16.safetensors \
  --ae=/home/kade/diffusion/comfy/models/vae/ae.safetensors \
  --network_module="lycoris.kohya" \
  --network_args \
    preset=full \
    conv_dim=100000 \
    decompose_both=False \
    conv_alpha=64 \
    rank_dropout=0 \
    module_dropout=0 \
    use_tucker=True \
    use_scalar=False \
    rank_dropout_scale=False \
    algo=lokr \
    bypass_mode=False \
    factor=16 \
    dora_wd=True \
    train_norm=False \
  --network_dim=100000 \
  --network_alpha=64 \
  --network_dropout=0 \
  --no_half_vae --sdpa --mixed_precision="bf16" \
  --save_model_as="safetensors" \
  --save_precision="fp16" \
  --save_every_n_steps=100 \
  --sample_prompts=/home/kade/toolkit/diffusion/datasets/blaidd/sample-prompts.txt \
  --sample_every_n_steps=100 \
  --sample_sampler="euler" \
  --sample_at_first \
  --optimizer_type=SAVEUS \
  --train_batch_size=14 \
  --max_grad_norm=1 \
  --gradient_checkpointing \
  --lr_warmup_steps=0 \
  --learning_rate=0.0004 \
  --unet_lr=0.0004 \
  --text_encoder_lr=0.0001 \
  --lr_scheduler="cosine" \
  --lr_scheduler_args="num_cycles=0.375" \
  --multires_noise_iterations=12 \
  --multires_noise_discount=0.4 \
  --log_with=tensorboard \
  --seed=1728871242 \
  --dataset_repeats=1 \
  --enable_bucket \
  --bucket_reso_steps=64 \
  --min_bucket_reso=256 \
  --max_bucket_reso=2048 \
  --flip_aug \
  --shuffle_caption \
  --cache_latents \
  --cache_latents_to_disk \
  --max_data_loader_n_workers=8 \
  --persistent_data_loader_workers \
  --debiased_estimation_loss \
  --max_token_length=225 \
  --keep_tokens=1 \
  --keep_tokens_separator="|||" \
  --output_dir="/home/kade/diffusion/output_dir/lumina/blaidd-lycoris" \
  --output_name="blaidd-lycoris" \
  --log_prefix="blaidd-lycoris-" \
  --logging_dir="/home/kade/output_dir/logs" \
  --max_train_steps=1600 