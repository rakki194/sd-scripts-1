#!/usr/bin/env zsh

export NAME=blaidd &&

python /home/kade/diffusion/sd-scripts/lumina_train_network.py \
  --resolution 1024,1024 \
  --train_data_dir=/home/kade/toolkit/diffusion/datasets/blaidd/ \
  --caption_extension=".txt" \
  --pretrained_model_name_or_path=/home/kade/diffusion/comfy/models/checkpoints/lumina_2_model_bf16.safetensors \
  --gemma2=/home/kade/diffusion/comfy/models/text_encoders/gemma_2_2b_fp16.safetensors \
  --ae=/home/kade/diffusion/comfy/models/vae/ae.safetensors \
  --network_args \
    network_module=networks.lora_lumina \
    module_dropout=0.0 \
  --network_module="networks.lora_lumina" \
  --no_half_vae --sdpa --mixed_precision="bf16" \
  --save_model_as="safetensors" \
  --save_precision="fp16" \
  --save_every_n_steps=100 \
  --sample_prompts=/home/kade/toolkit/diffusion/datasets/blaidd/sample-prompts.txt \
  --sample_every_n_steps=100 \
  --sample_sampler="euler" \
  --sample_at_first \
  --network_dropout=0 \
  --optimizer_type=SAVEUS \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_grad_norm=1 \
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
  --max_data_loader_n_workers=2 \
  --network_dim=32 \
  --network_alpha=32 \
  --debiased_estimation_loss \
  --max_token_length=225 \
  --keep_tokens=1 \
  --keep_tokens_separator="|||" \
  --output_dir="/home/kade/diffusion/output_dir/lumina/blaidd-v1s1600" \
  --output_name="blaidd-v1s1600" \
  --log_prefix="blaidd-v1s1600-" \
  --logging_dir="/home/kade/output_dir/logs" \
  --max_train_steps=1600
