# training with captions
# XXX dropped option: hypernetwork training

import argparse
import math
import os
from multiprocessing import Value
import toml

from tqdm import tqdm

import torch
from library import deepspeed_utils
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from diffusers import DDPMScheduler

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.train_util as train_util
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    apply_debiased_estimation,
)
import torch.profiler


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    cache_latents = args.cache_latents

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    tokenizer = train_util.load_tokenizer(args)

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(False, True, False, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            user_config = {
                "datasets": [
                    {
                        "subsets": [
                            {
                                "image_dir": args.train_data_dir,
                                "metadata_file": args.in_json,
                            }
                        ]
                    }
                ]
            }

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(64)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

    # モデルを読み込む
    text_encoder, vae, unet, load_stable_diffusion_format = train_util.load_target_model(args, weight_dtype, accelerator)

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())

    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        #   model.set_use_memory_efficient_attention_xformers(valid)            # 次のリリースでなくなりそう
        # pipeが自動で再帰的にset_use_memory_efficient_attention_xformersを探すんだって(;´Д｀)
        # U-Netだけ使う時にはどうすればいいのか……仕方ないからコピって使うか
        # 0.10.2でなんか巻き戻って個別に指定するようになった(;^ω^)

        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # Incorporate xformers and memory efficient attention into the model
    if args.diffusers_xformers:
        accelerator.print("Use xformers by Diffusers")
        set_diffusers_xformers_flag(unet, True)
    else:
        # Windows version of xformers cannot train with float, so we need to make it possible to disable xformers
        accelerator.print("Disable Diffusers' xformers")
        set_diffusers_xformers_flag(unet, False)
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)

    # Prepare for training
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # Prepare for training: Set models to appropriate states
    training_models = []
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    training_models.append(unet)

    if args.train_text_encoder:
        accelerator.print("enable text encoder training")
        if args.gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
        training_models.append(text_encoder)
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)  # text encoder is not trained
        if args.gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
            text_encoder.train()  # required for gradient_checkpointing
        else:
            text_encoder.eval()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    for m in training_models:
        m.requires_grad_(True)

    trainable_params = []
    if args.learning_rate_te is None or not args.train_text_encoder:
        for m in training_models:
            trainable_params.extend(m.parameters())
    else:
        trainable_params = [
            {"params": list(unet.parameters()), "lr": args.learning_rate},
            {"params": list(text_encoder.parameters()), "lr": args.learning_rate_te},
        ]

    # Prepare classes needed for training
    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = train_util.get_optimizer(args, trainable_params=trainable_params)

    # Prepare dataloader
    # Note: DataLoader process count of 0 cannot use persistent_workers
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # Calculate number of training steps
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is: {args.max_train_steps}"
        )

    # Send training steps to dataset side
    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # Prepare learning rate scheduler
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # Experimental feature: Enable full fp16 training including gradients - convert entire model to fp16
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16'"
        accelerator.print("enable full fp16 training.")
        unet.to(weight_dtype)
        text_encoder.to(weight_dtype)

    if args.deepspeed:
        if args.train_text_encoder:
            ds_model = deepspeed_utils.prepare_deepspeed_model(args, unet=unet, text_encoder=text_encoder)
        else:
            ds_model = deepspeed_utils.prepare_deepspeed_model(args, unet=unet)
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]
    else:
        # accelerator handles model preparation appropriately
        if args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)

    # Experimental feature: Enable full fp16 training including gradients - Apply patch to PyTorch to enable grad scale in fp16
    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # Start training
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training")
    accelerator.print(f"  num examples: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
    accelerator.print(f"  num epochs: {num_train_epochs}")
    accelerator.print(f"  batch size per device: {args.train_batch_size}")
    accelerator.print(
        f"  total train batch size (with parallel & distributed & accumulation): {total_batch_size}"
    )
    accelerator.print(f"  gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "finetuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    # For --sample_at_first
    train_util.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

    loss_recorder = train_util.LossRecorder()
    profiler_ctx = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) if getattr(args, "profiler_output", None) else None
    )
    if profiler_ctx:
        with profiler_ctx as prof:
            for epoch in range(num_train_epochs):
                accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
                current_epoch.value = epoch + 1

                for m in training_models:
                    m.train()

                for step, batch in enumerate(train_dataloader):
                    current_step.value = global_step
                    with accelerator.accumulate(*training_models):
                        with torch.no_grad():
                            if "latents" in batch and batch["latents"] is not None:
                                latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                            else:
                                # convert to latent
                                latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample().to(weight_dtype)
                            latents = latents * 0.18215
                        b_size = latents.shape[0]

                        with torch.set_grad_enabled(args.train_text_encoder):
                            # Get the text embedding for conditioning
                            if args.weighted_captions:
                                encoder_hidden_states = get_weighted_text_embeddings(
                                    tokenizer,
                                    text_encoder,
                                    batch["captions"],
                                    accelerator.device,
                                    args.max_token_length // 75 if args.max_token_length else 1,
                                    clip_skip=args.clip_skip,
                                )
                            else:
                                input_ids = batch["input_ids"].to(accelerator.device)
                                encoder_hidden_states = train_util.get_hidden_states(
                                    args, input_ids, tokenizer, text_encoder, None if not args.full_fp16 else weight_dtype
                                )

                        # Sample noise, sample a random timestep for each image, and add noise to the latents,
                        # with noise offset and/or multires noise if specified
                        noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(
                            args, noise_scheduler, latents
                        )

                        # Predict the noise residual
                        with accelerator.autocast():
                            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                        if args.v_parameterization:
                            # v-parameterization training
                            target = noise_scheduler.get_velocity(latents, noise, timesteps)
                        else:
                            target = noise

                        if args.min_snr_gamma or args.scale_v_pred_loss_like_noise_pred or args.debiased_estimation_loss:
                            # do not mean over batch dimension for snr weight or scale v-pred loss
                            loss = train_util.conditional_loss(
                                noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c
                            )
                            loss = loss.mean([1, 2, 3])

                            if args.min_snr_gamma:
                                loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                            if args.scale_v_pred_loss_like_noise_pred:
                                loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                            if args.debiased_estimation_loss:
                                loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)

                            loss = loss.mean()  # mean over batch dimension
                        else:
                            loss = train_util.conditional_loss(
                                noise_pred.float(), target.float(), reduction="mean", loss_type=args.loss_type, huber_c=huber_c
                            )
                        # Apply scaled MSE loss if enabled
                        if getattr(args, "scale_mse_loss", None) is not None and args.loss_type == "l2":
                            loss = loss * args.scale_mse_loss

                        accelerator.backward(loss)
                        if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                            params_to_clip = []
                            for m in training_models:
                                params_to_clip.extend(m.parameters())
                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        if optimizer.__class__.__name__ == "SPARKLES" and getattr(args, "optimizer_profiling", False):
                            train_util.optimizer_step_with_profiling(optimizer, profiling_enabled=True)
                        else:
                            optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                        train_util.sample_images(
                            accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet
                        )

                        # Save model at specified steps
                        if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:
                                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                                train_util.save_sd_model_on_epoch_end_or_stepwise(
                                    args,
                                    False,
                                    accelerator,
                                    src_path,
                                    save_stable_diffusion_format,
                                    use_safetensors,
                                    save_dtype,
                                    epoch,
                                    num_train_epochs,
                                    global_step,
                                    accelerator.unwrap_model(text_encoder),
                                    accelerator.unwrap_model(unet),
                                    vae,
                                )

                    current_loss = loss.detach().item()  # Average loss, so batch size doesn't matter
                    if args.logging_dir is not None:
                        logs = {"loss": current_loss}
                        train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=True)
                        accelerator.log(logs, step=global_step)

                    loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                    avr_loss: float = loss_recorder.moving_average
                    logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)

                    if global_step >= args.max_train_steps:
                        break

                if args.logging_dir is not None:
                    logs = {"loss/epoch": loss_recorder.moving_average}
                    accelerator.log(logs, step=epoch + 1)

                accelerator.wait_for_everyone()

                if args.save_every_n_epochs is not None:
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        train_util.save_sd_model_on_epoch_end_or_stepwise(
                            args,
                            True,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder),
                            accelerator.unwrap_model(unet),
                            vae,
                        )

                train_util.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

                prof.export_chrome_trace(args.profiler_output)
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    else:
        for epoch in range(num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            for m in training_models:
                m.train()

            for step, batch in enumerate(train_dataloader):
                current_step.value = global_step
                with accelerator.accumulate(*training_models):
                    with torch.no_grad():
                        if "latents" in batch and batch["latents"] is not None:
                            latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                        else:
                            # latentに変換
                            latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample().to(weight_dtype)
                        latents = latents * 0.18215
                    b_size = latents.shape[0]

                    with torch.set_grad_enabled(args.train_text_encoder):
                        # Get the text embedding for conditioning
                        if args.weighted_captions:
                            encoder_hidden_states = get_weighted_text_embeddings(
                                tokenizer,
                                text_encoder,
                                batch["captions"],
                                accelerator.device,
                                args.max_token_length // 75 if args.max_token_length else 1,
                                clip_skip=args.clip_skip,
                            )
                        else:
                            input_ids = batch["input_ids"].to(accelerator.device)
                            encoder_hidden_states = train_util.get_hidden_states(
                                args, input_ids, tokenizer, text_encoder, None if not args.full_fp16 else weight_dtype
                            )

                    # Sample noise, sample a random timestep for each image, and add noise to the latents,
                    # with noise offset and/or multires noise if specified
                    noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(
                        args, noise_scheduler, latents
                    )

                    # Predict the noise residual
                    with accelerator.autocast():
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if args.v_parameterization:
                        # v-parameterization training
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    if args.min_snr_gamma or args.scale_v_pred_loss_like_noise_pred or args.debiased_estimation_loss:
                        # do not mean over batch dimension for snr weight or scale v-pred loss
                        loss = train_util.conditional_loss(
                            noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c
                        )
                        loss = loss.mean([1, 2, 3])

                        if args.min_snr_gamma:
                            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                        if args.scale_v_pred_loss_like_noise_pred:
                            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                        if args.debiased_estimation_loss:
                            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)

                        loss = loss.mean()  # mean over batch dimension
                    else:
                        loss = train_util.conditional_loss(
                            noise_pred.float(), target.float(), reduction="mean", loss_type=args.loss_type, huber_c=huber_c
                        )
                    # Apply scaled MSE loss if enabled
                    if getattr(args, "scale_mse_loss", None) is not None and args.loss_type == "l2":
                        loss = loss * args.scale_mse_loss

                    accelerator.backward(loss)
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in training_models:
                            params_to_clip.extend(m.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    if optimizer.__class__.__name__ == "SPARKLES" and getattr(args, "optimizer_profiling", False):
                        train_util.optimizer_step_with_profiling(optimizer, profiling_enabled=True)
                    else:
                        optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    train_util.sample_images(
                        accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet
                    )

                    # Save model at specified steps
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                            train_util.save_sd_model_on_epoch_end_or_stepwise(
                                args,
                                False,
                                accelerator,
                                src_path,
                                save_stable_diffusion_format,
                                use_safetensors,
                                save_dtype,
                                epoch,
                                num_train_epochs,
                                global_step,
                                accelerator.unwrap_model(text_encoder),
                                accelerator.unwrap_model(unet),
                                vae,
                            )

                current_loss = loss.detach().item()  # Average loss, so batch size doesn't matter
                if args.logging_dir is not None:
                    logs = {"loss": current_loss}
                    train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=True)
                    accelerator.log(logs, step=global_step)

                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            if args.save_every_n_epochs is not None:
                if accelerator.is_main_process:
                    src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                    train_util.save_sd_model_on_epoch_end_or_stepwise(
                        args,
                        True,
                        accelerator,
                        src_path,
                        save_stable_diffusion_format,
                        use_safetensors,
                        save_dtype,
                        epoch,
                        num_train_epochs,
                        global_step,
                        accelerator.unwrap_model(text_encoder),
                        accelerator.unwrap_model(unet),
                        vae,
                    )

            train_util.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

    is_main_process = accelerator.is_main_process
    if is_main_process:
        unet = accelerator.unwrap_model(unet)
        text_encoder = accelerator.unwrap_model(text_encoder)

    accelerator.end_training()

    if is_main_process and (args.save_state or args.save_state_on_train_end):
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # Delete this as we'll need memory for subsequent operations

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        train_util.save_sd_model_on_train_end(
            args, src_path, save_stable_diffusion_format, use_safetensors, save_dtype, epoch, global_step, text_encoder, unet, vae
        )
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, False, True, True)
    train_util.add_training_arguments(parser, False)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--diffusers_xformers", action="store_true", help="use xformers by diffusers"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder")
    parser.add_argument(
        "--learning_rate_te",
        type=float,
        default=None,
        help="learning rate for text encoder, default is same as unet",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE)",
    )
    parser.add_argument(
        "--optimizer_profiling",
        action="store_true",
        help="Enable PyTorch profiler for optimizer step (SPARKLES only)",
    )
    parser.add_argument(
        "--profiler_output",
        type=str,
        default=None,
        help="If set, enables PyTorch profiler and exports trace to this file (e.g. trace.json or a directory for TensorBoard)",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)
