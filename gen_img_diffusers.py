"""
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
"""

import itertools
import json
from typing import Any, List, NamedTuple, Optional, Tuple, Union, Callable
import glob
import importlib
import inspect
import time
import zipfile
from diffusers.utils import deprecate
from diffusers.configuration_utils import FrozenDict
import argparse
import math
import os
import random
import re

import diffusers
import numpy as np

import torch
from library.device_utils import init_ipex, clean_memory, get_preferred_device
init_ipex()

import torchvision
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    # UNet2DConditionModel,
    StableDiffusionPipeline,
)
from einops import rearrange
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPTextConfig
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import library.model_util as model_util
import library.train_util as train_util
from networks.lora import LoRANetwork
import tools.original_control_net as original_control_net
from tools.original_control_net import ControlNetInfo
from library.original_unet import UNet2DConditionModel, InferUNet2DConditionModel
from library.original_unet import FlashAttentionFunction
from library.utils import GradualLatent, EulerAncestralDiscreteSchedulerGL

from XTI_hijack import unet_forward_XTI, downblock_forward_XTI, upblock_forward_XTI
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

# その他の設定
LATENT_CHANNELS = 4
DOWNSAMPLING_FACTOR = 8

# CLIP_ID_L14_336 = "openai/clip-vit-large-patch14-336"

# CLIP guided SD関連
CLIP_MODEL_PATH = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
FEATURE_EXTRACTOR_SIZE = (224, 224)
FEATURE_EXTRACTOR_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
FEATURE_EXTRACTOR_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

VGG16_IMAGE_MEAN = [0.485, 0.456, 0.406]
VGG16_IMAGE_STD = [0.229, 0.224, 0.225]
VGG16_INPUT_RESIZE_DIV = 4

# CLIP特徴量の取得時にcutoutを使うか：使う場合にはソースを書き換えてください
NUM_CUTOUTS = 4
USE_CUTOUTS = False

# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""


def replace_unet_modules(unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        logger.info("Enable memory efficient attention for U-Net")

        # これはDiffusersのU-Netではなく自前のU-Netなので置き換えなくても良い
        unet.set_use_memory_efficient_attention(False, True)
    elif xformers:
        logger.info("Enable xformers for U-Net")
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("No xformers / xformersがインストールされていないようです")

        unet.set_use_memory_efficient_attention(True, False)
    elif sdpa:
        logger.info("Enable SDPA for U-Net")
        unet.set_use_memory_efficient_attention(False, False)
        unet.set_use_sdpa(True)


# TODO common train_util.py
def replace_vae_modules(vae: diffusers.models.AutoencoderKL, mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        replace_vae_attn_to_memory_efficient()
    elif xformers:
        replace_vae_attn_to_xformers()
    elif sdpa:
        replace_vae_attn_to_sdpa()


def replace_vae_attn_to_memory_efficient():
    logger.info("VAE Attention.forward has been replaced to FlashAttention (not xformers)")
    flash_func = FlashAttentionFunction

    def forward_flash_attn(self, hidden_states, **kwargs):
        q_bucket_size = 512
        k_bucket_size = 1024

        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.to_q(hidden_states)
        key_proj = self.to_k(hidden_states)
        value_proj = self.to_v(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (query_proj, key_proj, value_proj)
        )

        out = flash_func.apply(query_proj, key_proj, value_proj, None, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, "b h n d -> b n (h d)")

        # compute next hidden_states
        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    def forward_flash_attn_0_14(self, hidden_states, **kwargs):
        if not hasattr(self, "to_q"):
            self.to_q = self.query
            self.to_k = self.key
            self.to_v = self.value
            self.to_out = [self.proj_attn, torch.nn.Identity()]
            self.heads = self.num_heads
        return forward_flash_attn(self, hidden_states, **kwargs)

    if diffusers.__version__ < "0.15.0":
        diffusers.models.attention.AttentionBlock.forward = forward_flash_attn_0_14
    else:
        diffusers.models.attention_processor.Attention.forward = forward_flash_attn


def replace_vae_attn_to_xformers():
    logger.info("VAE: Attention.forward has been replaced to xformers")
    import xformers.ops

    def forward_xformers(self, hidden_states, **kwargs):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.to_q(hidden_states)
        key_proj = self.to_k(hidden_states)
        value_proj = self.to_v(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (query_proj, key_proj, value_proj)
        )

        query_proj = query_proj.contiguous()
        key_proj = key_proj.contiguous()
        value_proj = value_proj.contiguous()
        out = xformers.ops.memory_efficient_attention(query_proj, key_proj, value_proj, attn_bias=None)

        out = rearrange(out, "b h n d -> b n (h d)")

        # compute next hidden_states
        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    def forward_xformers_0_14(self, hidden_states, **kwargs):
        if not hasattr(self, "to_q"):
            self.to_q = self.query
            self.to_k = self.key
            self.to_v = self.value
            self.to_out = [self.proj_attn, torch.nn.Identity()]
            self.heads = self.num_heads
        return forward_xformers(self, hidden_states, **kwargs)

    if diffusers.__version__ < "0.15.0":
        diffusers.models.attention.AttentionBlock.forward = forward_xformers_0_14
    else:
        diffusers.models.attention_processor.Attention.forward = forward_xformers


def replace_vae_attn_to_sdpa():
    logger.info("VAE: Attention.forward has been replaced to sdpa")

    def forward_sdpa(self, hidden_states, **kwargs):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.to_q(hidden_states)
        key_proj = self.to_k(hidden_states)
        value_proj = self.to_v(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b n h d", h=self.heads), (query_proj, key_proj, value_proj)
        )

        out = torch.nn.functional.scaled_dot_product_attention(
            query_proj, key_proj, value_proj, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        out = rearrange(out, "b n h d -> b n (h d)")

        # compute next hidden_states
        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    def forward_sdpa_0_14(self, hidden_states, **kwargs):
        if not hasattr(self, "to_q"):
            self.to_q = self.query
            self.to_k = self.key
            self.to_v = self.value
            self.to_out = [self.proj_attn, torch.nn.Identity()]
            self.heads = self.num_heads
        return forward_sdpa(self, hidden_states, **kwargs)

    if diffusers.__version__ < "0.15.0":
        diffusers.models.attention.AttentionBlock.forward = forward_sdpa_0_14
    else:
        diffusers.models.attention_processor.Attention.forward = forward_sdpa


# endregion

# region 画像生成の本体：lpw_stable_diffusion.py （ASL）からコピーして修正
# https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion.py
# Pipelineだけ独立して使えないのと機能追加するのとでコピーして修正


class PipelineLike:
    r"""
    Pipeline for text-to-image generation using Stable Diffusion without tokens length limit, and support parsing
    weighting in prompt.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        device,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: InferUNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        clip_skip: int,
        clip_model: CLIPModel,
        clip_guidance_scale: float,
        clip_image_guidance_scale: float,
        vgg16_model: torchvision.models.VGG,
        vgg16_guidance_scale: float,
        vgg16_layer_no: int,
        # safety_checker: StableDiffusionSafetyChecker,
        # feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.device = device
        self.clip_skip = clip_skip

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.safety_checker = None

        # Textual Inversion
        self.token_replacements = {}

        # XTI
        self.token_replacements_XTI = {}

        # CLIP guidance
        self.clip_guidance_scale = clip_guidance_scale
        self.clip_image_guidance_scale = clip_image_guidance_scale
        self.clip_model = clip_model
        self.normalize = transforms.Normalize(mean=FEATURE_EXTRACTOR_IMAGE_MEAN, std=FEATURE_EXTRACTOR_IMAGE_STD)
        self.make_cutouts = MakeCutouts(FEATURE_EXTRACTOR_SIZE)

        # VGG16 guidance
        self.vgg16_guidance_scale = vgg16_guidance_scale
        if self.vgg16_guidance_scale > 0.0:
            return_layers = {f"{vgg16_layer_no}": "feat"}
            self.vgg16_feat_model = torchvision.models._utils.IntermediateLayerGetter(
                vgg16_model.features, return_layers=return_layers
            )
            self.vgg16_normalize = transforms.Normalize(mean=VGG16_IMAGE_MEAN, std=VGG16_IMAGE_STD)

        # ControlNet
        self.control_nets: List[ControlNetInfo] = []
        self.control_net_enabled = True  # control_netsが空ならTrueでもFalseでもControlNetは動作しない

        self.gradual_latent: GradualLatent = None

    # Textual Inversion
    def add_token_replacement(self, target_token_id, rep_token_ids):
        self.token_replacements[target_token_id] = rep_token_ids

    def set_enable_control_net(self, en: bool):
        self.control_net_enabled = en

    def replace_token(self, tokens, layer=None):
        new_tokens = []
        for token in tokens:
            if token in self.token_replacements:
                replacer_ = self.token_replacements[token]
                if layer:
                    replacer = []
                for r in replacer_:
                    if r in self.token_replacements_XTI:
                        replacer.append(self.token_replacements_XTI[r][layer])
                    else:
                        replacer = replacer_
                new_tokens.extend(replacer)
            else:
                new_tokens.append(token)
        return new_tokens

    def add_token_replacement_XTI(self, target_token_id, rep_token_ids):
        self.token_replacements_XTI[target_token_id] = rep_token_ids

    def set_control_nets(self, ctrl_nets):
        self.control_nets = ctrl_nets

    def set_gradual_latent(self, gradual_latent):
        if gradual_latent is None:
            logger.info("gradual_latent is disabled")
            self.gradual_latent = None
        else:
            logger.info(f"gradual_latent is enabled: {gradual_latent}")
            self.gradual_latent = gradual_latent  # (ds_ratio, start_timesteps, every_n_steps, ratio_step)

    # region xformersとか使う部分：独自に書き換えるので関係なし

    def enable_xformers_memory_efficient_attention(self):
        r"""
        Enable memory efficient attention as implemented in xformers.
        When this option is enabled, you should observe lower GPU memory usage and a potential speed up at inference
        time. Speed up at training time is not guaranteed.
        Warning: When Memory Efficient Attention and Sliced attention are both enabled, the Memory Efficient Attention
        is used.
        """
        self.unet.set_use_memory_efficient_attention_xformers(True)

    def disable_xformers_memory_efficient_attention(self):
        r"""
        Disable memory efficient attention as implemented in xformers.
        """
        self.unet.set_use_memory_efficient_attention_xformers(False)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.
        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def enable_sequential_cpu_offload(self):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        # accelerateが必要になるのでとりあえず省略
        raise NotImplementedError("cpu_offload is omitted.")
        # if is_accelerate_available():
        #   from accelerate import cpu_offload
        # else:
        #   raise ImportError("Please install accelerate via `pip install accelerate`")

        # device = self.device

        # for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.safety_checker]:
        #   if cpu_offloaded_model is not None:
        #     cpu_offload(cpu_offloaded_model, device)

    # endregion

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        init_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_scale: float = None,
        strength: float = 0.8,
        # num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        vae_batch_size: float = None,
        return_latents: bool = False,
        # return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: Optional[int] = 1,
        img2img_noise=None,
        clip_prompts=None,
        clip_guide_images=None,
        networks: Optional[List[LoRANetwork]] = None,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            is_cancelled_callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. If the function returns
                `True`, the inference will be cancelled.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            `None` if cancelled by `is_cancelled_callback`,
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        num_images_per_prompt = 1  # fixed

        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        reginonal_network = " AND " in prompt[0]

        vae_batch_size = (
            batch_size
            if vae_batch_size is None
            else (int(vae_batch_size) if vae_batch_size >= 1 else max(1, int(batch_size * vae_batch_size)))
        )

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type" f" {type(callback_steps)}."
            )

        # get prompt text embeddings

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if not do_classifier_free_guidance and negative_scale is not None:
            logger.warning(f"negative_scale is ignored if guidance scalle <= 1.0")
            negative_scale = None

        # get unconditional embeddings for classifier free guidance
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        if not self.token_replacements_XTI:
            text_embeddings, uncond_embeddings, prompt_tokens = get_weighted_text_embeddings(
                pipe=self,
                prompt=prompt,
                uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
                max_embeddings_multiples=max_embeddings_multiples,
                clip_skip=self.clip_skip,
                **kwargs,
            )

        if negative_scale is not None:
            _, real_uncond_embeddings, _ = get_weighted_text_embeddings(
                pipe=self,
                prompt=prompt,  # こちらのトークン長に合わせてuncondを作るので75トークン超で必須
                uncond_prompt=[""] * batch_size,
                max_embeddings_multiples=max_embeddings_multiples,
                clip_skip=self.clip_skip,
                **kwargs,
            )

        if self.token_replacements_XTI:
            text_embeddings_concat = []
            for layer in [
                "IN01",
                "IN02",
                "IN04",
                "IN05",
                "IN07",
                "IN08",
                "MID",
                "OUT03",
                "OUT04",
                "OUT05",
                "OUT06",
                "OUT07",
                "OUT08",
                "OUT09",
                "OUT10",
                "OUT11",
            ]:
                text_embeddings, uncond_embeddings, prompt_tokens = get_weighted_text_embeddings(
                    pipe=self,
                    prompt=prompt,
                    uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
                    max_embeddings_multiples=max_embeddings_multiples,
                    clip_skip=self.clip_skip,
                    layer=layer,
                    **kwargs,
                )
                if do_classifier_free_guidance:
                    if negative_scale is None:
                        text_embeddings_concat.append(torch.cat([uncond_embeddings, text_embeddings]))
                    else:
                        text_embeddings_concat.append(torch.cat([uncond_embeddings, text_embeddings, real_uncond_embeddings]))
                text_embeddings = torch.stack(text_embeddings_concat)
        else:
            if do_classifier_free_guidance:
                if negative_scale is None:
                    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                else:
                    text_embeddings = torch.cat([uncond_embeddings, text_embeddings, real_uncond_embeddings])

        # CLIP guidanceで使用するembeddingsを取得する
        if self.clip_guidance_scale > 0:
            clip_text_input = prompt_tokens
            if clip_text_input.shape[1] > self.tokenizer.model_max_length:
                # TODO 75文字を超えたら警告を出す？
                logger.info(f"trim text input {clip_text_input.shape}")
                clip_text_input = torch.cat(
                    [clip_text_input[:, : self.tokenizer.model_max_length - 1], clip_text_input[:, -1].unsqueeze(1)], dim=1
                )
                logger.info(f"trimmed {clip_text_input.shape}")

            for i, clip_prompt in enumerate(clip_prompts):
                if clip_prompt is not None:  # clip_promptがあれば上書きする
                    clip_text_input[i] = self.tokenizer(
                        clip_prompt,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(self.device)

            text_embeddings_clip = self.clip_model.get_text_features(clip_text_input)
            text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(p=2, dim=-1, keepdim=True)  # prompt複数件でもOK

        if (
            self.clip_image_guidance_scale > 0
            or self.vgg16_guidance_scale > 0
            and clip_guide_images is not None
            or self.control_nets
        ):
            if isinstance(clip_guide_images, PIL.Image.Image):
                clip_guide_images = [clip_guide_images]

            if self.clip_image_guidance_scale > 0:
                clip_guide_images = [preprocess_guide_image(im) for im in clip_guide_images]
                clip_guide_images = torch.cat(clip_guide_images, dim=0)

                clip_guide_images = self.normalize(clip_guide_images).to(self.device).to(text_embeddings.dtype)
                image_embeddings_clip = self.clip_model.get_image_features(clip_guide_images)
                image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
                if len(image_embeddings_clip) == 1:
                    image_embeddings_clip = image_embeddings_clip.repeat((batch_size, 1, 1, 1))
            elif self.vgg16_guidance_scale > 0:
                size = (width // VGG16_INPUT_RESIZE_DIV, height // VGG16_INPUT_RESIZE_DIV)  # とりあえず1/4に（小さいか?）
                clip_guide_images = [preprocess_vgg16_guide_image(im, size) for im in clip_guide_images]
                clip_guide_images = torch.cat(clip_guide_images, dim=0)

                clip_guide_images = self.vgg16_normalize(clip_guide_images).to(self.device).to(text_embeddings.dtype)
                image_embeddings_vgg16 = self.vgg16_feat_model(clip_guide_images)["feat"]
                if len(image_embeddings_vgg16) == 1:
                    image_embeddings_vgg16 = image_embeddings_vgg16.repeat((batch_size, 1, 1, 1))
            else:
                # ControlNetのhintにguide imageを流用する
                # 前処理はControlNet側で行う
                pass

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, self.device)

        latents_dtype = text_embeddings.dtype
        init_latents_orig = None
        mask = None

        if init_image is None:
            # get the initial random noise unless the user supplied it

            # Unlike in other pipelines, latents need to be generated in the target device
            # for 1-to-1 results reproducibility with the CompVis implementation.
            # However this currently doesn't work in `mps`.
            latents_shape = (
                batch_size * num_images_per_prompt,
                self.unet.in_channels,
                height // 8,
                width // 8,
            )

            if latents is None:
                if self.device.type == "mps":
                    # randn does not exist on mps
                    latents = torch.randn(
                        latents_shape,
                        generator=generator,
                        device="cpu",
                        dtype=latents_dtype,
                    ).to(self.device)
                else:
                    latents = torch.randn(
                        latents_shape,
                        generator=generator,
                        device=self.device,
                        dtype=latents_dtype,
                    )
            else:
                if latents.shape != latents_shape:
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
                latents = latents.to(self.device)

            timesteps = self.scheduler.timesteps.to(self.device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        else:
            # image to tensor
            if isinstance(init_image, PIL.Image.Image):
                init_image = [init_image]
            if isinstance(init_image[0], PIL.Image.Image):
                init_image = [preprocess_image(im) for im in init_image]
                init_image = torch.cat(init_image)
            if isinstance(init_image, list):
                init_image = torch.stack(init_image)

            # mask image to tensor
            if mask_image is not None:
                if isinstance(mask_image, PIL.Image.Image):
                    mask_image = [mask_image]
                if isinstance(mask_image[0], PIL.Image.Image):
                    mask_image = torch.cat([preprocess_mask(im) for im in mask_image])  # H*W, 0 for repaint

            # encode the init image into latents and scale the latents
            init_image = init_image.to(device=self.device, dtype=latents_dtype)
            if init_image.size()[-2:] == (height // 8, width // 8):
                init_latents = init_image
            else:
                if vae_batch_size >= batch_size:
                    init_latent_dist = self.vae.encode(init_image).latent_dist
                    init_latents = init_latent_dist.sample(generator=generator)
                else:
                    clean_memory()
                    init_latents = []
                    for i in tqdm(range(0, min(batch_size, len(init_image)), vae_batch_size)):
                        init_latent_dist = self.vae.encode(
                            init_image[i : i + vae_batch_size] if vae_batch_size > 1 else init_image[i].unsqueeze(0)
                        ).latent_dist
                        init_latents.append(init_latent_dist.sample(generator=generator))
                    init_latents = torch.cat(init_latents)

                init_latents = 0.18215 * init_latents

            if len(init_latents) == 1:
                init_latents = init_latents.repeat((batch_size, 1, 1, 1))
            init_latents_orig = init_latents

            # preprocess mask
            if mask_image is not None:
                mask = mask_image.to(device=self.device, dtype=latents_dtype)
                if len(mask) == 1:
                    mask = mask.repeat((batch_size, 1, 1, 1))

                # check sizes
                if not mask.shape == init_latents.shape:
                    raise ValueError("The mask and init_image should be the same size!")

            # get the original timestep using init_timestep
            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)

            timesteps = self.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps] * batch_size * num_images_per_prompt, device=self.device)

            # add noise to latents using the timesteps
            latents = self.scheduler.add_noise(init_latents, img2img_noise, timesteps)

            t_start = max(num_inference_steps - init_timestep + offset, 0)
            timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        num_latent_input = (3 if negative_scale is not None else 2) if do_classifier_free_guidance else 1

        if self.control_nets:
            guided_hints = original_control_net.get_guided_hints(self.control_nets, num_latent_input, batch_size, clip_guide_images)

        if reginonal_network:
            num_sub_and_neg_prompts = len(text_embeddings) // batch_size
            # last subprompt and negative prompt
            text_emb_last = []
            for j in range(batch_size):
                text_emb_last.append(text_embeddings[(j + 1) * num_sub_and_neg_prompts - 2])
                text_emb_last.append(text_embeddings[(j + 1) * num_sub_and_neg_prompts - 1])
            text_emb_last = torch.stack(text_emb_last)
        else:
            text_emb_last = text_embeddings

        enable_gradual_latent = False
        if self.gradual_latent:
            if not hasattr(self.scheduler, "set_gradual_latent_params"):
                logger.info("gradual_latent is not supported for this scheduler. Ignoring.")
                logger.info(f'{self.scheduler.__class__.__name__}')
            else:
                enable_gradual_latent = True
                step_elapsed = 1000
                current_ratio = self.gradual_latent.ratio

                # first, we downscale the latents to the specified ratio / 最初に指定された比率にlatentsをダウンスケールする
                height, width = latents.shape[-2:]
                org_dtype = latents.dtype
                if org_dtype == torch.bfloat16:
                    latents = latents.float()
                latents = torch.nn.functional.interpolate(
                    latents, scale_factor=current_ratio, mode="bicubic", align_corners=False
                ).to(org_dtype)

                # apply unsharp mask / アンシャープマスクを適用する
                if self.gradual_latent.gaussian_blur_ksize:
                    latents = self.gradual_latent.apply_unshark_mask(latents)

        for i, t in enumerate(tqdm(timesteps)):
            resized_size = None
            if enable_gradual_latent:
                # gradually upscale the latents / latentsを徐々にアップスケールする
                if (
                    t < self.gradual_latent.start_timesteps
                    and current_ratio < 1.0
                    and step_elapsed >= self.gradual_latent.every_n_steps
                ):
                    current_ratio = min(current_ratio + self.gradual_latent.ratio_step, 1.0)
                    # make divisible by 8 because size of latents must be divisible at bottom of UNet
                    h = int(height * current_ratio) // 8 * 8
                    w = int(width * current_ratio) // 8 * 8
                    resized_size = (h, w)
                    self.scheduler.set_gradual_latent_params(resized_size, self.gradual_latent)
                    step_elapsed = 0
                else:
                    self.scheduler.set_gradual_latent_params(None, None)
                step_elapsed += 1

            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            if self.control_nets and self.control_net_enabled:
                noise_pred = original_control_net.call_unet_and_control_net(
                    i,
                    num_latent_input,
                    self.unet,
                    self.control_nets,
                    guided_hints,
                    i / len(timesteps),
                    latent_model_input,
                    t,
                    text_embeddings,
                    text_emb_last,
                ).sample
            else:
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                if negative_scale is None:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred_negative, noise_pred_text, noise_pred_uncond = noise_pred.chunk(
                        num_latent_input
                    )  # uncond is real uncond
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        - negative_scale * (noise_pred_negative - noise_pred_uncond)
                    )

            # perform clip guidance
            if self.clip_guidance_scale > 0 or self.clip_image_guidance_scale > 0 or self.vgg16_guidance_scale > 0:
                text_embeddings_for_guidance = (
                    text_embeddings.chunk(num_latent_input)[1] if do_classifier_free_guidance else text_embeddings
                )

                if self.clip_guidance_scale > 0:
                    noise_pred, latents = self.cond_fn(
                        latents,
                        t,
                        i,
                        text_embeddings_for_guidance,
                        noise_pred,
                        text_embeddings_clip,
                        self.clip_guidance_scale,
                        NUM_CUTOUTS,
                        USE_CUTOUTS,
                    )
                if self.clip_image_guidance_scale > 0 and clip_guide_images is not None:
                    noise_pred, latents = self.cond_fn(
                        latents,
                        t,
                        i,
                        text_embeddings_for_guidance,
                        noise_pred,
                        image_embeddings_clip,
                        self.clip_image_guidance_scale,
                        NUM_CUTOUTS,
                        USE_CUTOUTS,
                    )
                if self.vgg16_guidance_scale > 0 and clip_guide_images is not None:
                    noise_pred, latents = self.cond_fn_vgg16(
                        latents, t, i, text_embeddings_for_guidance, noise_pred, image_embeddings_vgg16, self.vgg16_guidance_scale
                    )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if mask is not None:
                # masking
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, img2img_noise, torch.tensor([t]))
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

            # call the callback, if provided
            if i % callback_steps == 0:
                if callback is not None:
                    callback(i, t, latents)
                if is_cancelled_callback is not None and is_cancelled_callback():
                    return None

        if return_latents:
            return (latents, False)

        latents = 1 / 0.18215 * latents
        if vae_batch_size >= batch_size:
            image = self.vae.decode(latents).sample
        else:
            clean_memory()
            images = []
            for i in tqdm(range(0, batch_size, vae_batch_size)):
                images.append(
                    self.vae.decode(latents[i : i + vae_batch_size] if vae_batch_size > 1 else latents[i].unsqueeze(0)).sample
                )
            image = torch.cat(images)

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype),
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            # image = self.numpy_to_pil(image)
            image = (image * 255).round().astype("uint8")
            image = [Image.fromarray(im) for im in image]

        # if not return_dict:
        return (image, has_nsfw_concept)

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def text2img(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function for text-to-image generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

    def img2img(
        self,
        init_image: Union[torch.FloatTensor, PIL.Image.Image],
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function for image-to-image generation.
        Args:
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

    def inpaint(
        self,
        init_image: Union[torch.FloatTensor, PIL.Image.Image],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image],
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function for inpaint.
        Args:
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. This is the image whose masked region will be inpainted.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to inpaint the masked area. Must be between 0 and 1. When `strength`
                is 1, the denoising process will be run on the masked area for the full number of iterations specified
                in `num_inference_steps`. `init_image` will be used as a reference for the masked area, adding more
                noise to that region the larger the `strength`. If `strength` is 0, no inpainting will occur.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The reference number of denoising steps. More denoising steps usually lead to a higher quality image at
                the expense of slower inference. This parameter will be modulated by `strength`, as explained above.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

    # CLIP guidance StableDiffusion
    # copy from https://github.com/huggingface/diffusers/blob/main/examples/community/clip_guided_stable_diffusion.py

    # バッチを分解して1件ずつ処理する
    def cond_fn(
        self,
        latents,
        timestep,
        index,
        text_embeddings,
        noise_pred_original,
        guide_embeddings_clip,
        clip_guidance_scale,
        num_cutouts,
        use_cutouts=True,
    ):
        if len(latents) == 1:
            return self.cond_fn1(
                latents,
                timestep,
                index,
                text_embeddings,
                noise_pred_original,
                guide_embeddings_clip,
                clip_guidance_scale,
                num_cutouts,
                use_cutouts,
            )

        noise_pred = []
        cond_latents = []
        for i in range(len(latents)):
            lat1 = latents[i].unsqueeze(0)
            tem1 = text_embeddings[i].unsqueeze(0)
            npo1 = noise_pred_original[i].unsqueeze(0)
            gem1 = guide_embeddings_clip[i].unsqueeze(0)
            npr1, cla1 = self.cond_fn1(lat1, timestep, index, tem1, npo1, gem1, clip_guidance_scale, num_cutouts, use_cutouts)
            noise_pred.append(npr1)
            cond_latents.append(cla1)

        noise_pred = torch.cat(noise_pred)
        cond_latents = torch.cat(cond_latents)
        return noise_pred, cond_latents

    @torch.enable_grad()
    def cond_fn1(
        self,
        latents,
        timestep,
        index,
        text_embeddings,
        noise_pred_original,
        guide_embeddings_clip,
        clip_guidance_scale,
        num_cutouts,
        use_cutouts=True,
    ):
        latents = latents.detach().requires_grad_()

        if isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
            latent_model_input = latents / ((sigma**2 + 1) ** 0.5)
        else:
            latent_model_input = latents

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler)):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            # compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

            fac = torch.sqrt(beta_prod_t)
            sample = pred_original_sample * (fac) + latents * (1 - fac)
        elif isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            sample = latents - sigma * noise_pred
        else:
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

        sample = 1 / 0.18215 * sample
        image = self.vae.decode(sample).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        if use_cutouts:
            image = self.make_cutouts(image, num_cutouts)
        else:
            image = transforms.Resize(FEATURE_EXTRACTOR_SIZE)(image)
        image = self.normalize(image).to(latents.dtype)

        image_embeddings_clip = self.clip_model.get_image_features(image)
        image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)

        if use_cutouts:
            dists = spherical_dist_loss(image_embeddings_clip, guide_embeddings_clip)
            dists = dists.view([num_cutouts, sample.shape[0], -1])
            loss = dists.sum(2).mean(0).sum() * clip_guidance_scale
        else:
            # バッチサイズが複数だと正しく動くかわからない
            loss = spherical_dist_loss(image_embeddings_clip, guide_embeddings_clip).mean() * clip_guidance_scale

        grads = -torch.autograd.grad(loss, latents)[0]

        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents.detach() + grads * (sigma**2)
            noise_pred = noise_pred_original
        else:
            noise_pred = noise_pred_original - torch.sqrt(beta_prod_t) * grads
        return noise_pred, latents

    # バッチを分解して一件ずつ処理する
    def cond_fn_vgg16(self, latents, timestep, index, text_embeddings, noise_pred_original, guide_embeddings, guidance_scale):
        if len(latents) == 1:
            return self.cond_fn_vgg16_b1(
                latents, timestep, index, text_embeddings, noise_pred_original, guide_embeddings, guidance_scale
            )

        noise_pred = []
        cond_latents = []
        for i in range(len(latents)):
            lat1 = latents[i].unsqueeze(0)
            tem1 = text_embeddings[i].unsqueeze(0)
            npo1 = noise_pred_original[i].unsqueeze(0)
            gem1 = guide_embeddings[i].unsqueeze(0)
            npr1, cla1 = self.cond_fn_vgg16_b1(lat1, timestep, index, tem1, npo1, gem1, guidance_scale)
            noise_pred.append(npr1)
            cond_latents.append(cla1)

        noise_pred = torch.cat(noise_pred)
        cond_latents = torch.cat(cond_latents)
        return noise_pred, cond_latents

    # 1件だけ処理する
    @torch.enable_grad()
    def cond_fn_vgg16_b1(self, latents, timestep, index, text_embeddings, noise_pred_original, guide_embeddings, guidance_scale):
        latents = latents.detach().requires_grad_()

        if isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
            latent_model_input = latents / ((sigma**2 + 1) ** 0.5)
        else:
            latent_model_input = latents

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler)):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            # compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

            fac = torch.sqrt(beta_prod_t)
            sample = pred_original_sample * (fac) + latents * (1 - fac)
        elif isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            sample = latents - sigma * noise_pred
        else:
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

        sample = 1 / 0.18215 * sample
        image = self.vae.decode(sample).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = transforms.Resize((image.shape[-2] // VGG16_INPUT_RESIZE_DIV, image.shape[-1] // VGG16_INPUT_RESIZE_DIV))(image)
        image = self.vgg16_normalize(image).to(latents.dtype)

        image_embeddings = self.vgg16_feat_model(image)["feat"]

        # バッチサイズが複数だと正しく動くかわからない
        loss = (
            (image_embeddings - guide_embeddings) ** 2
        ).mean() * guidance_scale  # MSE style transferでコンテンツの損失はMSEなので

        grads = -torch.autograd.grad(loss, latents)[0]
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents.detach() + grads * (sigma**2)
            noise_pred = noise_pred_original
        else:
            noise_pred = noise_pred_original - torch.sqrt(beta_prod_t) * grads
        return noise_pred, latents


class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cut_power=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cut_power = cut_power

    def forward(self, pixel_values, num_cutouts):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = pixel_values[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(torch.nn.functional.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    # keep break as separate token
    text = text.replace("BREAK", "\\BREAK\\")

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1] and res[i][0].strip() != "BREAK" and res[i + 1][0].strip() != "BREAK":
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(pipe: PipelineLike, prompt: List[str], max_length: int, layer=None):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.
    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False

    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            if word.strip() == "BREAK":
                # pad until next multiple of tokenizer's max token length
                pad_len = pipe.tokenizer.model_max_length - (len(text_token) % pipe.tokenizer.model_max_length)
                logger.info(f"BREAK pad_len: {pad_len}")
                for i in range(pad_len):
                    # v2のときEOSをつけるべきかどうかわからないぜ
                    # if i == 0:
                    #     text_token.append(pipe.tokenizer.eos_token_id)
                    # else:
                    text_token.append(pipe.tokenizer.pad_token_id)
                    text_weight.append(1.0)
                continue

            # tokenize and discard the starting and the ending token
            token = pipe.tokenizer(word).input_ids[1:-1]

            token = pipe.replace_token(token, layer=layer)

            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        logger.warning("Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] + [pad] * (max_length - 2 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    pipe: PipelineLike,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            if pad == eos:  # v1
                text_input_chunk[:, -1] = text_input[0, -1]
            else:  # v2
                for j in range(len(text_input_chunk)):
                    if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:  # 最後に普通の文字がある
                        text_input_chunk[j, -1] = eos
                    if text_input_chunk[j, 1] == pad:  # BOSだけであとはPAD
                        text_input_chunk[j, 1] = eos

            if clip_skip is None or clip_skip == 1:
                text_embedding = pipe.text_encoder(text_input_chunk)[0]
            else:
                enc_out = pipe.text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
                text_embedding = enc_out["hidden_states"][-clip_skip]
                text_embedding = pipe.text_encoder.text_model.final_layer_norm(text_embedding)

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        if clip_skip is None or clip_skip == 1:
            text_embeddings = pipe.text_encoder(text_input)[0]
        else:
            enc_out = pipe.text_encoder(text_input, output_hidden_states=True, return_dict=True)
            text_embeddings = enc_out["hidden_states"][-clip_skip]
            text_embeddings = pipe.text_encoder.text_model.final_layer_norm(text_embeddings)
    return text_embeddings


def get_weighted_text_embeddings(
    pipe: PipelineLike,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 1,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
    clip_skip=None,
    layer=None,
    **kwargs,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.
    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.
    Args:
        pipe (`DiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `1`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    # split the prompts with "AND". each prompt must have the same number of splits
    new_prompts = []
    for p in prompt:
        new_prompts.extend(p.split(" AND "))
    prompt = new_prompts

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt, max_length - 2, layer=layer)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, uncond_prompt, max_length - 2, layer=layer)
    else:
        prompt_tokens = [token[1:-1] for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1] for token in pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    pad = pipe.tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.tokenizer.model_max_length,
        )
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=pipe.device)
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.tokenizer.model_max_length,
            clip_skip,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
        )
        uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device)

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    # →全体でいいんじゃないかな
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings, prompt_tokens
    return text_embeddings, None, prompt_tokens


def preprocess_guide_image(image):
    image = image.resize(FEATURE_EXTRACTOR_SIZE, resample=Image.NEAREST)  # cond_fnと合わせる
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)  # nchw
    image = torch.from_numpy(image)
    return image  # 0 to 1


# VGG16の入力は任意サイズでよいので入力画像を適宜リサイズする
def preprocess_vgg16_guide_image(image, size):
    image = image.resize(size, resample=Image.NEAREST)  # cond_fnと合わせる
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)  # nchw
    image = torch.from_numpy(image)
    return image  # 0 to 1


def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.BILINEAR)  # LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask


# regular expression for dynamic prompt:
# starts and ends with "{" and "}"
# contains at least one variant divided by "|"
# optional framgments divided by "$$" at start
# if the first fragment is "E" or "e", enumerate all variants
# if the second fragment is a number or two numbers, repeat the variants in the range
# if the third fragment is a string, use it as a separator

RE_DYNAMIC_PROMPT = re.compile(r"\{((e|E)\$\$)?(([\d\-]+)\$\$)?(([^\|\}]+?)\$\$)?(.+?((\|).+?)*?)\}")


def handle_dynamic_prompt_variants(prompt, repeat_count):
    founds = list(RE_DYNAMIC_PROMPT.finditer(prompt))
    if not founds:
        return [prompt]

    # make each replacement for each variant
    enumerating = False
    replacers = []
    for found in founds:
        # if "e$$" is found, enumerate all variants
        found_enumerating = found.group(2) is not None
        enumerating = enumerating or found_enumerating

        separator = ", " if found.group(6) is None else found.group(6)
        variants = found.group(7).split("|")

        # parse count range
        count_range = found.group(4)
        if count_range is None:
            count_range = [1, 1]
        else:
            count_range = count_range.split("-")
            if len(count_range) == 1:
                count_range = [int(count_range[0]), int(count_range[0])]
            elif len(count_range) == 2:
                count_range = [int(count_range[0]), int(count_range[1])]
            else:
                logger.warning(f"invalid count range: {count_range}")
                count_range = [1, 1]
            if count_range[0] > count_range[1]:
                count_range = [count_range[1], count_range[0]]
            if count_range[0] < 0:
                count_range[0] = 0
            if count_range[1] > len(variants):
                count_range[1] = len(variants)

        if found_enumerating:
            # make function to enumerate all combinations
            def make_replacer_enum(vari, cr, sep):
                def replacer():
                    values = []
                    for count in range(cr[0], cr[1] + 1):
                        for comb in itertools.combinations(vari, count):
                            values.append(sep.join(comb))
                    return values

                return replacer

            replacers.append(make_replacer_enum(variants, count_range, separator))
        else:
            # make function to choose random combinations
            def make_replacer_single(vari, cr, sep):
                def replacer():
                    count = random.randint(cr[0], cr[1])
                    comb = random.sample(vari, count)
                    return [sep.join(comb)]

                return replacer

            replacers.append(make_replacer_single(variants, count_range, separator))

    # make each prompt
    if not enumerating:
        # if not enumerating, repeat the prompt, replace each variant randomly
        prompts = []
        for _ in range(repeat_count):
            current = prompt
            for found, replacer in zip(founds, replacers):
                current = current.replace(found.group(0), replacer()[0], 1)
            prompts.append(current)
    else:
        # if enumerating, iterate all combinations for previous prompts
        prompts = [prompt]

        for found, replacer in zip(founds, replacers):
            if found.group(2) is not None:
                # make all combinations for existing prompts
                new_prompts = []
                for current in prompts:
                    replecements = replacer()
                    for replecement in replecements:
                        new_prompts.append(current.replace(found.group(0), replecement, 1))
                prompts = new_prompts

        for found, replacer in zip(founds, replacers):
            # make random selection for existing prompts
            if found.group(2) is None:
                for i in range(len(prompts)):
                    prompts[i] = prompts[i].replace(found.group(0), replacer()[0], 1)

    return prompts


# endregion


# def load_clip_l14_336(dtype):
#   logger.info(f"loading CLIP: {CLIP_ID_L14_336}")
#   text_encoder = CLIPTextModel.from_pretrained(CLIP_ID_L14_336, torch_dtype=dtype)
#   return text_encoder


class BatchDataBase(NamedTuple):
    # バッチ分割が必要ないデータ
    step: int
    prompt: str
    negative_prompt: str
    seed: int
    init_image: Any
    mask_image: Any
    clip_prompt: str
    guide_image: Any
    raw_prompt: str


class BatchDataExt(NamedTuple):
    # バッチ分割が必要なデータ
    width: int
    height: int
    steps: int
    scale: float
    negative_scale: float
    strength: float
    network_muls: Tuple[float]
    num_sub_prompts: int


class BatchData(NamedTuple):
    return_latents: bool
    base: BatchDataBase
    ext: BatchDataExt


def main(args):
    if args.fp16:
        dtype = torch.float16
    elif args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    highres_fix = args.highres_fix_scale is not None
    # assert not highres_fix or args.image_path is None, f"highres_fix doesn't work with img2img / highres_fixはimg2imgと同時に使えません"

    if args.v2 and args.clip_skip is not None:
        logger.warning("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

    # モデルを読み込む
    if not os.path.isfile(args.ckpt):  # ファイルがないならパターンで探し、一つだけ該当すればそれを使う
        files = glob.glob(args.ckpt)
        if len(files) == 1:
            args.ckpt = files[0]

    use_stable_diffusion_format = os.path.isfile(args.ckpt)
    if use_stable_diffusion_format:
        logger.info("load StableDiffusion checkpoint")
        text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.ckpt)
    else:
        logger.info("load Diffusers pretrained models")
        loading_pipe = StableDiffusionPipeline.from_pretrained(args.ckpt, safety_checker=None, torch_dtype=dtype)
        text_encoder = loading_pipe.text_encoder
        vae = loading_pipe.vae
        unet = loading_pipe.unet
        tokenizer = loading_pipe.tokenizer
        del loading_pipe

        # Diffusers U-Net to original U-Net
        original_unet = UNet2DConditionModel(
            unet.config.sample_size,
            unet.config.attention_head_dim,
            unet.config.cross_attention_dim,
            unet.config.use_linear_projection,
            unet.config.upcast_attention,
        )
        original_unet.load_state_dict(unet.state_dict())
        unet = original_unet
    unet: InferUNet2DConditionModel = InferUNet2DConditionModel(unet)

    # VAEを読み込む
    if args.vae is not None:
        vae = model_util.load_vae(args.vae, dtype)
        logger.info("additional VAE loaded")

    # # 置換するCLIPを読み込む
    # if args.replace_clip_l14_336:
    #   text_encoder = load_clip_l14_336(dtype)
    #   logger.info(f"large clip {CLIP_ID_L14_336} is loaded")

    if args.clip_guidance_scale > 0.0 or args.clip_image_guidance_scale:
        logger.info("prepare clip model")
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, torch_dtype=dtype)
    else:
        clip_model = None

    if args.vgg16_guidance_scale > 0.0:
        logger.info("prepare resnet model")
        vgg16_model = torchvision.models.vgg16(torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    else:
        vgg16_model = None

    # xformers、Hypernetwork対応
    if not args.diffusers_xformers:
        mem_eff = not (args.xformers or args.sdpa)
        replace_unet_modules(unet, mem_eff, args.xformers, args.sdpa)
        replace_vae_modules(vae, mem_eff, args.xformers, args.sdpa)

    # tokenizerを読み込む
    logger.info("loading tokenizer")
    if use_stable_diffusion_format:
        tokenizer = train_util.load_tokenizer(args)

    # schedulerを用意する
    sched_init_args = {}
    scheduler_num_noises_per_step = 1
    if args.sampler == "ddim":
        scheduler_cls = DDIMScheduler
        scheduler_module = diffusers.schedulers.scheduling_ddim
    elif args.sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
        scheduler_module = diffusers.schedulers.scheduling_ddpm
    elif args.sampler == "pndm":
        scheduler_cls = PNDMScheduler
        scheduler_module = diffusers.schedulers.scheduling_pndm
    elif args.sampler == "lms" or args.sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_lms_discrete
    elif args.sampler == "euler" or args.sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_euler_discrete
    elif args.sampler == "euler_a" or args.sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteSchedulerGL
        scheduler_module = diffusers.schedulers.scheduling_euler_ancestral_discrete
    elif args.sampler == "dpmsolver" or args.sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = args.sampler
        scheduler_module = diffusers.schedulers.scheduling_dpmsolver_multistep
    elif args.sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
        scheduler_module = diffusers.schedulers.scheduling_dpmsolver_singlestep
    elif args.sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_heun_discrete
    elif args.sampler == "dpm_2" or args.sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_discrete
    elif args.sampler == "dpm_2_a" or args.sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete
        scheduler_num_noises_per_step = 2

    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    # samplerの乱数をあらかじめ指定するための処理

    # replace randn
    class NoiseManager:
        def __init__(self):
            self.sampler_noises = None
            self.sampler_noise_index = 0

        def reset_sampler_noises(self, noises):
            self.sampler_noise_index = 0
            self.sampler_noises = noises

        def randn(self, shape, device=None, dtype=None, layout=None, generator=None):
            # logger.info(f"replacing {shape} {len(self.sampler_noises)} {self.sampler_noise_index}")
            if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises):
                noise = self.sampler_noises[self.sampler_noise_index]
                if shape != noise.shape:
                    noise = None
            else:
                noise = None

            if noise == None:
                logger.warning(f"unexpected noise request: {self.sampler_noise_index}, {shape}")
                noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)

            self.sampler_noise_index += 1
            return noise

    class TorchRandReplacer:
        def __init__(self, noise_manager):
            self.noise_manager = noise_manager

        def __getattr__(self, item):
            if item == "randn":
                return self.noise_manager.randn
            if hasattr(torch, item):
                return getattr(torch, item)
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    noise_manager = NoiseManager()
    if scheduler_module is not None:
        scheduler_module.torch = TorchRandReplacer(noise_manager)

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        logger.info("set clip_sample to True")
        scheduler.config.clip_sample = True

    # deviceを決定する
    device = get_preferred_device()

    # custom pipelineをコピったやつを生成する
    if args.vae_slices:
        from library.slicing_vae import SlicingAutoencoderKL

        sli_vae = SlicingAutoencoderKL(
            act_fn="silu",
            block_out_channels=(128, 256, 512, 512),
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            in_channels=3,
            latent_channels=4,
            layers_per_block=2,
            norm_num_groups=32,
            out_channels=3,
            sample_size=512,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            num_slices=args.vae_slices,
        )
        sli_vae.load_state_dict(vae.state_dict())  # vaeのパラメータをコピーする
        vae = sli_vae
        del sli_vae
    vae.to(dtype).to(device)
    vae.eval()

    text_encoder.to(dtype).to(device)
    unet.to(dtype).to(device)

    text_encoder.eval()
    unet.eval()

    if clip_model is not None:
        clip_model.to(dtype).to(device)
        clip_model.eval()
    if vgg16_model is not None:
        vgg16_model.to(dtype).to(device)
        vgg16_model.eval()

    # networkを組み込む
    if args.network_module:
        networks = []
        network_default_muls = []
        network_pre_calc = args.network_pre_calc

        # merge関連の引数を統合する
        if args.network_merge:
            network_merge = len(args.network_module)  # all networks are merged
        elif args.network_merge_n_models:
            network_merge = args.network_merge_n_models
        else:
            network_merge = 0

        for i, network_module in enumerate(args.network_module):
            logger.info(f"import network module: {network_module}")
            imported_module = importlib.import_module(network_module)

            network_mul = 1.0 if args.network_mul is None or len(args.network_mul) <= i else args.network_mul[i]

            net_kwargs = {}
            if args.network_args and i < len(args.network_args):
                network_args = args.network_args[i]
                # TODO escape special chars
                network_args = network_args.split(";")
                for net_arg in network_args:
                    key, value = net_arg.split("=")
                    net_kwargs[key] = value

            if args.network_weights is None or len(args.network_weights) <= i:
                raise ValueError("No weight. Weight is required.")

            network_weight = args.network_weights[i]
            logger.info(f"load network weights from: {network_weight}")

            if model_util.is_safetensors(network_weight) and args.network_show_meta:
                from safetensors.torch import safe_open

                with safe_open(network_weight, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is not None:
                    logger.info(f"metadata for: {network_weight}: {metadata}")

            network, weights_sd = imported_module.create_network_from_weights(
                network_mul, network_weight, vae, text_encoder, unet, for_inference=True, **net_kwargs
            )
            if network is None:
                return

            mergeable = network.is_mergeable()
            if network_merge and not mergeable:
                logger.warning("network is not mergiable. ignore merge option.")

            if not mergeable or i >= network_merge:
                # not merging
                network.apply_to(text_encoder, unet)
                info = network.load_state_dict(weights_sd, False)  # network.load_weightsを使うようにするとよい
                logger.info(f"weights are loaded: {info}")

                if args.opt_channels_last:
                    network.to(memory_format=torch.channels_last)
                network.to(dtype).to(device)

                if network_pre_calc:
                    logger.info("backup original weights")
                    network.backup_weights()

                networks.append(network)
                network_default_muls.append(network_mul)
            else:
                network.merge_to(text_encoder, unet, weights_sd, dtype, device)

    else:
        networks = []

    # upscalerの指定があれば取得する
    upscaler = None
    if args.highres_fix_upscaler:
        logger.info(f"import upscaler module {args.highres_fix_upscaler}")
        imported_module = importlib.import_module(args.highres_fix_upscaler)

        us_kwargs = {}
        if args.highres_fix_upscaler_args:
            for net_arg in args.highres_fix_upscaler_args.split(";"):
                key, value = net_arg.split("=")
                us_kwargs[key] = value

        logger.info("create upscaler")
        upscaler = imported_module.create_upscaler(**us_kwargs)
        upscaler.to(dtype).to(device)

    # ControlNetの処理
    control_nets: List[ControlNetInfo] = []
    if args.control_net_models:
        for i, model in enumerate(args.control_net_models):
            prep_type = None if not args.control_net_preps or len(args.control_net_preps) <= i else args.control_net_preps[i]
            weight = 1.0 if not args.control_net_weights or len(args.control_net_weights) <= i else args.control_net_weights[i]
            ratio = 1.0 if not args.control_net_ratios or len(args.control_net_ratios) <= i else args.control_net_ratios[i]

            ctrl_unet, ctrl_net = original_control_net.load_control_net(args.v2, unet, model)
            prep = original_control_net.load_preprocess(prep_type)
            control_nets.append(ControlNetInfo(ctrl_unet, ctrl_net, prep, weight, ratio))

    if args.opt_channels_last:
        logger.info(f"set optimizing: channels last")
        text_encoder.to(memory_format=torch.channels_last)
        vae.to(memory_format=torch.channels_last)
        unet.to(memory_format=torch.channels_last)
        if clip_model is not None:
            clip_model.to(memory_format=torch.channels_last)
        if networks:
            for network in networks:
                network.to(memory_format=torch.channels_last)
        if vgg16_model is not None:
            vgg16_model.to(memory_format=torch.channels_last)

        for cn in control_nets:
            cn.unet.to(memory_format=torch.channels_last)
            cn.net.to(memory_format=torch.channels_last)

    pipe = PipelineLike(
        device,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        args.clip_skip,
        clip_model,
        args.clip_guidance_scale,
        args.clip_image_guidance_scale,
        vgg16_model,
        args.vgg16_guidance_scale,
        args.vgg16_guidance_layer,
    )
    pipe.set_control_nets(control_nets)
    logger.info("pipeline is ready.")

    if args.diffusers_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    # Deep Shrink
    if args.ds_depth_1 is not None:
        unet.set_deep_shrink(args.ds_depth_1, args.ds_timesteps_1, args.ds_depth_2, args.ds_timesteps_2, args.ds_ratio)

    # Gradual Latent
    if args.gradual_latent_timesteps is not None:
        if args.gradual_latent_unsharp_params:
            us_params = args.gradual_latent_unsharp_params.split(",")
            us_ksize, us_sigma, us_strength = [float(v) for v in us_params[:3]]
            us_target_x = True if len(us_params) <= 3 else bool(int(us_params[3]))
            us_ksize = int(us_ksize)
        else:
            us_ksize, us_sigma, us_strength, us_target_x = None, None, None, None

        gradual_latent = GradualLatent(
            args.gradual_latent_ratio,
            args.gradual_latent_timesteps,
            args.gradual_latent_every_n_steps,
            args.gradual_latent_ratio_step,
            args.gradual_latent_s_noise,
            us_ksize,
            us_sigma,
            us_strength,
            us_target_x,
        )
        pipe.set_gradual_latent(gradual_latent)

    # Extended Textual Inversion および Textual Inversionを処理する
    if args.XTI_embeddings:
        diffusers.models.UNet2DConditionModel.forward = unet_forward_XTI
        diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D.forward = downblock_forward_XTI
        diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D.forward = upblock_forward_XTI

    if args.textual_inversion_embeddings:
        token_ids_embeds = []
        for embeds_file in args.textual_inversion_embeddings:
            if model_util.is_safetensors(embeds_file):
                from safetensors.torch import load_file

                data = load_file(embeds_file)
            else:
                data = torch.load(embeds_file, map_location="cpu")

            if "string_to_param" in data:
                data = data["string_to_param"]
            embeds = next(iter(data.values()))

            if type(embeds) != torch.Tensor:
                raise ValueError(
                    f"weight file does not contains Tensor / 重みファイルのデータがTensorではありません: {embeds_file}"
                )

            num_vectors_per_token = embeds.size()[0]
            token_string = os.path.splitext(os.path.basename(embeds_file))[0]
            token_strings = [token_string] + [f"{token_string}{i+1}" for i in range(num_vectors_per_token - 1)]

            # add new word to tokenizer, count is num_vectors_per_token
            num_added_tokens = tokenizer.add_tokens(token_strings)
            assert (
                num_added_tokens == num_vectors_per_token
            ), f"tokenizer has same word to token string (filename). please rename the file / 指定した名前（ファイル名）のトークンが既に存在します。ファイルをリネームしてください: {embeds_file}"

            token_ids = tokenizer.convert_tokens_to_ids(token_strings)
            logger.info(f"Textual Inversion embeddings `{token_string}` loaded. Tokens are added: {token_ids}")
            assert (
                min(token_ids) == token_ids[0] and token_ids[-1] == token_ids[0] + len(token_ids) - 1
            ), f"token ids is not ordered"
            assert len(tokenizer) - 1 == token_ids[-1], f"token ids is not end of tokenize: {len(tokenizer)}"

            if num_vectors_per_token > 1:
                pipe.add_token_replacement(token_ids[0], token_ids)

            token_ids_embeds.append((token_ids, embeds))

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for token_ids, embeds in token_ids_embeds:
            for token_id, embed in zip(token_ids, embeds):
                token_embeds[token_id] = embed

    if args.XTI_embeddings:
        XTI_layers = [
            "IN01",
            "IN02",
            "IN04",
            "IN05",
            "IN07",
            "IN08",
            "MID",
            "OUT03",
            "OUT04",
            "OUT05",
            "OUT06",
            "OUT07",
            "OUT08",
            "OUT09",
            "OUT10",
            "OUT11",
        ]
        token_ids_embeds_XTI = []
        for embeds_file in args.XTI_embeddings:
            if model_util.is_safetensors(embeds_file):
                from safetensors.torch import load_file

                data = load_file(embeds_file)
            else:
                data = torch.load(embeds_file, map_location="cpu")
            if set(data.keys()) != set(XTI_layers):
                raise ValueError("NOT XTI")
            embeds = torch.concat(list(data.values()))
            num_vectors_per_token = data["MID"].size()[0]

            token_string = os.path.splitext(os.path.basename(embeds_file))[0]
            token_strings = [token_string] + [f"{token_string}{i+1}" for i in range(num_vectors_per_token - 1)]

            # add new word to tokenizer, count is num_vectors_per_token
            num_added_tokens = tokenizer.add_tokens(token_strings)
            assert (
                num_added_tokens == num_vectors_per_token
            ), f"tokenizer has same word to token string (filename). please rename the file / 指定した名前（ファイル名）のトークンが既に存在します。ファイルをリネームしてください: {embeds_file}"

            token_ids = tokenizer.convert_tokens_to_ids(token_strings)
            logger.info(f"XTI embeddings `{token_string}` loaded. Tokens are added: {token_ids}")

            # if num_vectors_per_token > 1:
            pipe.add_token_replacement(token_ids[0], token_ids)

            token_strings_XTI = []
            for layer_name in XTI_layers:
                token_strings_XTI += [f"{t}_{layer_name}" for t in token_strings]
            tokenizer.add_tokens(token_strings_XTI)
            token_ids_XTI = tokenizer.convert_tokens_to_ids(token_strings_XTI)
            token_ids_embeds_XTI.append((token_ids_XTI, embeds))
            for t in token_ids:
                t_XTI_dic = {}
                for i, layer_name in enumerate(XTI_layers):
                    t_XTI_dic[layer_name] = t + (i + 1) * num_added_tokens
                pipe.add_token_replacement_XTI(t, t_XTI_dic)

            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data
            for token_ids, embeds in token_ids_embeds_XTI:
                for token_id, embed in zip(token_ids, embeds):
                    token_embeds[token_id] = embed

    # promptを取得する
    if args.from_file is not None:
        logger.info(f"reading prompts from {args.from_file}")
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompt_list = f.read().splitlines()
            prompt_list = [d for d in prompt_list if len(d.strip()) > 0 and d[0] != "#"]
    elif args.prompt is not None:
        prompt_list = [args.prompt]
    else:
        prompt_list = []

    if args.interactive:
        args.n_iter = 1

    # img2imgの前処理、画像の読み込みなど
    def load_images(path):
        if os.path.isfile(path):
            paths = [path]
        else:
            paths = (
                glob.glob(os.path.join(path, "*.png"))
                + glob.glob(os.path.join(path, "*.jpg"))
                + glob.glob(os.path.join(path, "*.jpeg"))
                + glob.glob(os.path.join(path, "*.webp"))
            )
            paths.sort()

        images = []
        for p in paths:
            image = Image.open(p)
            if image.mode != "RGB":
                logger.info(f"convert image to RGB from {image.mode}: {p}")
                image = image.convert("RGB")
            images.append(image)

        return images

    def resize_images(imgs, size):
        resized = []
        for img in imgs:
            r_img = img.resize(size, Image.Resampling.LANCZOS)
            if hasattr(img, "filename"):  # filename属性がない場合があるらしい
                r_img.filename = img.filename
            resized.append(r_img)
        return resized

    if args.image_path is not None:
        logger.info(f"load image for img2img: {args.image_path}")
        init_images = load_images(args.image_path)
        assert len(init_images) > 0, f"No image / 画像がありません: {args.image_path}"
        logger.info(f"loaded {len(init_images)} images for img2img")
    else:
        init_images = None

    if args.mask_path is not None:
        logger.info(f"load mask for inpainting: {args.mask_path}")
        mask_images = load_images(args.mask_path)
        assert len(mask_images) > 0, f"No mask image / マスク画像がありません: {args.image_path}"
        logger.info(f"loaded {len(mask_images)} mask images for inpainting")
    else:
        mask_images = None

    # promptがないとき、画像のPngInfoから取得する
    if init_images is not None and len(prompt_list) == 0 and not args.interactive:
        logger.info("get prompts from images' meta data")
        for img in init_images:
            if "prompt" in img.text:
                prompt = img.text["prompt"]
                if "negative-prompt" in img.text:
                    prompt += " --n " + img.text["negative-prompt"]
                prompt_list.append(prompt)

        # プロンプトと画像を一致させるため指定回数だけ繰り返す（画像を増幅する）
        l = []
        for im in init_images:
            l.extend([im] * args.images_per_prompt)
        init_images = l

        if mask_images is not None:
            l = []
            for im in mask_images:
                l.extend([im] * args.images_per_prompt)
            mask_images = l

    # 画像サイズにオプション指定があるときはリサイズする
    if args.W is not None and args.H is not None:
        # highres fix を考慮に入れる
        w, h = args.W, args.H
        if highres_fix:
            w = int(w * args.highres_fix_scale + 0.5)
            h = int(h * args.highres_fix_scale + 0.5)

        if init_images is not None:
            logger.info(f"resize img2img source images to {w}*{h}")
            init_images = resize_images(init_images, (w, h))
        if mask_images is not None:
            logger.info(f"resize img2img mask images to {w}*{h}")
            mask_images = resize_images(mask_images, (w, h))

    regional_network = False
    if networks and mask_images:
        # mask を領域情報として流用する、現在は一回のコマンド呼び出しで1枚だけ対応
        regional_network = True
        logger.info("use mask as region")

        size = None
        for i, network in enumerate(networks):
            if (i < 3 and args.network_regional_mask_max_color_codes is None) or i < args.network_regional_mask_max_color_codes:
                np_mask = np.array(mask_images[0])

                if args.network_regional_mask_max_color_codes:
                    # カラーコードでマスクを指定する
                    ch0 = (i + 1) & 1
                    ch1 = ((i + 1) >> 1) & 1
                    ch2 = ((i + 1) >> 2) & 1
                    np_mask = np.all(np_mask == np.array([ch0, ch1, ch2]) * 255, axis=2)
                    np_mask = np_mask.astype(np.uint8) * 255
                else:
                    np_mask = np_mask[:, :, i]
                size = np_mask.shape
            else:
                np_mask = np.full(size, 255, dtype=np.uint8)
            mask = torch.from_numpy(np_mask.astype(np.float32) / 255.0)
            network.set_region(i, i == len(networks) - 1, mask)
        mask_images = None

    prev_image = None  # for VGG16 guided
    if args.guide_image_path is not None:
        logger.info(f"load image for CLIP/VGG16/ControlNet guidance: {args.guide_image_path}")
        guide_images = []
        for p in args.guide_image_path:
            guide_images.extend(load_images(p))

        logger.info(f"loaded {len(guide_images)} guide images for guidance")
        if len(guide_images) == 0:
            logger.info(
                f"No guide image, use previous generated image. / ガイド画像がありません。直前に生成した画像を使います: {args.image_path}"
            )
            guide_images = None
    else:
        guide_images = None

    # seed指定時はseedを決めておく
    if args.seed is not None:
        # dynamic promptを使うと足りなくなる→images_per_promptを適当に大きくしておいてもらう
        random.seed(args.seed)
        predefined_seeds = [random.randint(0, 0x7FFFFFFF) for _ in range(args.n_iter * len(prompt_list) * args.images_per_prompt)]
        if len(predefined_seeds) == 1:
            predefined_seeds[0] = args.seed
    else:
        predefined_seeds = None

    # デフォルト画像サイズを設定する：img2imgではこれらの値は無視される（またはW*Hにリサイズ済み）
    if args.W is None:
        args.W = 512
    if args.H is None:
        args.H = 512

    # 画像生成のループ
    os.makedirs(args.outdir, exist_ok=True)
    max_embeddings_multiples = 1 if args.max_embeddings_multiples is None else args.max_embeddings_multiples

    for gen_iter in range(args.n_iter):
        logger.info(f"iteration {gen_iter+1}/{args.n_iter}")
        iter_seed = random.randint(0, 0x7FFFFFFF)

        # shuffle prompt list
        if args.shuffle_prompts:
            random.shuffle(prompt_list)

        # バッチ処理の関数
        def process_batch(batch: List[BatchData], highres_fix, highres_1st=False):
            batch_size = len(batch)

            # highres_fixの処理
            if highres_fix and not highres_1st:
                # 1st stageのバッチを作成して呼び出す：サイズを小さくして呼び出す
                is_1st_latent = upscaler.support_latents() if upscaler else args.highres_fix_latents_upscaling

                logger.info("process 1st stage")
                batch_1st = []
                for _, base, ext in batch:
                    width_1st = int(ext.width * args.highres_fix_scale + 0.5)
                    height_1st = int(ext.height * args.highres_fix_scale + 0.5)
                    width_1st = width_1st - width_1st % 32
                    height_1st = height_1st - height_1st % 32

                    strength_1st = ext.strength if args.highres_fix_strength is None else args.highres_fix_strength

                    ext_1st = BatchDataExt(
                        width_1st,
                        height_1st,
                        args.highres_fix_steps,
                        ext.scale,
                        ext.negative_scale,
                        strength_1st,
                        ext.network_muls,
                        ext.num_sub_prompts,
                    )
                    batch_1st.append(BatchData(is_1st_latent, base, ext_1st))

                pipe.set_enable_control_net(True)  # 1st stageではControlNetを有効にする
                images_1st = process_batch(batch_1st, True, True)

                # 2nd stageのバッチを作成して以下処理する
                logger.info("process 2nd stage")
                width_2nd, height_2nd = batch[0].ext.width, batch[0].ext.height

                if upscaler:
                    # upscalerを使って画像を拡大する
                    lowreso_imgs = None if is_1st_latent else images_1st
                    lowreso_latents = None if not is_1st_latent else images_1st

                    # 戻り値はPIL.Image.Imageかtorch.Tensorのlatents
                    batch_size = len(images_1st)
                    vae_batch_size = (
                        batch_size
                        if args.vae_batch_size is None
                        else (max(1, int(batch_size * args.vae_batch_size)) if args.vae_batch_size < 1 else args.vae_batch_size)
                    )
                    vae_batch_size = int(vae_batch_size)
                    images_1st = upscaler.upscale(
                        vae, lowreso_imgs, lowreso_latents, dtype, width_2nd, height_2nd, batch_size, vae_batch_size
                    )

                elif args.highres_fix_latents_upscaling:
                    # latentを拡大する
                    org_dtype = images_1st.dtype
                    if images_1st.dtype == torch.bfloat16:
                        images_1st = images_1st.to(torch.float)  # interpolateがbf16をサポートしていない
                    images_1st = torch.nn.functional.interpolate(
                        images_1st, (batch[0].ext.height // 8, batch[0].ext.width // 8), mode="bilinear"
                    )  # , antialias=True)
                    images_1st = images_1st.to(org_dtype)

                else:
                    # 画像をLANCZOSで拡大する
                    images_1st = [image.resize((width_2nd, height_2nd), resample=PIL.Image.LANCZOS) for image in images_1st]

                batch_2nd = []
                for i, (bd, image) in enumerate(zip(batch, images_1st)):
                    bd_2nd = BatchData(False, BatchDataBase(*bd.base[0:3], bd.base.seed + 1, image, None, *bd.base[6:]), bd.ext)
                    batch_2nd.append(bd_2nd)
                batch = batch_2nd

                if args.highres_fix_disable_control_net:
                    pipe.set_enable_control_net(False)  # オプション指定時、2nd stageではControlNetを無効にする

            # このバッチの情報を取り出す
            (
                return_latents,
                (step_first, _, _, _, init_image, mask_image, _, guide_image, _),
                (width, height, steps, scale, negative_scale, strength, network_muls, num_sub_prompts),
            ) = batch[0]
            noise_shape = (LATENT_CHANNELS, height // DOWNSAMPLING_FACTOR, width // DOWNSAMPLING_FACTOR)

            prompts = []
            negative_prompts = []
            raw_prompts = []
            start_code = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
            noises = [
                torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
                for _ in range(steps * scheduler_num_noises_per_step)
            ]
            seeds = []
            clip_prompts = []

            if init_image is not None:  # img2img?
                i2i_noises = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
                init_images = []

                if mask_image is not None:
                    mask_images = []
                else:
                    mask_images = None
            else:
                i2i_noises = None
                init_images = None
                mask_images = None

            if guide_image is not None:  # CLIP image guided?
                guide_images = []
            else:
                guide_images = None

            # バッチ内の位置に関わらず同じ乱数を使うためにここで乱数を生成しておく。あわせてimage/maskがbatch内で同一かチェックする
            all_images_are_same = True
            all_masks_are_same = True
            all_guide_images_are_same = True
            for i, (
                _,
                (_, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image, raw_prompt),
                _,
            ) in enumerate(batch):
                prompts.append(prompt)
                negative_prompts.append(negative_prompt)
                seeds.append(seed)
                clip_prompts.append(clip_prompt)
                raw_prompts.append(raw_prompt)

                if init_image is not None:
                    init_images.append(init_image)
                    if i > 0 and all_images_are_same:
                        all_images_are_same = init_images[-2] is init_image

                if mask_image is not None:
                    mask_images.append(mask_image)
                    if i > 0 and all_masks_are_same:
                        all_masks_are_same = mask_images[-2] is mask_image

                if guide_image is not None:
                    if type(guide_image) is list:
                        guide_images.extend(guide_image)
                        all_guide_images_are_same = False
                    else:
                        guide_images.append(guide_image)
                        if i > 0 and all_guide_images_are_same:
                            all_guide_images_are_same = guide_images[-2] is guide_image

                # make start code
                torch.manual_seed(seed)
                start_code[i] = torch.randn(noise_shape, device=device, dtype=dtype)

                # make each noises
                for j in range(steps * scheduler_num_noises_per_step):
                    noises[j][i] = torch.randn(noise_shape, device=device, dtype=dtype)

                if i2i_noises is not None:  # img2img noise
                    i2i_noises[i] = torch.randn(noise_shape, device=device, dtype=dtype)

            noise_manager.reset_sampler_noises(noises)

            # すべての画像が同じなら1枚だけpipeに渡すことでpipe側で処理を高速化する
            if init_images is not None and all_images_are_same:
                init_images = init_images[0]
            if mask_images is not None and all_masks_are_same:
                mask_images = mask_images[0]
            if guide_images is not None and all_guide_images_are_same:
                guide_images = guide_images[0]

            # ControlNet使用時はguide imageをリサイズする
            if control_nets:
                # TODO resampleのメソッド
                guide_images = guide_images if type(guide_images) == list else [guide_images]
                guide_images = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in guide_images]
                if len(guide_images) == 1:
                    guide_images = guide_images[0]

            # generate
            if networks:
                # 追加ネットワークの処理
                shared = {}
                for n, m in zip(networks, network_muls if network_muls else network_default_muls):
                    n.set_multiplier(m)
                    if regional_network:
                        n.set_current_generation(batch_size, num_sub_prompts, width, height, shared)

                if not regional_network and network_pre_calc:
                    for n in networks:
                        n.restore_weights()
                    for n in networks:
                        n.pre_calculation()
                    logger.info("pre-calculation... done")

            images = pipe(
                prompts,
                negative_prompts,
                init_images,
                mask_images,
                height,
                width,
                steps,
                scale,
                negative_scale,
                strength,
                latents=start_code,
                output_type="pil",
                max_embeddings_multiples=max_embeddings_multiples,
                img2img_noise=i2i_noises,
                vae_batch_size=args.vae_batch_size,
                return_latents=return_latents,
                clip_prompts=clip_prompts,
                clip_guide_images=guide_images,
            )[0]
            if highres_1st and not args.highres_fix_save_1st:  # return images or latents
                return images

            # save image
            highres_prefix = ("0" if highres_1st else "1") if highres_fix else ""
            ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
            for i, (image, prompt, negative_prompts, seed, clip_prompt, raw_prompt) in enumerate(
                zip(images, prompts, negative_prompts, seeds, clip_prompts, raw_prompts)
            ):
                if highres_fix:
                    seed -= 1  # record original seed
                metadata = PngInfo()
                metadata.add_text("prompt", prompt)
                metadata.add_text("seed", str(seed))
                metadata.add_text("sampler", args.sampler)
                metadata.add_text("steps", str(steps))
                metadata.add_text("scale", str(scale))
                if negative_prompt is not None:
                    metadata.add_text("negative-prompt", negative_prompt)
                if negative_scale is not None:
                    metadata.add_text("negative-scale", str(negative_scale))
                if clip_prompt is not None:
                    metadata.add_text("clip-prompt", clip_prompt)
                if raw_prompt is not None:
                    metadata.add_text("raw-prompt", raw_prompt)

                if args.use_original_file_name and init_images is not None:
                    if type(init_images) is list:
                        fln = os.path.splitext(os.path.basename(init_images[i % len(init_images)].filename))[0] + ".png"
                    else:
                        fln = os.path.splitext(os.path.basename(init_images.filename))[0] + ".png"
                elif args.sequential_file_name:
                    fln = f"im_{highres_prefix}{step_first + i + 1:06d}.png"
                else:
                    fln = f"im_{ts_str}_{highres_prefix}{i:03d}_{seed}.png"

                image.save(os.path.join(args.outdir, fln), pnginfo=metadata)

            if not args.no_preview and not highres_1st and args.interactive:
                try:
                    import cv2

                    for prompt, image in zip(prompts, images):
                        cv2.imshow(prompt[:128], np.array(image)[:, :, ::-1])  # プロンプトが長いと死ぬ
                        cv2.waitKey()
                        cv2.destroyAllWindows()
                except ImportError:
                    logger.info(
                        "opencv-python is not installed, cannot preview / opencv-pythonがインストールされていないためプレビューできません"
                    )

            return images

        # 画像生成のプロンプトが一周するまでのループ
        prompt_index = 0
        global_step = 0
        batch_data = []
        while args.interactive or prompt_index < len(prompt_list):
            if len(prompt_list) == 0:
                # interactive
                valid = False
                while not valid:
                    logger.info("")
                    logger.info("Type prompt:")
                    try:
                        raw_prompt = input()
                    except EOFError:
                        break

                    valid = len(raw_prompt.strip().split(" --")[0].strip()) > 0
                if not valid:  # EOF, end app
                    break
            else:
                raw_prompt = prompt_list[prompt_index]

            # sd-dynamic-prompts like variants:
            # count is 1 (not dynamic) or images_per_prompt (no enumeration) or arbitrary (enumeration)
            raw_prompts = handle_dynamic_prompt_variants(raw_prompt, args.images_per_prompt)

            # repeat prompt
            for pi in range(args.images_per_prompt if len(raw_prompts) == 1 else len(raw_prompts)):
                raw_prompt = raw_prompts[pi] if len(raw_prompts) > 1 else raw_prompts[0]

                if pi == 0 or len(raw_prompts) > 1:
                    # parse prompt: if prompt is not changed, skip parsing
                    width = args.W
                    height = args.H
                    scale = args.scale
                    negative_scale = args.negative_scale
                    steps = args.steps
                    seed = None
                    seeds = None
                    strength = 0.8 if args.strength is None else args.strength
                    negative_prompt = ""
                    clip_prompt = None
                    network_muls = None

                    # Deep Shrink
                    ds_depth_1 = None  # means no override
                    ds_timesteps_1 = args.ds_timesteps_1
                    ds_depth_2 = args.ds_depth_2
                    ds_timesteps_2 = args.ds_timesteps_2
                    ds_ratio = args.ds_ratio

                    # Gradual Latent
                    gl_timesteps = None  # means no override
                    gl_ratio = args.gradual_latent_ratio
                    gl_every_n_steps = args.gradual_latent_every_n_steps
                    gl_ratio_step = args.gradual_latent_ratio_step
                    gl_s_noise = args.gradual_latent_s_noise
                    gl_unsharp_params = args.gradual_latent_unsharp_params

                    prompt_args = raw_prompt.strip().split(" --")
                    prompt = prompt_args[0]
                    logger.info(f"prompt {prompt_index+1}/{len(prompt_list)}: {prompt}")

                    for parg in prompt_args[1:]:
                        try:
                            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
                            if m:
                                width = int(m.group(1))
                                logger.info(f"width: {width}")
                                continue

                            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
                            if m:
                                height = int(m.group(1))
                                logger.info(f"height: {height}")
                                continue

                            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
                            if m:  # steps
                                steps = max(1, min(1000, int(m.group(1))))
                                logger.info(f"steps: {steps}")
                                continue

                            m = re.match(r"d ([\d,]+)", parg, re.IGNORECASE)
                            if m:  # seed
                                seeds = [int(d) for d in m.group(1).split(",")]
                                logger.info(f"seeds: {seeds}")
                                continue

                            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # scale
                                scale = float(m.group(1))
                                logger.info(f"scale: {scale}")
                                continue

                            m = re.match(r"nl ([\d\.]+|none|None)", parg, re.IGNORECASE)
                            if m:  # negative scale
                                if m.group(1).lower() == "none":
                                    negative_scale = None
                                else:
                                    negative_scale = float(m.group(1))
                                logger.info(f"negative scale: {negative_scale}")
                                continue

                            m = re.match(r"t ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # strength
                                strength = float(m.group(1))
                                logger.info(f"strength: {strength}")
                                continue

                            m = re.match(r"n (.+)", parg, re.IGNORECASE)
                            if m:  # negative prompt
                                negative_prompt = m.group(1)
                                logger.info(f"negative prompt: {negative_prompt}")
                                continue

                            m = re.match(r"c (.+)", parg, re.IGNORECASE)
                            if m:  # clip prompt
                                clip_prompt = m.group(1)
                                logger.info(f"clip prompt: {clip_prompt}")
                                continue

                            m = re.match(r"am ([\d\.\-,]+)", parg, re.IGNORECASE)
                            if m:  # network multiplies
                                network_muls = [float(v) for v in m.group(1).split(",")]
                                while len(network_muls) < len(networks):
                                    network_muls.append(network_muls[-1])
                                logger.info(f"network mul: {network_muls}")
                                continue

                            # Deep Shrink
                            m = re.match(r"dsd1 ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # deep shrink depth 1
                                ds_depth_1 = int(m.group(1))
                                logger.info(f"deep shrink depth 1: {ds_depth_1}")
                                continue

                            m = re.match(r"dst1 ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # deep shrink timesteps 1
                                ds_timesteps_1 = int(m.group(1))
                                ds_depth_1 = ds_depth_1 if ds_depth_1 is not None else -1  # -1 means override
                                logger.info(f"deep shrink timesteps 1: {ds_timesteps_1}")
                                continue

                            m = re.match(r"dsd2 ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # deep shrink depth 2
                                ds_depth_2 = int(m.group(1))
                                ds_depth_1 = ds_depth_1 if ds_depth_1 is not None else -1  # -1 means override
                                logger.info(f"deep shrink depth 2: {ds_depth_2}")
                                continue

                            m = re.match(r"dst2 ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # deep shrink timesteps 2
                                ds_timesteps_2 = int(m.group(1))
                                ds_depth_1 = ds_depth_1 if ds_depth_1 is not None else -1  # -1 means override
                                logger.info(f"deep shrink timesteps 2: {ds_timesteps_2}")
                                continue

                            m = re.match(r"dsr ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # deep shrink ratio
                                ds_ratio = float(m.group(1))
                                ds_depth_1 = ds_depth_1 if ds_depth_1 is not None else -1  # -1 means override
                                logger.info(f"deep shrink ratio: {ds_ratio}")
                                continue

                            # Gradual Latent
                            m = re.match(r"glt ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # gradual latent timesteps
                                gl_timesteps = int(m.group(1))
                                logger.info(f"gradual latent timesteps: {gl_timesteps}")
                                continue

                            m = re.match(r"glr ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # gradual latent ratio
                                gl_ratio = float(m.group(1))
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1  # -1 means override
                                logger.info(f"gradual latent ratio: {ds_ratio}")
                                continue

                            m = re.match(r"gle ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # gradual latent every n steps
                                gl_every_n_steps = int(m.group(1))
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1  # -1 means override
                                logger.info(f"gradual latent every n steps: {gl_every_n_steps}")
                                continue

                            m = re.match(r"gls ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # gradual latent ratio step
                                gl_ratio_step = float(m.group(1))
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1  # -1 means override
                                logger.info(f"gradual latent ratio step: {gl_ratio_step}")
                                continue

                            m = re.match(r"glsn ([\d\.]+)", parg, re.IGNORECASE)
                            if m:  # gradual latent s noise
                                gl_s_noise = float(m.group(1))
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1  # -1 means override
                                logger.info(f"gradual latent s noise: {gl_s_noise}")
                                continue

                            m = re.match(r"glus ([\d\.\-,]+)", parg, re.IGNORECASE)
                            if m:  # gradual latent unsharp params
                                gl_unsharp_params = m.group(1)
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1  # -1 means override
                                logger.info(f"gradual latent unsharp params: {gl_unsharp_params}")
                                continue

                        except ValueError as ex:
                            logger.info(f"Exception in parsing / 解析エラー: {parg}")
                            logger.info(ex)

                # override Deep Shrink
                if ds_depth_1 is not None:
                    if ds_depth_1 < 0:
                        ds_depth_1 = args.ds_depth_1 or 3
                    unet.set_deep_shrink(ds_depth_1, ds_timesteps_1, ds_depth_2, ds_timesteps_2, ds_ratio)

                # override Gradual Latent
                if gl_timesteps is not None:
                    if gl_timesteps < 0:
                        gl_timesteps = args.gradual_latent_timesteps or 650
                    if gl_unsharp_params is not None:
                        unsharp_params = gl_unsharp_params.split(",")
                        us_ksize, us_sigma, us_strength = [float(v) for v in unsharp_params[:3]]
                        logger.info(f'{unsharp_params}')
                        us_target_x = True if len(unsharp_params) < 4 else bool(int(unsharp_params[3]))
                        us_ksize = int(us_ksize)
                    else:
                        us_ksize, us_sigma, us_strength, us_target_x = None, None, None, None
                    gradual_latent = GradualLatent(
                        gl_ratio,
                        gl_timesteps,
                        gl_every_n_steps,
                        gl_ratio_step,
                        gl_s_noise,
                        us_ksize,
                        us_sigma,
                        us_strength,
                        us_target_x,
                    )
                    pipe.set_gradual_latent(gradual_latent)

                # prepare seed
                if seeds is not None:  # given in prompt
                    # 数が足りないなら前のをそのまま使う
                    if len(seeds) > 0:
                        seed = seeds.pop(0)
                else:
                    if predefined_seeds is not None:
                        if len(predefined_seeds) > 0:
                            seed = predefined_seeds.pop(0)
                        else:
                            logger.info("predefined seeds are exhausted")
                            seed = None
                    elif args.iter_same_seed:
                        seed = iter_seed
                    else:
                        seed = None  # 前のを消す

                if seed is None:
                    seed = random.randint(0, 0x7FFFFFFF)
                if args.interactive:
                    logger.info(f"seed: {seed}")

                # prepare init image, guide image and mask
                init_image = mask_image = guide_image = None

                # 同一イメージを使うとき、本当はlatentに変換しておくと無駄がないが面倒なのでとりあえず毎回処理する
                if init_images is not None:
                    init_image = init_images[global_step % len(init_images)]

                    # img2imgの場合は、基本的に元画像のサイズで生成する。highres fixの場合はargs.W, args.Hとscaleに従いリサイズ済みなので無視する
                    # 32単位に丸めたやつにresizeされるので踏襲する
                    if not highres_fix:
                        width, height = init_image.size
                        width = width - width % 32
                        height = height - height % 32
                        if width != init_image.size[0] or height != init_image.size[1]:
                            logger.info(
                                f"img2img image size is not divisible by 32 so aspect ratio is changed / img2imgの画像サイズが32で割り切れないためリサイズされます。画像が歪みます"
                            )

                if mask_images is not None:
                    mask_image = mask_images[global_step % len(mask_images)]

                if guide_images is not None:
                    if control_nets:  # 複数件の場合あり
                        c = len(control_nets)
                        p = global_step % (len(guide_images) // c)
                        guide_image = guide_images[p * c : p * c + c]
                    else:
                        guide_image = guide_images[global_step % len(guide_images)]
                elif args.clip_image_guidance_scale > 0 or args.vgg16_guidance_scale > 0:
                    if prev_image is None:
                        logger.info("Generate 1st image without guide image.")
                    else:
                        logger.info("Use previous image as guide image.")
                        guide_image = prev_image

                if regional_network:
                    num_sub_prompts = len(prompt.split(" AND "))
                    assert (
                        len(networks) <= num_sub_prompts
                    ), "Number of networks must be less than or equal to number of sub prompts."
                else:
                    num_sub_prompts = None

                b1 = BatchData(
                    False,
                    BatchDataBase(
                        global_step, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image, raw_prompt
                    ),
                    BatchDataExt(
                        width,
                        height,
                        steps,
                        scale,
                        negative_scale,
                        strength,
                        tuple(network_muls) if network_muls else None,
                        num_sub_prompts,
                    ),
                )
                if len(batch_data) > 0 and batch_data[-1].ext != b1.ext:  # バッチ分割必要？
                    process_batch(batch_data, highres_fix)
                    batch_data.clear()

                batch_data.append(b1)
                if len(batch_data) == args.batch_size:
                    prev_image = process_batch(batch_data, highres_fix)[0]
                    batch_data.clear()

                global_step += 1

            prompt_index += 1

        if len(batch_data) > 0:
            process_batch(batch_data, highres_fix)
            batch_data.clear()

    logger.info("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)

    parser.add_argument(
        "--v2", action="store_true", help="load Stable Diffusion v2.0 model"
    )
    parser.add_argument(
        "--v_parameterization", action="store_true", help="enable v-parameterization training"
    )
    parser.add_argument("--prompt", type=str, default=None, help="prompt")
    parser.add_argument(
        "--from_file",
        type=str,
        default=None,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="interactive mode (generates one image)",
    )
    parser.add_argument(
        "--no_preview", action="store_true", help="do not show generated image in interactive mode"
    )
    parser.add_argument(
        "--image_path", type=str, default=None, help="image to inpaint or to generate from"
    )
    parser.add_argument("--mask_path", type=str, default=None, help="mask in inpainting")
    parser.add_argument("--strength", type=float, default=None, help="img2img strength")
    parser.add_argument("--images_per_prompt", type=int, default=1, help="number of images per prompt")
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to")
    parser.add_argument(
        "--sequential_file_name", action="store_true", help="sequential output file name"
    )
    parser.add_argument(
        "--use_original_file_name",
        action="store_true",
        help="prepend original file name in img2img",
    )
    # parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often")
    parser.add_argument("--H", type=int, default=None, help="image height")
    parser.add_argument("--W", type=int, default=None, help="image width")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--vae_batch_size",
        type=float,
        default=None,
        help="batch size for VAE, < 1.0 for ratio",
    )
    parser.add_argument(
        "--vae_slices",
        type=int,
        default=None,
        help="number of slices to split image into for VAE to reduce VRAM usage, None for no splitting (default), slower if specified",
    )
    parser.add_argument("--steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=[
            "ddim",
            "pndm",
            "lms",
            "euler",
            "euler_a",
            "heun",
            "dpm_2",
            "dpm_2_a",
            "dpmsolver",
            "dpmsolver++",
            "dpmsingle",
            "k_lms",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        help=f"sampler (scheduler) type",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)) / guidance scale",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="path to checkpoint of model"
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="path to checkpoint of vae to replace",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed, or seed of seeds in multiple generation",
    )
    parser.add_argument(
        "--iter_same_seed",
        action="store_true",
        help="use same seed for all prompts in iteration",
    )
    parser.add_argument(
        "--shuffle_prompts",
        action="store_true",
        help="shuffle prompts in iteration",
    )
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    parser.add_argument("--bf16", action="store_true", help="use bfloat16")
    parser.add_argument("--xformers", action="store_true", help="use xformers")
    parser.add_argument("--sdpa", action="store_true", help="use sdpa in PyTorch 2")
    parser.add_argument(
        "--diffusers_xformers",
        action="store_true",
        help="use xformers by diffusers (Hypernetworks doesn't work)",
    )
    parser.add_argument(
        "--opt_channels_last",
        action="store_true",
        help="set channels last option to model",
    )
    parser.add_argument(
        "--network_module",
        type=str,
        default=None,
        nargs="*",
        help="additional network module to use",
    )
    parser.add_argument(
        "--network_weights", type=str, default=None, nargs="*", help="additional network weights to load"
    )
    parser.add_argument(
        "--network_mul", type=float, default=None, nargs="*", help="additional network multiplier"
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value)",
    )
    parser.add_argument(
        "--network_show_meta", action="store_true", help="show metadata of network model"
    )
    parser.add_argument(
        "--network_merge_n_models",
        type=int,
        default=None,
        help="merge this number of networks",
    )
    parser.add_argument(
        "--network_merge", action="store_true", help="merge network weights to original model"
    )
    parser.add_argument(
        "--network_pre_calc",
        action="store_true",
        help="pre-calculate network for generation",
    )
    parser.add_argument(
        "--network_regional_mask_max_color_codes",
        type=int,
        default=None,
        help="max color codes for regional mask (default is None, mask by channel)",
    )
    parser.add_argument(
        "--textual_inversion_embeddings",
        type=str,
        default=None,
        nargs="*",
        help="Embeddings files of Textual Inversion",
    )
    parser.add_argument(
        "--XTI_embeddings",
        type=str,
        default=None,
        nargs="*",
        help="Embeddings files of Extended Textual Inversion",
    )
    parser.add_argument(
        "--clip_skip", type=int, default=None, help="layer number from bottom to use in CLIP"
    )
    parser.add_argument(
        "--max_embeddings_multiples",
        type=int,
        default=None,
        help="max embedding multiples, max token length is 75 * multiples",
    )
    parser.add_argument(
        "--clip_guidance_scale",
        type=float,
        default=0.0,
        help="enable CLIP guided SD, scale for guidance (DDIM, PNDM, LMS samplers only)",
    )
    parser.add_argument(
        "--clip_image_guidance_scale",
        type=float,
        default=0.0,
        help="enable CLIP guided SD by image, scale for guidance",
    )
    parser.add_argument(
        "--vgg16_guidance_scale",
        type=float,
        default=0.0,
        help="enable VGG16 guided SD by image, scale for guidance",
    )
    parser.add_argument(
        "--vgg16_guidance_layer",
        type=int,
        default=20,
        help="layer of VGG16 to calculate contents guide (1~30, 20 for conv4_2)",
    )
    parser.add_argument(
        "--guide_image_path", type=str, default=None, nargs="*", help="image to CLIP guidance"
    )
    parser.add_argument(
        "--highres_fix_scale",
        type=float,
        default=None,
        help="enable highres fix, reso scale for 1st stage",
    )
    parser.add_argument(
        "--highres_fix_steps",
        type=int,
        default=28,
        help="1st stage steps for highres fix",
    )
    parser.add_argument(
        "--highres_fix_strength",
        type=float,
        default=None,
        help="1st stage img2img strength for highres fix",
    )
    parser.add_argument(
        "--highres_fix_save_1st",
        action="store_true",
        help="save 1st stage images for highres fix",
    )
    parser.add_argument(
        "--highres_fix_latents_upscaling",
        action="store_true",
        help="use latents upscaling for highres fix",
    )
    parser.add_argument(
        "--highres_fix_upscaler",
        type=str,
        default=None,
        help="upscaler module for highres fix",
    )
    parser.add_argument(
        "--highres_fix_upscaler_args",
        type=str,
        default=None,
        help="additional arguments for upscaler (key=value)",
    )
    parser.add_argument(
        "--highres_fix_disable_control_net",
        action="store_true",
        help="disable ControlNet for highres fix",
    )

    parser.add_argument(
        "--negative_scale",
        type=float,
        default=None,
        help="set another guidance scale for negative prompt",
    )

    parser.add_argument(
        "--control_net_models", type=str, default=None, nargs="*", help="ControlNet models to use"
    )
    parser.add_argument(
        "--control_net_preps",
        type=str,
        default=None,
        nargs="*",
        help="ControlNet preprocess to use",
    )
    parser.add_argument("--control_net_weights", type=float, default=None, nargs="*", help="ControlNet weights")
    parser.add_argument(
        "--control_net_ratios",
        type=float,
        default=None,
        nargs="*",
        help="ControlNet guidance ratio for steps",
    )

    # Deep Shrink
    parser.add_argument(
        "--ds_depth_1",
        type=int,
        default=None,
        help="Enable Deep Shrink with this depth 1, valid values are 0 to 3",
    )
    parser.add_argument(
        "--ds_timesteps_1",
        type=int,
        default=650,
        help="Apply Deep Shrink depth 1 until this timesteps",
    )
    parser.add_argument("--ds_depth_2", type=int, default=None, help="Deep Shrink depth 2")
    parser.add_argument(
        "--ds_timesteps_2",
        type=int,
        default=650,
        help="Apply Deep Shrink depth 2 until this timesteps",
    )
    parser.add_argument(
        "--ds_ratio", type=float, default=0.5, help="Deep Shrink ratio for downsampling"
    )

    # gradual latent
    parser.add_argument(
        "--gradual_latent_timesteps",
        type=int,
        default=None,
        help="enable Gradual Latent hires fix and apply upscaling from this timesteps",
    )
    parser.add_argument(
        "--gradual_latent_ratio",
        type=float,
        default=0.5,
        help=" this size ratio, 0.5 means 1/2",
    )
    parser.add_argument(
        "--gradual_latent_ratio_step",
        type=float,
        default=0.125,
        help="step to increase ratio for Gradual Latent",
    )
    parser.add_argument(
        "--gradual_latent_every_n_steps",
        type=int,
        default=3,
        help="steps to increase size of latents every this steps for Gradual Latent",
    )
    parser.add_argument(
        "--gradual_latent_s_noise",
        type=float,
        default=1.0,
        help="s_noise for Gradual Latent",
    )
    parser.add_argument(
        "--gradual_latent_unsharp_params",
        type=str,
        default=None,
        help="unsharp mask parameters for Gradual Latent: ksize, sigma, strength, target-x (1 means True). `3,0.5,0.5,1` or `3,1.0,1.0,0` is recommended",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    setup_logging(args, reset=True)
    main(args)
