import argparse
import copy
import logging
import math
import os
import random
import time
import warnings
from datetime import datetime
from typing import List, Tuple

import diffusers
import mlflow
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torch import nn
from tqdm.auto import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from joyhallo.animate.face_animate import FaceAnimatePipeline
from joyhallo.datasets.audio_processor import AudioProcessor
from joyhallo.datasets.image_processor import ImageProcessor
from joyhallo.datasets.talk_video import TalkingVideoDataset
from joyhallo.models.audio_proj import AudioProjModel
from joyhallo.models.face_locator import FaceLocator
from joyhallo.models.image_proj import ImageProjModel
from joyhallo.models.mutual_self_attention import ReferenceAttentionControl
from joyhallo.models.unet_2d_condition import UNet2DConditionModel
from joyhallo.models.unet_3d import UNet3DConditionModel
from joyhallo.utils.util import (compute_snr, delete_additional_ckpt,
                              import_filename, init_output_dir,
                              load_checkpoint, save_checkpoint,
                              seed_everything, tensor_to_video)

warnings.filterwarnings("ignore")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):
    """
    The Net class defines a neural network model that combines a reference UNet2DConditionModel,
    a denoising UNet3DConditionModel, a face locator, and other components to animate a face in a static image.

    Args:
        reference_unet (UNet2DConditionModel): The reference UNet2DConditionModel used for face animation.
        denoising_unet (UNet3DConditionModel): The denoising UNet3DConditionModel used for face animation.
        face_locator (FaceLocator): The face locator model used for face animation.
        reference_control_writer: The reference control writer component.
        reference_control_reader: The reference control reader component.
        imageproj: The image projection model.
        audioproj: The audio projection model.

    Forward method:
        noisy_latents (torch.Tensor): The noisy latents tensor.
        timesteps (torch.Tensor): The timesteps tensor.
        ref_image_latents (torch.Tensor): The reference image latents tensor.
        face_emb (torch.Tensor): The face embeddings tensor.
        audio_emb (torch.Tensor): The audio embeddings tensor.
        mask (torch.Tensor): Hard face mask for face locator.
        full_mask (torch.Tensor): Pose Mask.
        face_mask (torch.Tensor): Face Mask
        lip_mask (torch.Tensor): Lip Mask
        uncond_img_fwd (bool): A flag indicating whether to perform reference image unconditional forward pass.
        uncond_audio_fwd (bool): A flag indicating whether to perform audio unconditional forward pass.

    Returns:
        torch.Tensor: The output tensor of the neural network model.
    """
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
        reference_control_writer,
        reference_control_reader,
        imageproj,
        audioproj,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.imageproj = imageproj
        self.audioproj = audioproj

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        ref_image_latents: torch.Tensor,
        face_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        mask: torch.Tensor,
        full_mask: torch.Tensor,
        face_mask: torch.Tensor,
        lip_mask: torch.Tensor,
        uncond_img_fwd: bool = False,
        uncond_audio_fwd: bool = False,
    ):
        """
        simple docstring to prevent pylint error
        """
        face_emb = self.imageproj(face_emb)
        mask = mask.to(device="cuda")
        mask_feature = self.face_locator(mask)
        audio_emb = audio_emb.to(
            device=self.audioproj.device, dtype=self.audioproj.dtype)
        audio_emb = self.audioproj(audio_emb)

        # condition forward
        if not uncond_img_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            ref_timesteps = repeat(
                ref_timesteps,
                "b -> (repeat b)",
                repeat=ref_image_latents.size(0) // ref_timesteps.size(0),
            )
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=face_emb,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        if uncond_audio_fwd:
            audio_emb = torch.zeros_like(audio_emb).to(
                device=audio_emb.device, dtype=audio_emb.dtype
            )

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            mask_cond_fea=mask_feature,
            encoder_hidden_states=face_emb,
            audio_embedding=audio_emb,
            full_mask=full_mask,
            face_mask=face_mask,
            lip_mask=lip_mask
        ).sample

        return model_pred


def get_attention_mask(mask: torch.Tensor, weight_dtype: torch.dtype) -> torch.Tensor:
    """
    Rearrange the mask tensors to the required format.

    Args:
        mask (torch.Tensor): The input mask tensor.
        weight_dtype (torch.dtype): The data type for the mask tensor.

    Returns:
        torch.Tensor: The rearranged mask tensor.
    """
    if isinstance(mask, List):
        _mask = []
        for m in mask:
            _mask.append(
                rearrange(m, "b f 1 h w -> (b f) (h w)").to(weight_dtype))
        return _mask
    mask = rearrange(mask, "b f 1 h w -> (b f) (h w)").to(weight_dtype)
    return mask


def get_noise_scheduler(cfg: argparse.Namespace) -> Tuple[DDIMScheduler, DDIMScheduler]:
    """
    Create noise scheduler for training.

    Args:
        cfg (argparse.Namespace): Configuration object.

    Returns:
        Tuple[DDIMScheduler, DDIMScheduler]: Train noise scheduler and validation noise scheduler.
    """

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)

    return train_noise_scheduler, val_noise_scheduler


def process_audio_emb(audio_emb: torch.Tensor) -> torch.Tensor:
    """
    Process the audio embedding to concatenate with other tensors.

    Parameters:
        audio_emb (torch.Tensor): The audio embedding tensor to process.

    Returns:
        concatenated_tensors (List[torch.Tensor]): The concatenated tensor list.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0] - 1), 0)]for j in range(-2, 3)]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb


def log_validation(
    accelerator: Accelerator,
    vae: AutoencoderKL,
    net: Net,
    scheduler: DDIMScheduler,
    width: int,
    height: int,
    clip_length: int = 24,
    generator: torch.Generator = None,
    cfg: dict = None,
    save_dir: str = None,
    global_step: int = 0,
    times: int = None,
    face_analysis_model_path: str = "",
) -> None:
    """
    Log validation video during the training process.

    Args:
        accelerator (Accelerator): The accelerator for distributed training.
        vae (AutoencoderKL): The autoencoder model.
        net (Net): The main neural network model.
        scheduler (DDIMScheduler): The scheduler for noise.
        width (int): The width of the input images.
        height (int): The height of the input images.
        clip_length (int): The length of the video clips. Defaults to 24.
        generator (torch.Generator): The random number generator. Defaults to None.
        cfg (dict): The configuration dictionary. Defaults to None.
        save_dir (str): The directory to save validation results. Defaults to None.
        global_step (int): The current global step in training. Defaults to 0.
        times (int): The number of inference times. Defaults to None.
        face_analysis_model_path (str): The path to the face analysis model. Defaults to "".

    Returns:
        torch.Tensor: The tensor result of the validation.
    """
    ori_net = accelerator.unwrap_model(net)
    reference_unet = ori_net.reference_unet
    denoising_unet = ori_net.denoising_unet
    face_locator = ori_net.face_locator
    imageproj = ori_net.imageproj
    audioproj = ori_net.audioproj

    generator = torch.manual_seed(42)
    tmp_denoising_unet = copy.deepcopy(denoising_unet)

    pipeline = FaceAnimatePipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=tmp_denoising_unet,
        face_locator=face_locator,
        image_proj=imageproj,
        scheduler=scheduler,
    )
    pipeline = pipeline.to("cuda")

    image_processor = ImageProcessor((width, height), face_analysis_model_path)
    audio_processor = AudioProcessor(
        cfg.data.sample_rate,
        cfg.data.fps,
        cfg.wav2vec_config.model_path,
        cfg.wav2vec_config.features == "last",
        os.path.dirname(cfg.audio_separator.model_path),
        os.path.basename(cfg.audio_separator.model_path),
        os.path.join(save_dir, '.cache', "audio_preprocess")
    )

    for idx, ref_img_path in enumerate(cfg.ref_img_path):
        audio_path = cfg.audio_path[idx]
        source_image_pixels, \
        source_image_face_region, \
        source_image_face_emb, \
        source_image_full_mask, \
        source_image_face_mask, \
        source_image_lip_mask = image_processor.preprocess(
            ref_img_path, os.path.join(save_dir, '.cache'), cfg.face_expand_ratio)
        audio_emb, audio_length = audio_processor.preprocess(
            audio_path, clip_length)

        audio_emb = process_audio_emb(audio_emb)

        source_image_pixels = source_image_pixels.unsqueeze(0)
        source_image_face_region = source_image_face_region.unsqueeze(0)
        source_image_face_emb = source_image_face_emb.reshape(1, -1)
        source_image_face_emb = torch.tensor(source_image_face_emb)

        source_image_full_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_full_mask
        ]
        source_image_face_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_face_mask
        ]
        source_image_lip_mask = [
            (mask.repeat(clip_length, 1))
            for mask in source_image_lip_mask
        ]

        times = audio_emb.shape[0] // clip_length
        tensor_result = []
        generator = torch.manual_seed(42)
        for t in range(times):
            print(f"[{t+1}/{times}]")

            if len(tensor_result) == 0:
                # The first iteration
                motion_zeros = source_image_pixels.repeat(
                    cfg.data.n_motion_frames, 1, 1, 1)
                motion_zeros = motion_zeros.to(
                    dtype=source_image_pixels.dtype, device=source_image_pixels.device)
                pixel_values_ref_img = torch.cat(
                    [source_image_pixels, motion_zeros], dim=0)  # concat the ref image and the first motion frames
            else:
                motion_frames = tensor_result[-1][0]
                motion_frames = motion_frames.permute(1, 0, 2, 3)
                motion_frames = motion_frames[0 - cfg.data.n_motion_frames:]
                motion_frames = motion_frames * 2.0 - 1.0
                motion_frames = motion_frames.to(
                    dtype=source_image_pixels.dtype, device=source_image_pixels.device)
                pixel_values_ref_img = torch.cat(
                    [source_image_pixels, motion_frames], dim=0)  # concat the ref image and the motion frames

            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

            audio_tensor = audio_emb[
                t * clip_length: min((t + 1) * clip_length, audio_emb.shape[0])
            ]
            audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(
                device=audioproj.device, dtype=audioproj.dtype)
            audio_tensor = audioproj(audio_tensor)

            pipeline_output = pipeline(
                ref_image=pixel_values_ref_img,
                audio_tensor=audio_tensor,
                face_emb=source_image_face_emb,
                face_mask=source_image_face_region,
                pixel_values_full_mask=source_image_full_mask,
                pixel_values_face_mask=source_image_face_mask,
                pixel_values_lip_mask=source_image_lip_mask,
                width=cfg.data.train_width,
                height=cfg.data.train_height,
                video_length=clip_length,
                num_inference_steps=cfg.inference_steps,
                guidance_scale=cfg.cfg_scale,
                generator=generator,
            )

            tensor_result.append(pipeline_output.videos)

        tensor_result = torch.cat(tensor_result, dim=2)
        tensor_result = tensor_result.squeeze(0)
        tensor_result = tensor_result[:, :audio_length]
        audio_name = os.path.basename(audio_path).split('.')[0]
        ref_name = os.path.basename(ref_img_path).split('.')[0]
        output_file = os.path.join(save_dir,f"{global_step}_{ref_name}_{audio_name}.mp4")
        # save the result after all iteration
        tensor_to_video(tensor_result, output_file, audio_path)


    # clean up
    del tmp_denoising_unet
    del pipeline
    del image_processor
    del audio_processor
    torch.cuda.empty_cache()

    return tensor_result


def inference_process(cfg: argparse.Namespace) -> None:
    """
    Trains the model using the given configuration (cfg).

    Args:
        cfg (dict): The configuration dictionary containing the parameters for training.

    Notes:
        - This function trains the model using the given configuration.
        - It initializes the necessary components for training, such as the pipeline, optimizer, and scheduler.
        - The training progress is logged and tracked using the accelerator.
        - The trained model is saved after the training is completed.
    """
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with="mlflow",
        project_dir="./mlruns",
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # create output dir for training
    exp_name = cfg.exp_name
    save_dir = f"{cfg.output_dir}/{exp_name}"
    validation_dir = save_dir
    if accelerator.is_main_process:
        init_output_dir([save_dir])

    accelerator.wait_for_everyone()

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    # Create Models
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )
    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda", dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        cfg.mm_path,
        subfolder="unet",
        unet_additional_kwargs=OmegaConf.to_container(
            cfg.unet_additional_kwargs),
        use_landmark=False
    ).to(device="cuda", dtype=weight_dtype)
    imageproj = ImageProjModel(
        cross_attention_dim=denoising_unet.config.cross_attention_dim,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    ).to(device="cuda", dtype=weight_dtype)
    face_locator = FaceLocator(
        conditioning_embedding_channels=320,
    ).to(device="cuda", dtype=weight_dtype)
    audioproj = AudioProjModel(
        seq_len=5,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
    ).to(device="cuda", dtype=weight_dtype)

    # Freeze
    vae.requires_grad_(False)
    imageproj.requires_grad_(False)
    reference_unet.requires_grad_(False)
    denoising_unet.requires_grad_(False)
    face_locator.requires_grad_(False)
    audioproj.requires_grad_(True)

    # Set motion module learnable
    trainable_modules = cfg.trainable_para
    for name, module in denoising_unet.named_modules():
        if any(trainable_mod in name for trainable_mod in trainable_modules):
            for params in module.parameters():
                params.requires_grad_(True)

    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )

    net = Net(
        reference_unet,
        denoising_unet,
        face_locator,
        reference_control_writer,
        reference_control_reader,
        imageproj,
        audioproj,
    ).to(dtype=weight_dtype)

    m,u = net.load_state_dict(
        torch.load(
            cfg.audio_ckpt_dir,
            map_location="cpu",
        ),
    )
    assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
    print("loaded weight from ", os.path.join(cfg.audio_ckpt_dir))

    # get noise scheduler
    _, val_noise_scheduler = get_noise_scheduler(cfg)

    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()

        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            ) from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(
        filter(lambda p: p.requires_grad, net.parameters()))

    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )

    # get data loader
    train_dataset = TalkingVideoDataset(
        img_size=(cfg.data.train_width, cfg.data.train_height),
        sample_rate=cfg.data.sample_rate,
        n_sample_frames=cfg.data.n_sample_frames,
        n_motion_frames=cfg.data.n_motion_frames,
        audio_margin=cfg.data.audio_margin,
        data_meta_paths=cfg.data.train_meta_paths,
        wav2vec_cfg=cfg.wav2vec_config,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=16
    )

    # Prepare everything with our `accelerator`.
    (
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        net,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        accelerator.init_trackers(
            exp_name,
            init_kwargs={"mlflow": {"run_name": run_time}},
        )

    logger.info("***** Running Inferencing *****")

    # Inference
    log_validation(
        accelerator=accelerator,
        vae=vae,
        net=net,
        scheduler=val_noise_scheduler,
        width=cfg.data.train_width,
        height=cfg.data.train_height,
        clip_length=cfg.data.n_sample_frames,
        cfg=cfg,
        save_dir=validation_dir,
        global_step=0,
        times=cfg.single_inference_times if cfg.single_inference_times is not None else None,
        face_analysis_model_path=cfg.face_analysis_model_path
    )

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


def load_config(config_path: str) -> dict:
    """
    Loads the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """

    if config_path.endswith(".yaml"):
        return OmegaConf.load(config_path)
    if config_path.endswith(".py"):
        return import_filename(config_path).cfg
    raise ValueError("Unsupported format for config file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/inference/inference.yaml"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        inference_process(config)
    except Exception as e:
        logging.error("Failed to execute the training process: %s", e)
