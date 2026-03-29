import base64
import io
import os
import time
from functools import partial

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.utils.device_utils import build_accelerate_max_memory_map
from src.utils.inference_config import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_TRUE_CFG_SCALE,
    generate_random_seed,
    normalize_negative_prompt,
)
from src.models.transformer_qwenimage_tagmoe import QwenImageTransformer2DModel, TRANSFORMER_NUM_LAYERS, MOE_NUM_EXPERTS
from src.pipelines.pipeline_qwenimage_tagmoe import QwenImagePipeline


def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format="PNG")
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def image_to_base64(image: Image) -> str:
    return base64.b64encode(image_to_byte_array(image)).decode()


def base64_to_image(base64_str: str) -> Image:
    return Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")


PREFERRED_QWENIMAGE_RESOLUTIONS = [
    (512, 2048),
    (512, 1984),
    (512, 1920),
    (512, 1856),
    (512, 1792),
    (512, 1728),
    (512, 1664),
    (512, 1600),
    (512, 1536),
    (576, 1472),
    (640, 1408),
    (704, 1344),
    (768, 1280),
    (832, 1216),
    (896, 1152),
    (960, 1088),
    (1024, 1024),
    (1088, 960),
    (1152, 896),
    (1216, 832),
    (1280, 768),
    (1344, 704),
    (1408, 640),
    (1472, 576),
    (1536, 512),
    (1600, 512),
    (1664, 512),
    (1728, 512),
    (1792, 512),
    (1856, 512),
    (1920, 512),
    (1984, 512),
    (2048, 512),
]


QWEN_IMAGE_TRANSFORMER_BLOCK_DIM = 3072
SEMANTIC_DIM = 512
TAG_DICT = {
    "local editing": 0,
    "global editing": 1,
    "multi region editing": 2,
    "viewpoint editing": 3,
    "content customization": 4,
    "style customization": 5,
    "object editing": 6,
    "attribute editing": 7,
    "style transfer": 8,
    "pose editing": 9,
    "background editing": 10,
    "illumination editing": 11,
    "structure preservation": 12,
    "background preservation": 13,
    "identity preservation": 14,
    "face preservation": 15,
    "style preservation": 16,
    "image generation": 17,
}


class PredictionHead(nn.Module):
    def __init__(self, gating_dim: int = 4, semantic_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gating_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, semantic_dim),
        )

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.net(g)


class End2End:
    def __init__(
        self,
        pretrained_model_path,
        transformer_model_path=None,
        rank=0,
        device_ids=None,
        transformer_weight_name: str = "diffusion_pytorch_model.safetensors",
        transformer_subfolder: str | None = "transformer",
        transformer_revision: str | None = None,
        local_files_only: bool = False,
    ):
        self.device_ids = self._resolve_device_ids(rank, device_ids)
        self.is_multi_gpu = len(self.device_ids) > 1

        self.device, self.generator_device, torch_dtype = self._resolve_runtime_device()
        transformer = self._build_runtime_transformer(pretrained_model_path, torch_dtype)

        self.pipe = QwenImagePipeline.from_pretrained(
            pretrained_model_path,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )

        self.pipe.init_custom(
            transformer_model_path,
            weight_name=transformer_weight_name,
            subfolder=transformer_subfolder,
            revision=transformer_revision,
            local_files_only=local_files_only,
        )
        if self.is_multi_gpu:
            self._enable_multi_gpu_dispatch(torch_dtype=torch_dtype)
        else:
            self.pipe = self.pipe.to(self.device)

    @staticmethod
    def _resolve_device_ids(rank, device_ids):
        if device_ids is None:
            return [rank] if torch.cuda.is_available() else []
        return list(device_ids)

    def _resolve_runtime_device(self):
        if len(self.device_ids) > 0 and torch.cuda.is_available():
            primary_gpu = self.device_ids[0]
            torch.cuda.set_device(primary_gpu)
            device = f"cuda:{primary_gpu}"
            return device, device, torch.bfloat16
        return "cpu", "cpu", torch.float32

    def _build_runtime_transformer(self, pretrained_model_path, torch_dtype):
        transformer = QwenImageTransformer2DModel.from_pretrained(
            pretrained_model_path,
            subfolder="transformer",
            torch_dtype=torch_dtype,
        )
        self._replace_mlp_with_runtime_moe(transformer)
        self._attach_tag_modules(transformer)
        return transformer

    def _build_moe_args(self):
        from megablocks.layers.arguments import Arguments

        return Arguments(
            hidden_size=QWEN_IMAGE_TRANSFORMER_BLOCK_DIM,
            ffn_hidden_size=QWEN_IMAGE_TRANSFORMER_BLOCK_DIM * 4,
            num_layers=TRANSFORMER_NUM_LAYERS,
            bias=True,
            activation_fn=partial(F.gelu, approximate="tanh"),
            moe_num_experts=MOE_NUM_EXPERTS,
            moe_top_k=1,
            moe_loss_weight=0.01,
            moe_capacity_factor=1.25,
            mlp_type="mlp",
            shared_expert=False,
            mlp_impl="grouped",
            init_method=nn.init.xavier_uniform_,
            moe_expert_model_parallelism=False,
            expert_parallel_group=None,
            fp16=False,
            bf16=True,
            device=self.device,
        )

    def _replace_mlp_with_runtime_moe(self, transformer):
        from megablocks.layers.dmoe import dMoE

        moe_args = self._build_moe_args()
        replace_from_layer = 60 - TRANSFORMER_NUM_LAYERS
        replace_paths = []
        for name, _ in transformer.named_modules():
            if not name.startswith("transformer_blocks.") or not name.endswith("img_mlp"):
                continue
            block_idx = int(name.split(".")[1])
            if block_idx >= replace_from_layer:
                replace_paths.append(name)

        for path in replace_paths:
            parent_name, child_name = path.rsplit(".", 1)
            parent_module = transformer.get_submodule(parent_name)
            setattr(parent_module, child_name, dMoE(moe_args))

    def _attach_tag_modules(self, transformer):
        transformer.tag_embedding = nn.Embedding(len(TAG_DICT), SEMANTIC_DIM)
        transformer.router_head = PredictionHead(
            gating_dim=MOE_NUM_EXPERTS,
            semantic_dim=SEMANTIC_DIM,
            hidden_dim=256,
        )

    def _enable_multi_gpu_dispatch(self, torch_dtype):
        from accelerate import dispatch_model, infer_auto_device_map

        free_bytes_by_device = {}
        for device_id in self.device_ids:
            free_bytes, _ = torch.cuda.mem_get_info(device_id)
            free_bytes_by_device[device_id] = free_bytes
        max_memory = build_accelerate_max_memory_map(self.device_ids, free_bytes_by_device)

        transformer_device_map = infer_auto_device_map(
            self.pipe.transformer,
            max_memory=max_memory,
            no_split_module_classes=["QwenImageTransformerBlock"],
            dtype=torch_dtype,
        )

        offload_dir = None
        if any(device == "disk" for device in transformer_device_map.values()):
            offload_dir = os.path.join("/tmp", "tag_moe_offload")
            os.makedirs(offload_dir, exist_ok=True)

        self.pipe.transformer = dispatch_model(
            self.pipe.transformer,
            device_map=transformer_device_map,
            offload_dir=offload_dir,
        )

        text_encoder_device = f"cuda:{self.device_ids[-1]}"
        self.pipe.text_encoder = self.pipe.text_encoder.to(text_encoder_device)
        self.pipe.vae = self.pipe.vae.to(self.device)


    def predict(self, input_dict):
        out_dict = {}

        start_time = time.time()
        image = input_dict.get("image")
        if image is None:
            raise ValueError("Input image is required.")
        seed = int(input_dict.get("seed", DEFAULT_SEED))
        prompt = input_dict.get("prompt", "")
        negative_prompt = normalize_negative_prompt(
            input_dict.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        )
        num_inference_steps = int(
            input_dict.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)
        )
        true_cfg_scale = float(
            input_dict.get("true_cfg_scale", DEFAULT_TRUE_CFG_SCALE)
        )
        target_height = input_dict.get("target_height", None)
        target_width = input_dict.get("target_width", None)
        keep_original_size = bool(input_dict.get("keep_original_size", False))
        has_custom_target = target_height is not None or target_width is not None

        if seed < 0:
            seed = generate_random_seed()
        out_dict["seed"] = seed

        cond_image = image
        w_ori, h_ori = cond_image.size
        original_size = (w_ori, h_ori)

        white_bg = Image.new("RGB", cond_image.size, (255, 255, 255))
        if cond_image.mode == "RGBA":
            result = Image.alpha_composite(white_bg.convert("RGBA"), cond_image)
            cond_image = result.convert("RGB")
        else:
            cond_image = cond_image.convert("RGB")

        aspect_ratio = w_ori / h_ori
        _, snap_width, snap_height = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_QWENIMAGE_RESOLUTIONS
        )
        cond_image = cond_image.resize((snap_width, snap_height), Image.LANCZOS)

        if target_height is None:
            target_height = snap_height
        if target_width is None:
            target_width = snap_width

        out_image_pil = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=target_width,
            height=target_height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=torch.Generator(device=self.generator_device).manual_seed(seed),
            cond_image=cond_image,
        ).images[0]

        if keep_original_size and original_size is not None and not has_custom_target:
            out_image_pil = out_image_pil.resize(original_size, Image.LANCZOS)

        out_dict["generate_imgs_buffer"] = [image_to_base64(out_image_pil)]
        logger.info(f"Generation time: {time.time()-start_time:.2f}s")
        return out_dict
