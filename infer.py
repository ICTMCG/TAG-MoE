import argparse
import os

from PIL import Image

from src.utils.device_utils import resolve_device_ids
from src.utils.inference_config import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_TRUE_CFG_SCALE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="TAG-MoE inference")
    parser.add_argument(
        "--pretrained_model_path", type=str, required=True,
        help="Path to the base Qwen-Image model directory",
    )
    parser.add_argument(
        "--transformer_model_path", type=str, required=True,
        help=(
            "Transformer weights source: Hugging Face repo_id, local folder, or local checkpoint file "
            "(.safetensors/.bin/.pt, or sharded *.index.json layout)"
        ),
    )
    parser.add_argument(
        "--transformer_weight_name",
        type=str,
        default="diffusion_pytorch_model.safetensors",
        help="Weight filename (or index filename) inside --transformer_model_path when source is a repo_id or folder.",
    )
    parser.add_argument(
        "--transformer_subfolder",
        type=str,
        default="transformer",
        help="Subfolder inside --transformer_model_path for component-style layouts.",
    )
    parser.add_argument(
        "--transformer_revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision when --transformer_model_path is a repo_id.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Only load local cached files when --transformer_model_path is a repo_id.",
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text instruction",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save the output image",
    )
    parser.add_argument(
        "--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt (default: empty)",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Target output width (default: input image width)",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Target output height (default: input image height)",
    )
    parser.add_argument(
        "--cfg", type=float, default=DEFAULT_TRUE_CFG_SCALE,
        help=f"True CFG scale (default: {DEFAULT_TRUE_CFG_SCALE})",
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS,
        help=f"Number of inference steps (default: {DEFAULT_NUM_INFERENCE_STEPS})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}, random each run)",
    )
    parser.add_argument(
        "--keep_original_size", action="store_true",
        help="Resize output back to input size in editing mode when width/height are not customized",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device spec. Examples: '0' (single GPU), '0,1' (multi GPU), 'cpu'. Default: framework default (cuda:0 if available, else cpu).",
    )
    return parser.parse_args()


def main():
    from src.infer_tagmoe import End2End, base64_to_image

    args = parse_args()

    device_ids = resolve_device_ids(args.device)

    pipeline = End2End(
        args.pretrained_model_path,
        args.transformer_model_path,
        device_ids=device_ids,
        transformer_weight_name=args.transformer_weight_name,
        transformer_subfolder=args.transformer_subfolder,
        transformer_revision=args.transformer_revision,
        local_files_only=bool(args.local_files_only),
    )

    input_image = Image.open(args.image)
    input_dict = dict(
        image=input_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        target_width=int(args.width) if args.width is not None else None,
        target_height=int(args.height) if args.height is not None else None,
        true_cfg_scale=float(args.cfg),
        num_inference_steps=int(args.steps),
        seed=args.seed,
        keep_original_size=bool(args.keep_original_size),
    )

    res = pipeline.predict(input_dict)
    output_image = base64_to_image(res["generate_imgs_buffer"][0])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output_image.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
