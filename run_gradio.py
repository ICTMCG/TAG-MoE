import argparse

from src.utils.device_utils import resolve_device_ids
from src.utils.inference_config import (
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_TRUE_CFG_SCALE,
    DEFAULT_WIDTH,
    generate_random_seed,
)


LIGHT_LOGO_URL = "https://raw.githubusercontent.com/yuci-gpt/TAG-MoE/refs/heads/master/static/images/logo_light.png"
DARK_LOGO_URL = "https://raw.githubusercontent.com/yuci-gpt/TAG-MoE/refs/heads/master/static/images/logo_dark.png"


def parse_args():
    parser = argparse.ArgumentParser(description="TAG-MoE Gradio WebUI")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Path to the base Qwen-Image model directory",
    )
    parser.add_argument(
        "--transformer_model_path",
        type=str,
        required=True,
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
        "--device",
        type=str,
        default=None,
        help="Device spec. Examples: '0', '0,1', 'cpu'. Default: framework default (cuda:0 if available, else cpu).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Gradio host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Gradio port (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio public sharing",
    )
    return parser.parse_args()


def build_demo(gr, pipeline, base64_to_image_fn):
    def infer(
        image,
        prompt,
        negative_prompt,
        seed,
        gen_width,
        gen_height,
        cfg_scale,
        inference_steps,
    ):
        if prompt is None or not str(prompt).strip():
            raise gr.Error("Prompt cannot be empty.")

        if image is None:
            raise gr.Error("Image is required.")

        width_value = int(gen_width) if gen_width is not None else int(image.size[0])
        height_value = int(gen_height) if gen_height is not None else int(image.size[1])
        input_dict = {
            "image": image.convert("RGB"),
            "prompt": str(prompt).strip(),
            "negative_prompt": str(negative_prompt or DEFAULT_NEGATIVE_PROMPT),
            "seed": int(seed if seed is not None else DEFAULT_SEED),
            "target_width": width_value,
            "target_height": height_value,
            "true_cfg_scale": float(cfg_scale),
            "num_inference_steps": int(inference_steps),
            "keep_original_size": False,
        }
        result = pipeline.predict(input_dict)
        out_image = base64_to_image_fn(result["generate_imgs_buffer"][0])
        used_seed = int(result["seed"])
        return out_image, used_seed

    def randomize_seed():
        return generate_random_seed()

    def on_image_upload(image):
        if image is None:
            return gr.update(), gr.update()
        w, h = image.size
        return int(w), int(h)

    custom_css = """
    .tagmoe-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }
    .tagmoe-header img {
        width: 48px;
        height: 48px;
        object-fit: contain;
    }
    .tagmoe-header h1 {
        margin: 0;
        font-size: 1.8rem;
    }
    .tagmoe-header p {
        margin: 0;
        opacity: 0.85;
        font-size: 0.95rem;
    }
    .param-card {
        border: 1px solid var(--border-color-primary);
        border-radius: 12px;
        padding: 14px 14px 10px;
        margin-bottom: 10px;
    }
    .param-card .gradio-textbox textarea {
        min-height: 110px !important;
    }
    .run-btn button {
        height: 46px !important;
        font-weight: 600;
    }
    .image-panel {
        border: 1px solid var(--border-color-primary);
        border-radius: 12px;
        padding: 10px;
    }
    .tool-btn {
        margin-top: 28px !important;
        min-width: 42px !important;
        height: 42px !important;
        padding: 0 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    """

    title_html = f"""
    <div class="tagmoe-header">
      <picture>
        <source srcset="{DARK_LOGO_URL}" media="(prefers-color-scheme: dark)">
        <img src="{LIGHT_LOGO_URL}" alt="TAG-MoE logo">
      </picture>
      <div>
        <h1>TAG-MoE</h1>
        <p>Task-Aware Gating for Unified Generative Mixture-of-Experts</p>
      </div>
    </div>
    """

    with gr.Blocks(title="TAG-MoE WebUI", css=custom_css) as demo:
        gr.HTML(title_html)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes=["image-panel"]):
                image_input = gr.Image(
                    type="pil",
                    label="Input Image",
                    height=520,
                )
            with gr.Column(scale=1, elem_classes=["image-panel"]):
                image_output = gr.Image(
                    type="pil",
                    label="Output Image",
                    height=520,
                )

        with gr.Group(elem_classes=["param-card"]):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Describe the instruction",
                lines=3,
            )
            negative_prompt_input = gr.Textbox(
                label="Negative Prompt",
                value=DEFAULT_NEGATIVE_PROMPT,
                lines=2,
                placeholder="Optional negative prompt",
            )
            with gr.Row():
                gen_width_input = gr.Slider(
                    minimum=64,
                    maximum=4096,
                    step=1,
                    value=DEFAULT_WIDTH,
                    label="Width",
                )
                gen_height_input = gr.Slider(
                    minimum=64,
                    maximum=4096,
                    step=1,
                    value=DEFAULT_HEIGHT,
                    label="Height",
                )
            with gr.Row():
                cfg_scale_input = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=DEFAULT_TRUE_CFG_SCALE,
                    label="CFG Scale",
                )
                inference_steps_input = gr.Slider(
                    minimum=10,
                    maximum=100,
                    step=1,
                    value=DEFAULT_NUM_INFERENCE_STEPS,
                    label="Inference Steps",
                )
                with gr.Column(scale=1, min_width=200):
                    with gr.Row():
                        seed_input = gr.Number(
                            label="Seed",
                            value=generate_random_seed(),
                            precision=0,
                            scale=1,
                        )
                        random_seed_btn = gr.Button(
                            "🎲",
                            elem_classes=["tool-btn"],
                            scale=0,
                            min_width=42,
                            variant="secondary",
                        )

            run_btn = gr.Button("Run Inference", variant="primary", elem_classes=["run-btn"])

        run_btn.click(
            fn=infer,
            inputs=[
                image_input,
                prompt_input,
                negative_prompt_input,
                seed_input,
                gen_width_input,
                gen_height_input,
                cfg_scale_input,
                inference_steps_input,
            ],
            outputs=[image_output, seed_input],
        )

        image_input.change(
            fn=on_image_upload,
            inputs=[image_input],
            outputs=[gen_width_input, gen_height_input],
        )
        random_seed_btn.click(fn=randomize_seed, outputs=[seed_input])

    return demo


def main():
    args = parse_args()

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Please run `uv sync` to install dependencies."
        ) from exc
    from src.infer_tagmoe import End2End, base64_to_image

    device_ids = resolve_device_ids(args.device)
    if device_ids is None:
        print("Using default device selection (cuda:0 if available, else cpu).")
    else:
        print(f"Using device ids: {device_ids if device_ids else 'cpu'}")

    pipeline = End2End(
        args.pretrained_model_path,
        args.transformer_model_path,
        device_ids=device_ids,
        transformer_weight_name=args.transformer_weight_name,
        transformer_subfolder=args.transformer_subfolder,
        transformer_revision=args.transformer_revision,
        local_files_only=bool(args.local_files_only),
    )
    demo = build_demo(
        gr,
        pipeline,
        base64_to_image_fn=base64_to_image,
    )
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
