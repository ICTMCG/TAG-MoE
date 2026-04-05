# TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts

> **TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts**<br>
> Yu Xu<sup>1,2†</sup>, Hongbin Yan<sup>1</sup>, Juan Cao<sup>1</sup>, Yiji Cheng<sup>2</sup>, Tiankai Hang<sup>2</sup>, Runze He<sup>2</sup>, Zijin Yin<sup>2</sup>, Shiyi Zhang<sup>2</sup>, Yuxin Zhang<sup>1</sup>, Jintao Li<sup>1</sup>, Chunyu Wang<sup>2‡</sup>, Qinglin Lu<sup>2</sup>, Tong-Yee Lee<sup>3</sup>, Fan Tang<sup>1§</sup><br>
> <sup>1</sup>University of Chinese Academy of Sciences, <sup>2</sup>Tencent Hunyuan, <sup>3</sup>National Cheng-Kung University

<a href='https://arxiv.org/abs/2601.08881'><img src='https://img.shields.io/badge/ArXiv-2601.08881-red'></a> 
<a href='https://yuci-gpt.github.io/TAG-MoE/'><img src='https://img.shields.io/badge/Project%20Page-homepage-green'></a>
<a href='https://huggingface.co/YUXU915/TAG-MoE'><img src='https://img.shields.io/badge/HuggingFace-weights-F9A825?logo=huggingface'></a>
<a href='https://huggingface.co/spaces/YUXU915/TAG-MoE'><img src='https://img.shields.io/badge/HuggingFace-demo-4CAF50?logo=huggingface'></a>

![](https://raw.githubusercontent.com/yuci-gpt/TAG-MoE/refs/heads/master/static/images/teaser.png)

> **Abstract**:<br>
> Unified image generation and editing models suffer from severe task interference in dense diffusion transformers architectures, where a shared parameter space must compromise between conflicting objectives (e.g., local editing v.s. subject-driven generation). While the sparse Mixture-of-Experts (MoE) paradigm is a promising solution, its gating networks remain task-agnostic, operating based on local features, unaware of global task intent. This task-agnostic nature prevents meaningful specialization and fails to resolve the underlying task interference. In this paper, we propose a novel framework to inject semantic intent into MoE routing. We introduce a Hierarchical Task Semantic Annotation scheme to create structured task descriptors (e.g., scope, type, preservation). We then design Predictive Alignment Regularization to align internal routing decisions with the task's high-level semantics. This regularization evolves the gating network from a task-agnostic executor to a dispatch center. Our model effectively mitigates task interference, outperforming dense baselines in fidelity and quality, and our analysis shows that experts naturally develop clear and semantically correlated specializations.

---

## 🔧 Environment Setup

We recommend using [uv](https://docs.astral.sh/uv/) with the provided `pyproject.toml` / `uv.lock`.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create and activate virtual environment

```bash
git clone https://github.com/ICTMCG/TAG-MoE.git && cd TAG-MoE
uv venv --python 3.12 --python-preference only-managed
source .venv/bin/activate
```

### 3. Install dependencies

```bash
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126 uv sync
```

---

## 📦 Model Weights

- **Base model**: [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- **TAG-MoE weights**: [YUXU915/TAG-MoE](https://huggingface.co/YUXU915/TAG-MoE)

---

## 🚀 Inference

> **Note**:
>
> - TAG-MoE inference requires **60GB+ available VRAM**. We ran inference tests on 2× A100 40GB GPUs.
> - We leverage a VLM model to refine the instruction according to the input for better result.

```bash
uv run python infer.py \
    --pretrained_model_path Qwen/Qwen-Image \
    --transformer_model_path YUXU915/TAG-MoE \
    --device 0,1 \
    --image input.png \
    --prompt "Change the weather to sunny, remove the umbrella, the girl's hands hung naturally at her sides" \
    --output result.png
```

### WebUI

```bash
uv run python run_gradio.py \
    --pretrained_model_path Qwen/Qwen-Image \
    --transformer_model_path YUXU915/TAG-MoE \
    --device 0,1
```

---

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@misc{xu2026tagmoetaskawaregatingunified,
      title={TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts},
      author={Yu Xu and Hongbin Yan and Juan Cao and Yiji Cheng and Tiankai Hang and Runze He and Zijin Yin and Shiyi Zhang and Yuxin Zhang and Jintao Li and Chunyu Wang and Qinglin Lu and Tong-Yee Lee and Fan Tang},
      year={2026},
      eprint={2601.08881},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.08881},
}
```

---

## 🙏 Acknowledgements

This project builds upon the following excellent open-source works:

- **Diffusers** — https://github.com/huggingface/diffusers
- **MegaBlocks** — https://github.com/databricks/megablocks
