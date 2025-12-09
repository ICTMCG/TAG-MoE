# TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts

> **TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts**<br>
> Yu Xu</a><sup>1,2†</sup>, Hongbin Yan</a><sup>1</sup>, Juan Cao</a><sup>1</sup>, Yiji Cheng</a><sup>2</sup>, Tiankai Hang</a><sup>2</sup>, Runze He</a><sup>2</sup>, Zijin Yin</a><sup>2</sup>, Shiyi Zhang</a><sup>2</sup>, Yuxin Zhang</a><sup>1</sup>, Jintao Li</a><sup>1</sup>, Chunyu Wang</a><sup>2‡</sup>, Qinglin Lu</a><sup>2</sup>, Tong-Yee Lee</a><sup>3</sup>, Fan Tang</a><sup>1§</sup> <br>
> <sup>1</sup>University of Chinese Academy of Sciences, <sup>2</sup>Tencent Hunyuan, <sup>3</sup>National Cheng-Kung University

![](https://raw.githubusercontent.com/yuci-gpt/TAG-MoE/refs/heads/master/static/images/teaser.png)


>**Abstract**: <br>
>Unified image generation and editing models suffer from severe task interference in dense diffusion transformers architectures, where a shared parameter space must compromise between conflicting objectives (e.g., local editing v.s. subject-driven generation). While the sparse Mixture-of-Experts (MoE) paradigm is a promising solution, its gating networks remain task-agnostic, operating based on local features, unaware of global task intent. This task-agnostic nature prevents meaningful specialization and fails to resolve the underlying task interference. In this paper, we propose a novel framework to inject semantic intent into MoE routing. We introduce a Hierarchical Task Semantic Annotation scheme to create structured task descriptors (e.g., scope, type, preservation). We then design Predictive Alignment Regularization to align internal routing decisions with the task's high-level semantics. This regularization evolves the gating network from a task-agnostic executor to a dispatch center. Our model effectively mitigates task interference, outperforming dense baselines in fidelity and quality, and our analysis shows that experts naturally develop clear and semantically correlated specializations.
