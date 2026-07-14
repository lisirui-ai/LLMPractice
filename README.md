<div align="center">

# LLMPractice

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.12.1+cu130-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/PEFT-LoRA%20%7C%20Adapter%20%7C%20Prefix-8A2BE2?style=flat-square" />
  <img src="https://img.shields.io/badge/LlamaFactory-%E2%89%A50.9.0-FF6F00?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

<p>基于 PyTorch + HuggingFace 的大语言模型实战系列</p>
<p>以 <b>位置编码 → PEFT 方法 → 模型微调 → 架构复现 → 优化器</b> 为主线，循序渐进覆盖 LLM 核心技术栈</p>
<p>每个 Notebook / 脚本均配有详细中文注释，适合大模型入门与进阶学习</p>

</div>

---

## 📋 目录

- [项目概览](#-项目概览)
- [目录结构](#-目录结构)
- [Notebook 介绍](#-notebook-介绍)
- [数据集](#-数据集)
- [学习路径](#-学习路径)
- [环境依赖](#️-环境依赖)
- [快速开始](#-快速开始)

---

## 🗺 项目概览


| #   | Notebook / 脚本                                                  | 微调框架                 | 核心技术                                                                                               | 亮点                                                 |
| --- | -------------------------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| 01  | 位置编码：相对位置编码 & RoPE                                             | —                    | Relative Position Encoding · RoPE · 旋转矩阵 · 复数相对位置依赖                                                | 从零推导并实现两种现代位置编码，可视化位置偏置                            |
| 02  | PEFT 方法对比讲解（无框架手动实现）                                           | 无框架（纯 PyTorch 手写）    | BERT · SST-2 · Frozen · Full FT · BitFit · P-Tuning · Prefix Tuning · P-Tuning v2 · LoRA · Adapter | 八种 PEFT 方法横向对比，参数量 vs. 效果权衡直观展示                    |
| 03  | ChatGLM3-6B 本地推理                                               | —                    | ChatGLM3-6B · HuggingFace · float16 · 多轮对话                                                         | Colab / 本地 GPU 一键推理，模型结构查看与参数统计                    |
| 04  | ChatGLM3-6B LoRA 微调（PEFT 框架 & 命令行脚本）                           | **HuggingFace PEFT** | LoRA · `peft` 库 · AdvertiseGen · AutoDL · 命令行调用                                                    | Notebook 04 为主入口，内部通过命令行调用 04.1 微调、04.2 推理    |
| 05  | LlamaFactory LoRA 全流程向导                                        | **LlamaFactory**     | LlamaFactory · LoRA · Alpaca / ShareGPT 格式 · YAML 配置 · 权重合并                                        | 微调→推理→预测→评估→权重合并完整端到端向导，通用参考手册                     |
| 06  | LlamaFactory + DeepSeek-R1-Distill-Llama-8B（准备→微调→推理→ROUGE 评估） | **LlamaFactory**     | LlamaFactory · LoRA · DeepSeek-R1-Distill-Llama-8B · ROUGE · StripThink                            | 四子 Notebook 完整链路：数据准备 → LoRA 微调 → 推理对话 → 去思维链评估    |
| 07  | DeepSeek V3 架构实现（MLA + MoE + FP8）                              | —                    | MLA · MoE · FP8 Block-wise GEMM · YaRN · 分布式张量并行 · 官方 Triton Kernel                                | 从零复现 DeepSeek V3 完整推理架构，调用官方 `kernel.py` 实现 FP8 推理 |
| 08  | MuonClip 优化器                                                   | —                    | Muon · Nesterov 动量 · QK-Clip · Newton-Schulz 正交化                                                   | 教学简化版 MuonClip，融合 Muon 与 QK-Clip 两项技术              |
| 09  | DeepSeek V3.2 架构实现（LightningIndexer + MLA + MoE + FP8 + YaRN）  | —                    | LightningIndexer · MLA · MoE · FP8 · YaRN · 16384 长上下文                                             | 引入 LightningIndexer 稀疏检索模块，完整复现 V3.2 架构            |
| 10  | Attention Residuals 注意力残差聚合                                    | —                    | Full AttnRes · Block AttnRes · 伪查询矩阵 · RMSNorm · 可学习温度                                             | 实现并可视化跨层注意力残差聚合，对比标准 Transformer 残差连接              |


---

## 📁 目录结构

```
LLMPractice/
├── 📓 01.PositionEncoding_RelativePE_RoPE.ipynb
├── 📓 02.PEFTAopNoFrame_BertOnHuggingFace_LoadFromRemoteAndCached.ipynb
├── 📓 03.LocalInference_ChatGLM3-6BOnHuggingFace.ipynb
├── 🐍 04.1.FineTune_CallByCL_ChatGLM3.py
├── 🐍 04.2.inference_CallByCL_ChatGLM3.py
├── 📓 04.FineTune&Inference_PEFTFrame_LoRA_ChatGLM3OnHuggingFace_AutoDL.ipynb
├── 📓 05.LlamaFactory_LoRA_WorkFlow_Guide.ipynb
├── 📓 06.1.LlamaFactory_prepare.ipynb
├── 📓 06.2.LlamaFactory_FineTune_LoRA_DeepSeekR1DistillLlama8B_AutoDL.ipynb
├── 📓 06.3.LlamaFactory_Infer&Predict_DeepSeekR1DistillLlama8B_AutoDL.ipynb
├── 📓 06.4.ROUGE_Eval_StripThink_DeepSeekR1DistillLlama8B_AutoDL.ipynb
├── 📓 07.ModelArchitecture_DeepSeekV3_MLA&MoE_FP8.ipynb
├── 🐍 08.Optimizer_MuonClip_NesterovMomentum_QKClip.py
├── 📓 09.ModelArchitecture_DeepSeekV3.2_LightningIndexer+MLA&MoE_FP8&YaRN.ipynb
├── 📓 10.ModelArchitecture_AttentionResiduals_FullAttnRes&BlockAttnRes.ipynb
│
├── 📂 configs/                        # YAML 配置模板（LlamaFactory 五种流程 + PEFT 框架训练）
│   ├── lora_LlamaFactory_train_template.yaml   # LlamaFactory LoRA 训练配置模板
│   ├── lora_LlamaFactory_infer_template.yaml   # LlamaFactory LoRA 推理配置模板
│   ├── lora_LlamaFactory_predict_template.yaml # LlamaFactory LoRA 测试集预测配置模板
│   ├── lora_LlamaFactory_eval_template.yaml    # LlamaFactory LoRA 评估配置模板
│   ├── lora_LlamaFactory_merge_template.yaml   # LlamaFactory LoRA 权重合并配置模板
│   └── lora_PEFTFrame_train.yaml               # HuggingFace PEFT 框架 LoRA 训练配置（Notebook 04）
│
├── 📂 data/                           # 数据集目录（gitignored）
│   ├── sst-2/                  # [自动生成] SST-2 数据集（Notebook 02 首次运行自动下载）
│   ├── sst-2_tokenizer/        # [自动生成] BERT 分词器缓存（Notebook 02 首次运行自动下载）
│   ├── AdvertiseGen/           # [需手动下载] 广告文案生成原始数据集（Notebook 04 / 06）
│   └── AdvertiseGen_fix/       # [自动生成] ChatGLM3 格式数据（需先下载 AdvertiseGen/，Notebook 04 生成）
│
├── 📂 model/                          # 模型权重目录（gitignored）
│   ├── bert-base-uncased/             # [自动生成] BERT 基础模型（首次运行 Notebook 02 自动下载缓存）
│   ├── ChatGLM3/                      # [自动生成] ChatGLM3-6B 权重（首次运行 Notebook 03 自动下载缓存）
│   └── DeepSeekR1DistillLlama8BForLlamaFactory/  # [自动生成] DeepSeek-R1-Distill-Llama-8B（运行 06.1 自动下载）
│
├── 📂 output/                         # 微调输出目录，checkpoint 等（gitignored）
├── 📄 .gitignore
├── 📄 requirementsBase.txt            # 基础环境依赖（Notebook 01/02/03/05/07/08/09/10）
├── 📄 requirementsGLM.txt             # ChatGLM3 环境依赖（Notebook 04 系列）
├── 📄 requirementsR1DistillLlama8B.txt  # LlamaFactory + DeepSeek-R1 依赖（Notebook 06 系列）
├── 📄 LICENSE
└── 📄 README.md
```

---

## 📚 Notebook 介绍

### 01. 位置编码：相对位置编码 & RoPE

> `01.PositionEncoding_RelativePE_RoPE.ipynb`

从零推导并实现两种现代 Transformer 位置编码方案：**相对位置编码（Relative Position Encoding）** 通过可学习嵌入参数将 token 间相对距离注入注意力分数；**旋转位置编码（RoPE）** 借助复数旋转矩阵将绝对位置转化为相对位置依赖，是 LLaMA / GPT-NeoX / DeepSeek 等主流大模型的标准方案。


| 章节     | 内容                                                                              |
| ------ | ------------------------------------------------------------------------------- |
| 相对位置编码 | 相对距离矩阵构建 · 可学习嵌入参数（2L-1 种距离） · 注意力分数融合 · 输出验证                                   |
| 旋转位置编码 | RoPE 频率计算（θ_d=10000^{-2d/dim}） · 旋转矩阵构造 · Q/K 旋转注入 · 相对位置点积验证 · YaRN/ABF 长度外推概述 |


> **RoPE 核心原理** · 对每个 token 位置 m，将 Q/K 向量的相邻两维视为复数对，乘以旋转角 θ_{m,d}=m×10000^{-2d/dim}；旋转后 Q·Kᵀ 点积天然包含两者相对位置差 (m-n)，无需加法注入位置信息，也无须额外可学习参数。
>
> 参考论文：*RoFormer: Enhanced Transformer with Rotary Position Embedding*（Su et al., 2021）

---

### 02. PEFT 方法对比讲解（无框架手动实现）

> `02.PEFTAopNoFrame_BertOnHuggingFace_LoadFromRemoteAndCached.ipynb`

以 **BERT + SST-2 情感分类**任务为实验平台，从零手动实现并横向对比八种主流参数高效微调方法（Frozen / Full FT / BitFit / P-Tuning / Prefix Tuning / P-Tuning v2 / LoRA / Adapter Tuning），直观展示各方法在可训练参数量与模型效果之间的权衡。支持 HuggingFace 远程加载与本地缓存两种模式。


| 章节             | 内容                                            |
| -------------- | --------------------------------------------- |
| 准备工作           | SST-2 数据集加载 · BERT 分词器 · DataLoader 构建 · 评估函数 |
| 冻结特征提取（Frozen） | 冻结全部 BERT 参数，仅训练分类头                           |
| 全量微调（Full FT）  | 解冻全部参数，作为效果上界基准                               |
| BitFit         | 仅微调所有 Bias 项，参数量极小                            |
| P-Tuning       | 在输入层插入可学习连续提示向量                               |
| Prefix Tuning  | 在每个注意力层的 K/V 前拼接可学习前缀                         |
| P-Tuning v2    | 全层 Prefix + 重参数化 MLP 生成前缀                     |
| LoRA           | 对 Q/V 权重低秩分解（W = W₀ + BA），参数量极低               |
| Adapter Tuning | 在每个 Transformer 层后插入瓶颈 Adapter 模块             |


---

### 03. ChatGLM3-6B 本地推理实践

> `03.LocalInference_ChatGLM3-6BOnHuggingFace.ipynb`

演示如何在 Google Colab（推荐 V100 高显存）或本地 GPU 环境中加载 ChatGLM3-6B 模型并进行多轮对话推理。涵盖环境配置、模型加载（float16，约 12 GB 显存）、模型结构查看、参数统计及权重持久化。


| 章节          | 内容                                  |
| ----------- | ----------------------------------- |
| 克隆官方仓库      | 克隆 ChatGLM3 仓库 · 安装推理依赖             |
| 安装依赖 · 加载模型 | 环境安装 · float16 精度加载 · tokenizer 初始化 |
| 查看模型结构与统计   | 打印模型层次结构 · 统计总参数量                   |
| 多轮对话推理      | stream_chat / chat 接口调用 · 多轮上下文管理   |


> **运行建议**：推荐 V100（16 GB）或 A100；T4（16 GB）也可运行，但速度较慢。

---

### 04. ChatGLM3-6B LoRA 微调（HuggingFace PEFT 框架 & 命令行脚本）

> `04.FineTune&Inference_PEFTFrame_LoRA_ChatGLM3OnHuggingFace_AutoDL.ipynb`（主入口，可直接运行）
> `04.1.FineTune_CallByCL_ChatGLM3.py` · `04.2.inference_CallByCL_ChatGLM3.py`（由 Notebook 04 通过命令行调用，不可单独运行）

使用 **HuggingFace PEFT 库**（`peft`）对 ChatGLM3-6B 进行 LoRA 高效微调，任务为 **AdvertiseGen** 广告文案生成。PEFT 框架直接集成于 HuggingFace `transformers` 生态，通过 `get_peft_model` 在原始模型上注入 LoRA adapter，无需修改模型结构即可完成低秩分解微调。运行环境为 AutoDL（推荐 A10 / sm80 架构，显存 ≥ 24 GB）。Notebook 04 在内部通过命令行分别调用 `04.1` 完成微调训练、调用 `04.2` 完成推理。

> 💡 **与 06 系列的框架区别**：本系列（04）使用 **HuggingFace PEFT 库**，通过 Python API 手动配置 `LoraConfig`、调用 `Trainer` 完成微调，灵活度高、代码可见性强，适合深入理解 LoRA 微调原理；06 系列使用 **LlamaFactory 框架**，通过 YAML 配置文件 + 命令行一键完成相同流程，工程化程度更高，适合生产实践。


| 章节        | 内容                                             |
| --------- | ---------------------------------------------- |
| 准备数据集     | AdvertiseGen 格式转换 · 训练 / 验证集构建                 |
| LoRA 微调训练 | PEFT LoRA 配置 · 训练超参数 · 训练主循环 · checkpoint 保存   |
| 推理与对比     | 加载最优 checkpoint · LoRA adapter 合并 · 广告文案生成效果展示 |
| 命令行调用     | Notebook 04 通过命令行调用 `04.1` 执行微调训练 · 调用 `04.2` 执行推理 |


---

### 05. LlamaFactory LoRA 全流程实战向导

> `05.LlamaFactory_LoRA_WorkFlow_Guide.ipynb`

以**命令行示意**为主的综合参考向导，完整覆盖基于 **LlamaFactory 框架**的 LoRA 微调端到端全流程。LlamaFactory 采用 YAML 配置文件 + `llamafactory-cli` 命令行驱动模式，无需编写大量 Python 训练代码，支持数十种主流开源模型与多种微调策略（LoRA / QLoRA / 全量等），适合作为上手实践的一站式参考手册。硬件需求：显存 ≥ 16 GB（7B 模型 + LoRA + bf16），框架版本：LLaMA Factory ≥ 0.9.0。


| 章节        | 内容                                                                      |
| --------- | ----------------------------------------------------------------------- |
| 环境安装      | git clone LlamaFactory · pip 可编辑安装 · 版本验证                               |
| 数据集准备     | Alpaca 格式 · ShareGPT 格式 · `dataset_info.json` 注册                        |
| LoRA 微调训练 | YAML 配置文件详解 · `llamafactory-cli train` 命令 · 超参数说明                       |
| 对话推理      | `llamafactory-cli chat` · LoRA adapter 加载                               |
| 测试集预测     | `llamafactory-cli train`（predict 模式） · 生成 `generated_predictions.jsonl` |
| ROUGE 评估  | 计算 ROUGE-1 / ROUGE-2 / ROUGE-L                                          |
| 权重合并      | `llamafactory-cli export` · LoRA 与基座权重合并导出                              |


---

### 06. LlamaFactory + DeepSeek-R1-Distill-Llama-8B 完整实验链路

本系列由四个子 Notebook 构成，全程使用 **LlamaFactory 框架**（YAML 配置 + `llamafactory-cli` 命令行），覆盖从数据准备到评估的完整实验流程。目标任务与 04 系列相同（AdvertiseGen 广告文案生成），但模型从 ChatGLM3-6B 升级为 DeepSeek-R1-Distill-Llama-8B，框架从 HuggingFace PEFT 切换为 LlamaFactory，可直观对比两套微调工具链的工作方式。

> 💡 **与 04 系列的框架区别**：04 系列使用 **HuggingFace PEFT 库**，通过 Python API 手动组织训练循环；本系列（06）使用 **LlamaFactory 框架**，仅需编写 YAML 配置文件并执行 `llamafactory-cli train / chat / predict / export` 命令，框架自动处理数据加载、混合精度、梯度累积等细节，工程化程度更高。

#### 06.1 微调准备

> `06.1.LlamaFactory_prepare.ipynb`

完成 LoRA 微调前的全部准备工作，共五步：

1. **克隆 LlamaFactory 仓库**：`git clone --depth 1` 浅克隆训练框架
2. **数据集转换**：将 AdvertiseGen JSONL（`content` + `summary`）转换为 LlamaFactory Alpaca 格式（`instruction` + `input` + `output`），输出至 `data/AdvertiseGen/train_alpaca.json`
3. **模型下载**：调用 `huggingface_hub.snapshot_download` 拉取 DeepSeek-R1-Distill-Llama-8B，以**平铺（扁平）结构**将权重分片、配置文件、分词器等全部文件直接写入 `model/DeepSeekR1DistillLlama8BForLlamaFactory/`，使 LlamaFactory `train.yaml` 中的 `model_name_or_path` 可直接指向该目录
4. **注册训练集**：向 `LlamaFactory/data/dataset_info.json` 写入 `advertise_gen_train` 条目，供 `train.yaml` 的 `dataset` 字段引用
5. **采样测试集**：从训练集随机复制 1% 数据作为测试集，并注册至 `dataset_info.json`

#### 06.2 LoRA 微调训练

> `06.2.LlamaFactory_FineTune_LoRA_DeepSeekR1DistillLlama8B_AutoDL.ipynb`

使用 `llamafactory-cli train` 执行 LoRA 微调，以 AdvertiseGen 广告文案生成为任务目标。实时输出训练日志（loss、grad_norm、learning_rate 等），保存最优 checkpoint。

#### 06.3 推理与测试集预测

> `06.3.LlamaFactory_Infer&Predict_DeepSeekR1DistillLlama8B_AutoDL.ipynb`


| 章节     | 内容                                                       |
| ------ | -------------------------------------------------------- |
| 对话推理测试 | 训练前基模对话（基线） · 微调后 LoRA 模型对话 · 服装属性标签→广告文案效果对比            |
| 训练前预测  | 基模 predict 任务 · 生成 `generated_predictions.jsonl`（基线预测文件） |
| 训练后预测  | LoRA 模型 predict 任务 · 生成微调后预测文件，供 06.4 计算 ROUGE 对比        |


#### 06.4 去思维链后 ROUGE 评估

> `06.4.ROUGE_Eval_StripThink_DeepSeekR1DistillLlama8B_AutoDL.ipynb`

DeepSeek-R1 系列模型输出含 `<think>...</think>` 思维链标签。本 Notebook 使用正则表达式过滤 Think 标签，再调用 `rouge_chinese`（集成 jieba 分词）重新计算 ROUGE-1 / ROUGE-2 / ROUGE-L，得到更贴近实际答案质量的评估结果。

---

### 07. DeepSeek V3 模型架构实现（MLA + MoE + FP8）

> `07.ModelArchitecture_DeepSeekV3_MLA&MoE_FP8.ipynb`

从零手写 **DeepSeek V3** 完整推理架构，涵盖五大核心模块：


| 模块                                   | 说明                                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------------------ |
| **MLA**（Multi-head Latent Attention） | 多头潜在注意力，将 KV 压缩为低维潜在向量，大幅降低 KV Cache 显存占用                                            |
| **MoE**（Mixture of Experts）          | 混合专家，稀疏激活（Top-k 路由），提升参数量而不增加计算量；含路由分组与共享专家                                          |
| **FP8 量化**（Block-wise FP8 GEMM）      | 分块 FP8 矩阵乘法（block_size=128），调用 DeepSeek V3 官方 `kernel.py` 中的 Triton Kernel 降低显存并加速推理 |
| **YaRN**                             | 旋转位置编码外推方案，支持 128K 超长序列推理                                                            |
| **分布式并行**                            | 列 / 行并行线性层 + 词表并行嵌入，支持多 GPU 张量并行（`world_size` 可配）                                    |



| 章节              | 内容                                                                                                          |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| 运行环境准备          | Triton ≥ 3.1 检查 · 从 DeepSeek V3 官方仓库获取 `kernel.py`（wget / git clone） · 工作目录确认                               |
| 全局变量与 ModelArgs | `world_size` · `rank` · `block_size` · `gemm_impl` · `attn_impl` · 完整超参列表                                   |
| FP8 量化层         | 导入官方 `kernel.py`（`act_quant` · `fp8_gemm`） · Block-wise 缩放因子 · `ColumnParallelLinear` / `RowParallelLinear` |
| MLA 注意力         | KV 潜在压缩 · Q 低秩分解 · NoPE + RoPE 混合头 · KV Cache 分配 · absorb 模式                                                |
| MoE 前馈网络        | 路由专家 · 共享专家 · Top-k 门控 · 分组路由 · `route_scale` 归一化                                                           |
| Transformer 整体  | 嵌入层 · Pre-RMSNorm · MLA + MoE 堆叠 · YaRN RoPE · LM Head · 前向推理                                               |


> **运行要求**：L4 GPU（Colab Pro+ / AutoDL），PyTorch ≥ 2.6，Triton ≥ 3.1，需提前从 [DeepSeek-V3 官方仓库](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py) 获取 `kernel.py`。

---

### 08. MuonClip 优化器（Nesterov 动量 + QK-Clip）

> `08.Optimizer_MuonClip_NesterovMomentum_QKClip.py`

教学简化版 **MuonClip** 优化器，融合 Muon 与 QK-Clip 两项技术：

- **Muon**：核心思想是对梯度矩阵做 Newton-Schulz 正交化，使每步更新方向接近正交矩阵，避免方向冗余；本实现保留其 Nesterov 动量框架。
- **QK-Clip**：将 Q/K 权重更新量的 L2 范数限制在阈值以内，防止注意力分数因 Q/K 更新过大而膨胀，避免 softmax 退化为 one-hot 引发训练震荡。

包含完整的端到端训练演示（简单语言模型任务），可直接运行观察 loss 下降曲线与梯度裁剪效果。

---

### 09. DeepSeek V3.2 架构实现（LightningIndexer + MLA + MoE + FP8 + YaRN）

> `09.ModelArchitecture_DeepSeekV3.2_LightningIndexer+MLA&MoE_FP8&YaRN.ipynb`

在 V3 架构基础上引入 **LightningIndexer** 稀疏检索模块，完整复现 DeepSeek V3.2 的推理架构。LightningIndexer 通过 `index_head_dim` 与 `index_topk` 参数对 Key 做稀疏 Top-k 选择，降低长序列（最大 16384 Token）下的注意力计算量。包含完整的 `ModelArgs` 超参数定义（含 `q_lora_rank`、`kv_lora_rank`、`qk_nope_head_dim`、`qk_rope_head_dim`、YaRN 相关参数等），可作为 V3.2 架构的权威参考实现。

---

### 10. Attention Residuals 注意力残差聚合

> `10.ModelArchitecture_AttentionResiduals_FullAttnRes&BlockAttnRes.ipynb`

实现并可视化两种注意力残差聚合变体，替代标准 Transformer 中仅利用上一层输出的加法残差连接：


| 章节                     | 内容                                                                |
| ---------------------- | ----------------------------------------------------------------- |
| Full AttnRes 全注意力残差块   | 可学习伪查询矩阵（`queries`） · RMSNorm 归一化键向量 · 可学习温度系数 τ · Softmax 跨层加权聚合 |
| Block AttnRes 分块注意力残差块 | 将层按块分组，块内跨层聚合，块间串联传递                                              |
| 对比标准 Transformer       | Full AttnRes / Block AttnRes 与标准残差的输出分布对比                         |
| 单元测试                   | 验证 Full AttnRes 权重归一化（行和为 1）及数值正确性                                |
| 注意力权重可视化               | 热力图展示各层对前序层的注意力分布                                                 |


> **论文公式**：`scores_l = (w_l · RMSNorm(H_prev)) / τ`，`h_res = Σ α_{l,i} · H_prev[i]`，每层通过可学习查询向量自适应聚合任意历史层信息。

---

## 🗃 数据集

**SST-2** · 斯坦福情感树库（Stanford Sentiment Treebank）


| 属性   | 详情                                                                                                             |
| ---- | -------------------------------------------------------------------------------------------------------------- |
| 任务   | 二分类情感分析（正面 / 负面）                                                                                               |
| 数据规模 | 训练集 ~67,349 条 · 验证集 ~872 条                                                                                     |
| 获取方式 | **自动生成** · 首次运行 Notebook 02 时通过 HuggingFace `datasets` 自动下载并缓存至 `data/sst-2/` 和 `data/sst-2_tokenizer/`，无需手动操作 |
| 用途   | Notebook 02 PEFT 方法对比实验                                                                                        |


---

**AdvertiseGen** · 广告文案生成数据集


| 属性   | 详情                                                                                                                                                                                                                   |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 任务   | 输入：服装商品属性标签（格式：`属性类型#属性值*...`）→ 输出：自然语言广告文案                                                                                                                                                                          |
| 获取方式 | **需手动下载** · 从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或 [清华云盘](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载，解压后放置于 `data/AdvertiseGen/` |
| 衍生数据 | **自动生成** · 手动下载 `data/AdvertiseGen/` 后运行 Notebook 04，自动转换为 ChatGLM3 对话格式并输出至 `data/AdvertiseGen_fix/` |
| 用途   | Notebook 04 ChatGLM3-6B LoRA 微调（PEFT 框架） · Notebook 06 系列 DeepSeek-R1-Distill-Llama-8B LoRA 微调（LlamaFactory 框架）                                                                                                      |


---

## 🛤 学习路径

```
基础技术层
  01 位置编码（RelativePE / RoPE）
       ↓
  02 PEFT 方法对比（BERT + SST-2）
     └─ 无框架，纯 PyTorch 手写八种 PEFT 方法
       ↓
模型微调实践层（同一任务 AdvertiseGen，两套框架路线）
                                              ┌─────────────────────────────────────────┐
  03 ChatGLM3-6B 本地推理（推理入门）          │ 微调框架对比                              │
       ↓                                      │                                         │
  04 ChatGLM3-6B LoRA 微调                   │  HuggingFace PEFT 框架（04 系列）        │
     └─ 框架：HuggingFace PEFT 库             │  Python API · LoraConfig · Trainer      │
        工具：LoraConfig + Trainer            │  适合：深入理解微调原理                    │
        特点：代码可见 · 灵活可控              │                  VS                      │
       ↓                                      │  LlamaFactory 框架（05-06 系列）         │
  05 LlamaFactory LoRA 全流程向导（通用参考）  │  YAML 配置 + llamafactory-cli 命令行     │
     └─ 框架：LlamaFactory                   │  适合：工程化实践 · 快速上手              │
        工具：YAML 配置 + llamafactory-cli    └─────────────────────────────────────────┘
        特点：工程化 · 一键全流程
       ↓
  06 LlamaFactory + DeepSeek-R1-Distill-Llama-8B
     └─ 框架：LlamaFactory（同 05，进阶实战）
        06.1 准备 → 06.2 微调 → 06.3 推理与预测 → 06.4 去思维链 ROUGE 评估
       ↓
架构与算法深入层
  07 DeepSeek V3 架构（MLA + MoE + FP8 + YaRN + 分布式）
       ↓
  08 MuonClip 优化器（Nesterov 动量 + QK-Clip）
       ↓
  09 DeepSeek V3.2 架构（LightningIndexer + 完整实现）
       ↓
  10 Attention Residuals（Full AttnRes + Block AttnRes）
```

建议按编号顺序学习：先掌握位置编码与 PEFT 方法（01-02），再完成微调实战（03-06），最后深入架构复现与优化器原理（07-10）。**04 与 06 系列任务相近，建议对比学习：04 系列侧重通过 HuggingFace PEFT 库理解微调原理，06 系列侧重通过 LlamaFactory 掌握工程化实践。**

---

## ⚙️ 环境依赖

> **说明**：项目包含三套独立环境，请按需选择对应依赖文件。

**Notebook 01 / 02 / 03 / 05 / 07 / 08 / 09 / 10** · `requirementsBase.txt` · Python 3.14.5

| 包名             | 版本           | 用途                                       |
| -------------- | ------------ | ---------------------------------------- |
| `torch`        | 2.12.0+cu132 | 深度学习框架核心（CUDA 13.2）                      |
| `transformers` | 4.50.0       | HuggingFace 模型加载与推理                      |
| `peft`         | 0.19.1       | LoRA · Prefix Tuning · Adapter 等 PEFT 方法 |
| `accelerate`   | 1.14.0       | 多卡训练加速与混合精度支持                            |
| `datasets`     | 5.0.0        | HuggingFace 数据集加载（SST-2 等）               |

> Notebook 07 / 09 额外需要 Triton ≥ 3.1 及 [DeepSeek-V3 官方 `kernel.py`](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py)，建议在 Colab Pro+ / AutoDL L4 GPU 上运行。

---

**Notebook 04 系列**（ChatGLM3-6B LoRA 微调） · `requirementsGLM.txt` · Python 3.10.20

| 包名              | 版本           | 用途                                  |
| --------------- | ------------ | ----------------------------------- |
| `torch`         | 2.12.1+cu130 | 深度学习框架核心（CUDA 13.0）                 |
| `transformers`  | 4.40.0       | HuggingFace 模型加载与推理                 |
| `peft`          | 0.10.0       | LoRA · Prefix Tuning · Adapter 微调   |
| `accelerate`    | 1.14.0       | 多卡训练加速与混合精度支持                       |
| `datasets`      | 2.18.0       | HuggingFace 数据集加载                   |
| `sentencepiece` | 0.2.0        | 子词分词（ChatGLM3）                      |
| `rouge-chinese` | 1.0.3        | 中文 ROUGE 评估                         |
| `jieba`         | 0.42.1       | 中文分词                                |
| `safetensors`   | 0.8.0        | 模型权重安全加载与保存                         |

---

**Notebook 06 系列**（LlamaFactory + DeepSeek-R1-Distill-Llama-8B） · `requirementsR1DistillLlama8B.txt` · Python 3.12.3

| 包名             | 版本     | 用途                      |
| -------------- | ------ | ----------------------- |
| `transformers` | 4.56.1 | HuggingFace 模型加载与推理     |
| `peft`         | 0.18.1 | LoRA 微调支持               |
| `accelerate`   | 1.11.0 | 训练加速                    |

> 此外需安装 LlamaFactory 框架本体：
> ```bash
> git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
> cd LLaMA-Factory && pip install -e ".[torch,metrics]"
> ```

---

## 🚀 快速开始

**1. 克隆仓库**

```bash
git clone https://github.com/your-username/LLMPractice.git
cd LLMPractice
```

**2. 安装依赖（按需选择）**

Notebook 01 / 02 / 03 / 05 / 07 / 08 / 09 / 10：

```bash
pip install -r requirementsBase.txt
```

Notebook 04 系列（ChatGLM3-6B）：

```bash
pip install -r requirementsGLM.txt
```

Notebook 06 系列（LlamaFactory + DeepSeek-R1）：

```bash
pip install -r requirementsR1DistillLlama8B.txt
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]"
```

> ⚠️ `torch` 各文件均带 CUDA 后缀（`+cu130` / `+cu132`），如需 CPU 版本或其他 CUDA 版本，请参考 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/) 替换。

**3. 准备数据集**

- **SST-2**：无需手动下载，首次运行 Notebook 02 时自动通过 HuggingFace `datasets` 下载并缓存至 `data/sst-2/` 和 `data/sst-2_tokenizer/`。
- **AdvertiseGen**：需手动下载，从 [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) 或 [清华云盘](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) 下载，解压后放置于 `data/AdvertiseGen/`。`data/AdvertiseGen_fix/` 会在完成上述下载后运行 Notebook 04 自动生成。

**4. 准备模型权重（按需）**

- **ChatGLM3-6B**：通过 HuggingFace Hub 或本地加载（参见 Notebook 03 / 04）。
- **DeepSeek-R1-Distill-Llama-8B**：参见 `06.1.LlamaFactory_prepare.ipynb` 中的下载脚本。
- **DeepSeek V3 / V3.2**：Notebook 07 / 09 为架构复现，需另行获取 `kernel.py`（见 Notebook 内说明）。

**5. 启动 Jupyter**

```bash
jupyter notebook
```

按编号顺序打开 Notebook 开始学习即可。

---

## 📄 License

[MIT](LICENSE) © 2026