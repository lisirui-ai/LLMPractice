# 标准库导入：操作系统接口，用于路径操作、目录遍历等
import os
# jieba：中文分词库，用于将中文文本切分为词语列表，计算 ROUGE/BLEU 指标时使用
import jieba
# dataclasses：提供 @dataclass/@field 装饰器，自动生成 __init__、__repr__ 等方法
import dataclasses as dc
# functools：提供高阶函数工具，如 functools.partial（偏函数）、functools.cache（缓存）
import functools
# 从 collections.abc 导入抽象基类，用于类型注解
from collections.abc import Callable, Mapping, Sequence
# Path：面向对象的文件系统路径操作类，支持路径拼接、解析等
from pathlib import Path
# 类型注解相关：Annotated 用于为参数附加元数据，Any 表示任意类型，
# Optional 表示可为 None，Union 表示多类型联合
from typing import Annotated, Any, Optional, Union

# numpy：数值计算库，用于数组操作（均值计算、填充等）
import numpy as np
# ruamel.yaml：功能完整的 YAML 解析库，支持注释保留，用于加载配置文件
import ruamel.yaml as yaml
# torch：PyTorch 深度学习框架，提供张量运算和自动微分
import torch
# typer：基于类型注解的命令行接口构建库，自动生成 --help 文档
import typer
# datasets：HuggingFace 数据集加载和处理库
from datasets import Dataset, DatasetDict, NamedSplit, Split, load_dataset
# BLEU 评估指标：sentence_bleu 计算单句 BLEU 分数，SmoothingFunction 提供平滑方法
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
# peft：参数高效微调（PEFT）库，提供 LoRA、P-Tuning v2 等轻量微调方法
from peft import (
    PeftConfig,               # PEFT 配置基类，描述微调方法的超参数
    PeftModelForCausalLM,     # 封装了 PEFT adapter 的因果语言模型类
    get_peft_config,          # 从字典构建对应 PeftConfig 子类实例的工厂函数
    get_peft_model            # 将基础模型封装为 PEFT 模型（注入 adapter）的函数
)
# rouge_chinese：支持中文的 ROUGE 评估指标库
from rouge_chinese import Rouge
# torch.nn：PyTorch 神经网络模块，nn.Module 是所有神经网络层的基类
from torch import nn
# transformers：HuggingFace Transformers 库，提供预训练模型、分词器和训练工具
from transformers import (
    AutoModelForCausalLM,         # 自动识别并加载因果语言模型（如 GPT、ChatGLM）的类
    AutoTokenizer,                # 自动识别并加载对应分词器的类
    EarlyStoppingCallback,        # 早停回调：连续 N 次评估无改善时提前终止训练
    EvalPrediction,               # 封装评估预测结果的数据类（含 predictions 和 label_ids）
    GenerationConfig,             # 文本生成配置类，控制生成长度、采样策略等行为
    PreTrainedModel,              # 预训练模型基类，定义通用接口
    PreTrainedTokenizer,          # 预训练分词器基类（慢速版本，基于 Python 实现）
    PreTrainedTokenizerFast,      # 预训练分词器基类（快速版本，基于 Rust tokenizers 库）
    Seq2SeqTrainingArguments,     # Seq2Seq 训练参数配置类，继承自 TrainingArguments
    AutoConfig,                   # 自动从预训练模型目录或 Hub 加载模型配置的类
)
# DataCollatorForSeq2Seq：用于 Seq2Seq 任务的数据批次整理器，负责填充和张量转换
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
# Seq2SeqTrainer：HuggingFace 提供的 Seq2Seq 任务训练器，封装训练循环
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer

# ModelType：类型别名，模型可以是标准预训练模型或 PEFT 封装后的模型
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
# TokenizerType：类型别名，分词器可以是慢速或快速版本
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
# 创建 Typer CLI 应用实例（基于 Click，通过 Python 类型注解自动生成命令行界面）
# pretty_exceptions_show_locals=False：异常发生时不打印局部变量，防止模型权重等敏感信息泄露
# 后续使用 @app.command() 将函数注册为子命令；运行脚本时 Typer 自动解析参数并调用对应函数
app = typer.Typer(pretty_exceptions_show_locals=False)


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    """
    自定义 Seq2Seq 数据批次整理器，继承自 HuggingFace 的 DataCollatorForSeq2Seq。

    主要扩展：支持对 'output_ids' 字段进行独立填充，用于评估阶段分离输入序列
    和参考输出序列（评估时 input_ids 驱动生成，output_ids 作为参考答案）。

    参数（继承自父类 _DataCollatorForSeq2Seq）：
        tokenizer (PreTrainedTokenizer): 用于获取 pad_token_id 的分词器
        padding (bool/str): 填充策略，'longest' 表示填充到批次内最长序列
        return_tensors (str): 返回张量类型，'pt' 表示 PyTorch 张量
        pad_to_multiple_of (Optional[int]): 将填充长度对齐到该值的倍数（用于硬件优化）

    返回值：
        dict: 包含 'input_ids'、'attention_mask'、'labels' 等键的批次张量字典
    """

    def __call__(self, features, return_tensors=None):
        """
        对一批样本进行处理和填充，支持含 'output_ids' 的评估样本。

        参数：
            features (list[dict]): 样本列表，每个样本为包含 'input_ids'、'labels' 等键的字典；
                                   评估样本还可能包含 'output_ids' 键（存储参考答案）
            return_tensors (Optional[str]): 返回的张量类型，为 None 时使用父类默认值

        返回值：
            dict[str, torch.Tensor]: 经过填充的批次数据字典，各张量形状为 (batch_size, max_seq_len)
        """
        # 尝试从每个样本中提取 'output_ids' 字段，用于评估阶段的目标序列
        # output_ids 类型：Optional[list[list[int]]]，即可选的二维整数列表
        output_ids = (
            [feature['output_ids'] for feature in features]  # 提取各样本的 output_ids
            if 'output_ids' in features[0].keys()            # 仅当第一个样本含 'output_ids' 键时才提取
            else None                                         # 训练样本不含 output_ids，设为 None
        )
        if output_ids is not None:  # 若存在 output_ids，对其进行单独的填充处理
            # 计算批次内所有 output_ids 的最大长度，用于统一填充目标，类型：int
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:  # 若需要将长度对齐到某个值的倍数
                # 向上取整到 pad_to_multiple_of 的最近倍数，公式：ceil(x/m)*m（整数实现）
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:  # 遍历批次中的每个样本，逐一填充 output_ids
                # 计算当前样本需要补充的 padding token 数量，并生成 pad 列表，类型：list[int]
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])  # 缺少的长度
                )
                if isinstance(feature['output_ids'], list):  # output_ids 为 Python 列表时
                    # 列表拼接追加 padding，结果类型：list[int]，长度变为 max_output_length
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:  # output_ids 为 numpy 数组时
                    # numpy 拼接并转换为 int64 类型，形状：(max_output_length,)
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        # 调用父类 __call__ 处理 input_ids、labels 等字段的填充和张量转换
        # 返回类型：dict[str, torch.Tensor]，各张量形状为 (batch_size, max_seq_len)
        return super().__call__(features, return_tensors)


class Seq2SeqTrainer(_Seq2SeqTrainer):
    """
    自定义 Seq2Seq 训练器，继承自 HuggingFace 的 Seq2SeqTrainer。

    主要扩展：重写 prediction_step 方法，确保在生成模式下正确截除输入前缀，
    只保留模型实际生成的部分；并支持使用独立的 output_ids 作为评估参考标签。
    """

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys=None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        执行单步预测，处理生成任务中的输入/输出分离逻辑。

        参数：
            model (nn.Module): 正在进行评估的神经网络模型
            inputs (dict[str, Any]): 输入数据字典，含 'input_ids'、'attention_mask' 等键；
                                     评估时还可能包含 'output_ids' 键（参考答案）
            prediction_loss_only (bool): 若为 True，仅计算损失，不执行生成
            ignore_keys (Optional[list]): 模型输出中需要忽略的键列表
            **gen_kwargs: 透传给 model.generate() 的额外生成参数（如温度、top_p 等）

        返回值：
            tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
                - loss: 标量损失值（仅在 prediction_loss_only=True 时不为 None）
                - generated_tokens: 纯生成内容张量，形状 (batch_size, gen_len)，已去除输入前缀
                - labels: 参考标签张量，形状 (batch_size, label_len)
        """
        if self.args.predict_with_generate:  # 若使用生成模式进行预测评估
            # 从 inputs 中弹出 'output_ids'，避免其干扰父类的 prediction_step 逻辑
            # output_ids 类型：torch.Tensor，形状：(batch_size, output_seq_len)
            output_ids = inputs.pop('output_ids')
        # 保存原始输入序列，用于后续截取生成内容（去掉 prompt 前缀）
        # input_ids 类型：torch.Tensor，形状：(batch_size, input_seq_len)
        input_ids = inputs['input_ids']
        # 调用父类 prediction_step，得到损失值、完整生成序列和标签
        # generated_tokens 此时含输入前缀 + 生成内容，形状：(batch_size, input_len + gen_len)
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        # 截取生成内容，去除输入前缀（使用 input_ids.size()[1] 即输入长度作为起始索引）
        # 截取后 generated_tokens 形状：(batch_size, gen_len)
        generated_tokens = generated_tokens[:, input_ids.size()[1]:]
        if self.args.predict_with_generate:  # 若使用生成模式
            # 用预先保存的真实 output_ids 替换父类计算的 labels，作为评估参考
            labels = output_ids
        # 返回：损失值、纯生成 token 序列、参考标签
        return loss, generated_tokens, labels


def _resolve_path(path: Union[str, Path]) -> Path:
    """
    将输入路径解析为绝对路径。

    参数：
        path (Union[str, Path]): 输入路径，可以是字符串或 Path 对象，支持 '~' 展开

    返回值：
        Path: 展开用户目录后解析为绝对路径的 Path 对象
    """
    # expanduser() 将 '~' 展开为用户主目录，resolve() 转换为绝对路径并解析符号链接
    return Path(path).expanduser().resolve()


def _sanity_check(
        input_ids: Sequence[int],
        output_ids: Sequence[int],
        tokenizer: PreTrainedTokenizer,
):
    """
    对编码后的输入/输出 ID 序列进行健全性检查，打印每个 token 对应的文本和标签 ID。

    用于验证数据预处理是否正确，特别是 loss_mask 是否按预期屏蔽了非 assistant 的 token。

    参数：
        input_ids (Sequence[int]): 编码后的完整对话 token ID 序列，长度为 seq_len
        output_ids (Sequence[int]): 对应的标签序列，-100 表示该位置不参与 loss 计算，长度为 seq_len
        tokenizer (PreTrainedTokenizer): 用于将 token ID 解码为文本的 ChatGLM3 分词器
    """
    print('--> Sanity check')  # 打印健全性检查标题
    for in_id, out_id in zip(input_ids, output_ids):  # 并行遍历输入 ID 和对应标签 ID
        if in_id == 0:  # 跳过 padding token（ID=0 表示填充位置，无实际含义）
            continue
        # 判断当前 token 是否为特殊 token（如 [gMASK]、sop、<|user|> 等）
        if in_id in tokenizer.tokenizer.index_special_tokens:
            # 从特殊 token 索引表中获取其文本表示，类型：str
            in_text = tokenizer.tokenizer.index_special_tokens[in_id]
        else:
            # 将普通 token ID 解码为文本字符串，类型：str
            in_text = tokenizer.decode([in_id])
        # 格式化打印：token 文本（右对齐 20 字符）：输入 ID -> 标签 ID（-100 表示不计算 loss）
        print(f'{repr(in_text):>20}: {in_id} -> {out_id}')


@functools.cache
def _get_yaml_parser() -> yaml.YAML:
    """
    获取（并缓存）YAML 解析器实例（单例模式）。

    使用 @functools.cache 装饰器确保解析器只被初始化一次，
    避免多次加载配置文件时重复创建解析器的开销。

    返回值：
        yaml.YAML: 配置好的 ruamel.yaml 解析器实例，
                   使用 'safe' 类型（禁止执行任意 Python 对象构造，安全读取）
    """
    # typ='safe' 使用安全加载模式，pure=True 使用纯 Python 实现（无 C 扩展依赖）
    parser = yaml.YAML(typ='safe', pure=True)
    # 设置输出缩进格式：映射和偏移各 2 格，序列缩进 4 格
    parser.indent(mapping=2, offset=2, sequence=4)
    # 禁用流式风格，使用块状（多行）风格输出，提高可读性
    parser.default_flow_style = False
    return parser  # 返回配置好的解析器，类型：yaml.YAML


@dc.dataclass
class DataConfig(object):
    """
    数据配置数据类，指定训练、验证和测试数据集的文件路径及处理参数。

    属性：
        train_file (str): 训练集文件路径（必填），支持 .csv、.json、.jsonl 格式
        val_file (Optional[str]): 验证集文件路径（可选），为 None 时跳过评估阶段
        test_file (Optional[str]): 测试集文件路径（可选），为 None 时跳过测试阶段
        num_proc (Optional[int]): 数据处理并行进程数（可选），为 None 时使用单进程
    """
    train_file: str                    # 训练集文件路径，类型：str，必填字段
    val_file: Optional[str] = None     # 验证集文件路径，类型：Optional[str]，默认 None
    test_file: Optional[str] = None    # 测试集文件路径，类型：Optional[str]，默认 None
    num_proc: Optional[int] = None     # 并行进程数，类型：Optional[int]，默认 None（单进程）

    @property
    def data_format(self) -> str:
        """
        获取数据文件格式（文件扩展名，含点号前缀）。

        返回值：
            str: 训练文件的扩展名，例如 '.csv'、'.json'、'.jsonl'
        """
        # 从训练文件路径提取文件扩展名（含点号），类型：str，如 '.json'
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        """
        构建数据集划分名称到文件路径的映射字典，仅包含非 None 的文件路径。

        返回值：
            dict[NamedSplit, str]: 数据集划分枚举值到文件路径的映射，
                                   键类型为 NamedSplit（如 Split.TRAIN），值类型为 str
        """
        # 将三种划分与对应文件路径配对，过滤掉文件路径为 None 的项
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],      # 三种数据集划分枚举值
                [self.train_file, self.val_file, self.test_file],  # 对应的文件路径
            )
            if data_file is not None  # 过滤掉未指定（None）的文件路径
        }


@dc.dataclass
class FinetuningConfig(object):
    """
    微调配置数据类，聚合数据配置、序列长度限制、训练参数和 PEFT 配置。

    属性：
        data_config (DataConfig): 数据文件配置，指定训练/验证/测试集路径
        max_input_length (int): 输入序列最大 token 长度，超出部分将被截断
        max_output_length (int): 输出序列最大 token 长度，超出部分将被截断
        training_args (Seq2SeqTrainingArguments): HuggingFace 训练参数，控制训练行为
        peft_config (Optional[PeftConfig]): PEFT 配置（LoRA/P-Tuning），None 表示全参数微调
    """
    data_config: DataConfig          # 数据配置，类型：DataConfig
    max_input_length: int            # 输入序列最大 token 长度，类型：int
    max_output_length: int           # 输出序列最大 token 长度，类型：int
    # 训练参数，类型：Seq2SeqTrainingArguments，默认输出目录为 './output'
    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output')
    )
    peft_config: Optional[PeftConfig] = None  # PEFT 配置，类型：Optional[PeftConfig]，默认 None

    def __post_init__(self):
        """
        数据类初始化后的后处理钩子，自动调整评估相关配置的一致性。

        - 若不进行评估或未提供验证文件，则完全禁用评估阶段
        - 若进行评估且未单独指定评估批次大小，则复用训练批次大小
        """
        if not self.training_args.do_eval or self.data_config.val_file is None:
            # 未启用评估或未提供验证文件时，禁用评估阶段的所有相关配置
            self.training_args.do_eval = False              # 关闭评估开关
            self.training_args.evaluation_strategy = 'no'  # 评估策略设为 'no'（不在训练中评估）
            self.data_config.val_file = None                # 清除验证文件路径，避免后续误用
        else:
            # 若未单独指定每设备评估批次大小，则默认复用训练批次大小
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size  # 已指定则保持原值
                    or self.training_args.per_device_train_batch_size  # 否则使用训练批次大小
            )

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        """
        从关键字参数字典构建 FinetuningConfig 实例，自动进行嵌套对象的类型转换。

        会将字典形式的 training_args、data_config、peft_config 自动转换为对应的配置对象。

        参数：
            **kwargs: 包含配置键值对的字典，字段名与 FinetuningConfig 的属性名对应

        返回值：
            FinetuningConfig: 构建好的微调配置实例
        """
        # 获取 training_args 字段，可能是原始字典或已实例化的 Seq2SeqTrainingArguments
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments  # 若尚未转换为 Seq2SeqTrainingArguments
        ):
            # 从 training_args 字典中单独取出 generation_config 子配置
            gen_config = training_args.get('generation_config')
            # 若 generation_config 还是字典形式，则将其实例化为 GenerationConfig 对象
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config  # 将字典解包为 GenerationConfig 的构造参数
                )
            # 将 training_args 字典转换为 Seq2SeqTrainingArguments 实例
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        # 获取 data_config 字段，可能是字典或已实例化的 DataConfig 对象
        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):  # 若还不是 DataConfig 实例
            # 将字典解包为 DataConfig 构造参数，创建 DataConfig 实例
            kwargs['data_config'] = DataConfig(**data_config)

        # 获取 peft_config 字段，可能是字典或已实例化的 PeftConfig 对象
        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            # 使用 get_peft_config 工厂函数，根据 peft_type 字段自动创建对应的 PeftConfig 子类
            kwargs['peft_config'] = get_peft_config(peft_config)
        # 使用处理后的 kwargs 构建并返回 FinetuningConfig 实例
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        """
        从 YAML 配置文件加载并构建 FinetuningConfig 实例。

        参数：
            path (Union[str, Path]): YAML 配置文件路径，支持相对路径和 '~' 展开

        返回值：
            FinetuningConfig: 从文件加载并转换好的微调配置实例
        """
        # 将输入路径解析为绝对 Path 对象，处理相对路径和 '~' 符号
        path = _resolve_path(path)
        # 使用缓存的 YAML 解析器加载配置文件，返回 Python 字典，类型：dict
        kwargs = _get_yaml_parser().load(path)
        # 调用 from_dict 将字典逐层转换为对应的配置对象
        return cls.from_dict(**kwargs)


def _load_datasets(
        data_dir: Path,
        data_format: str,
        data_files: dict[NamedSplit, str],
        num_proc: Optional[int],
) -> DatasetDict:
    """
    根据数据格式从指定目录加载多个数据集划分。

    参数：
        data_dir (Path): 数据文件所在的目录路径（绝对路径）
        data_format (str): 数据文件格式，支持 '.csv'、'.json'、'.jsonl'（含点号前缀）
        data_files (dict[NamedSplit, str]): 数据集划分枚举值到文件名的映射
        num_proc (Optional[int]): 数据加载的并行进程数，为 None 时使用默认单进程

    返回值：
        DatasetDict: 包含各数据集划分的字典，键为 NamedSplit，值为对应的 Dataset 对象

    异常：
        NotImplementedError: 当 data_format 不在支持的格式列表中时抛出
    """
    if data_format in ('.csv', '.json', '.jsonl'):  # 判断文件格式是否在支持范围内
        # 调用 HuggingFace datasets 的 load_dataset 函数加载数据
        # data_format[1:] 去掉前缀点号，得到 'csv'、'json' 或 'jsonl' 字符串
        dataset_dct = load_dataset(
            data_format[1:],        # 文件格式标识符，类型：str
            data_dir=data_dir,      # 数据目录路径，类型：Path
            data_files=data_files,  # 划分名到文件名的映射，类型：dict[NamedSplit, str]
            num_proc=num_proc,      # 并行进程数，类型：Optional[int]
        )
    else:
        # 不支持的文件格式，构造错误信息并抛出 NotImplementedError
        err_msg = f"Cannot load dataset in the '{data_format}' format."
        raise NotImplementedError(err_msg)

    # 返回加载好的数据集字典，类型：DatasetDict
    return dataset_dct


class DataManager(object):
    """
    数据管理器，负责加载原始数据集并提供经处理函数映射后的数据集访问接口。

    封装了数据集的加载和 map 处理逻辑，支持多进程并行处理以加速数据预处理。

    属性：
        _num_proc (Optional[int]): 数据处理的并行进程数
        _dataset_dct (DatasetDict): 原始数据集字典，包含 train/validation/test 等划分
    """

    def __init__(self, data_dir: str, data_config: DataConfig):
        """
        初始化数据管理器，从指定目录加载所有数据集划分。

        参数：
            data_dir (str): 数据文件所在的根目录路径（字符串形式）
            data_config (DataConfig): 数据配置对象，包含文件路径和处理参数
        """
        # 保存并行进程数配置，后续 map 时使用，类型：Optional[int]
        self._num_proc = data_config.num_proc
        # 调用 _load_datasets 加载所有数据集划分到内存
        # _dataset_dct 类型：DatasetDict，包含 train/validation/test 键
        self._dataset_dct = _load_datasets(
            _resolve_path(data_dir),        # 将 data_dir 解析为绝对 Path 对象
            data_config.data_format,        # 数据格式，如 '.json'
            data_config.data_files,         # 划分名到文件路径的映射
            self._num_proc,                 # 并行进程数
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        """
        从内部数据集字典中获取指定划分的原始 Dataset 对象。

        参数：
            split (NamedSplit): 数据集划分枚举值，如 Split.TRAIN、Split.VALIDATION

        返回值：
            Optional[Dataset]: 指定划分的 Dataset 对象，若该划分不存在则返回 None
        """
        # 使用字典 get 方法安全获取，不存在时返回 None，类型：Optional[Dataset]
        return self._dataset_dct.get(split, None)

    def get_dataset(
            self,
            split: NamedSplit,
            process_fn: Callable[[dict[str, Any]], dict[str, Any]],
            batched: bool = True,
            remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        """
        获取经过处理函数映射后的数据集（惰性 map 操作）。

        参数：
            split (NamedSplit): 数据集划分，如 Split.TRAIN、Split.VALIDATION、Split.TEST
            process_fn (Callable[[dict, dict]]): 数据处理函数，
                接受一批原始样本字典，返回包含新字段的处理后字典
            batched (bool): 是否以批处理模式调用 process_fn，默认 True（批量处理更高效）
            remove_orig_columns (bool): 是否删除原始列（处理函数的输入列），默认 True

        返回值：
            Optional[Dataset]: 经过 process_fn 映射处理后的 Dataset 对象，
                               若该划分不存在则返回 None
        """
        # 获取指定划分的原始 Dataset，不存在时返回 None，类型：Optional[Dataset]
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:  # 若该划分不存在，直接返回 None（跳过处理）
            return

        if remove_orig_columns:  # 若需要删除原始列
            # 获取原始数据集所有列名，用于 map 后删除，类型：list[str]
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None  # 不删除任何列
        # 使用 Dataset.map 应用处理函数，支持批处理和多进程
        # 返回类型：Dataset，包含 process_fn 输出的新字段（如 input_ids、labels）
        return orig_dataset.map(
            process_fn,                     # 数据处理函数（如 process_batch）
            batched=batched,                # 是否批处理模式
            remove_columns=remove_columns,  # 要删除的原始列名列表
            num_proc=self._num_proc,        # 并行进程数
        )


def print_model_size(model: PreTrainedModel):
    """
    打印模型中所有可训练参数的总量（以百万 M 为单位）。

    参数：
        model (PreTrainedModel): 需要统计参数量的预训练模型
    """
    print("--> Model")  # 打印模型信息标题分隔符
    # 统计所有 requires_grad=True（可训练）的参数的元素总数，类型：int
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 将参数总量除以 1e6（1,000,000）转换为百万单位，并格式化打印
    print(f"\n--> model has {total_params / 1e6}M params\n")


def process_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    """
    对一批对话样本进行 token 编码，用于训练阶段（含 loss mask）。

    将多轮对话转换为 input_ids 和 labels：
    - assistant 角色的回复部分参与 loss 计算（labels 保留真实 ID）
    - system/user 部分被 loss mask 屏蔽（labels 设为 -100）

    参数：
        batch (Mapping[str, Sequence]): 批量原始数据，必须包含 'conversations' 键；
                                        可选包含 'tools' 键（工具调用，暂未实现）
        tokenizer (PreTrainedTokenizer): ChatGLM3 分词器，需支持 get_command 和 build_single_message
        max_input_length (int): 输入序列最大 token 长度（超出部分截断）
        max_output_length (int): 输出序列最大 token 长度（超出部分截断）

    返回值：
        dict[str, list]: 包含以下键的字典：
            - 'input_ids': list[list[int]]，编码后的完整对话序列
            - 'labels': list[list[int]]，对应标签序列，非 assistant 位置为 -100
    """
    # 获取批次中的工具调用信息（工具调用场景），类型：Optional[Sequence]
    batched_tools = batch.get('tools', None)
    # 获取批次中所有对话的列表，类型：Sequence[list[dict]]，每个对话为消息字典的列表
    batched_conv = batch['conversations']
    # 初始化批量输入 ID 列表，训练后逐条填入，类型：list[list[int]]
    batched_input_ids = []
    # 初始化批量标签列表（含 loss mask 处理结果），类型：list[list[int]]
    batched_labels = []

    if batched_tools is None:  # 若无工具调用信息
        # 填充为 None 列表，使 zip 操作时与对话列表等长，类型：list[None]
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):  # 遍历每个样本的工具和对话
        # 初始化 input_ids，以 [gMASK]（生成掩码）和 sop（序列开始）两个特殊 token 开头
        # 对应的 loss_masks 初始均为 False，表示这两个位置不参与 loss 计算
        input_ids, loss_masks = [
            tokenizer.get_command('[gMASK]'),  # [gMASK] 特殊 token 的整数 ID，类型：int
            tokenizer.get_command('sop'),       # sop（start of piece）token 的整数 ID，类型：int
        ], [False, False]  # 两个特殊开头 token 均不计算 loss

        if tools is not None:  # 工具调用功能尚未实现
            raise NotImplementedError()

        # conv：单个样本的完整多轮对话，类型：list[dict]
        #       由 batched_conv（即 batch['conversations']）解包而来，
        #       每个元素是描述一条消息的字典，结构示例：
        #   [
        #       {'role': 'system',    'content': '你是一个助手...'},
        #       {'role': 'user',      'content': '你好'},
        #       {'role': 'assistant', 'content': '你好！有什么可以帮你的？'},
        #       ...
        #   ]
        for message in conv:  # 遍历当前对话中的每条消息
            # message：单条消息字典，类型：dict
            #   'role':    str — 消息角色，取值为 'system' / 'user' / 'assistant' / 'tool'
            #   'content': str — 消息的文本内容
            # 根据角色决定该消息是否参与 loss 计算
            if message['role'] in ('system', 'user'):
                loss_mask_val = False  # system 和 user 消息不计算 loss（仅作为上下文）
            else:
                loss_mask_val = True   # assistant 消息计算 loss（这是模型需要学习生成的部分）

            if message['role'] == 'tool':  # 工具角色暂未实现
                raise NotImplementedError()
            else:
                # 将单条消息编码为 token ID 序列，包含角色标记和消息内容
                # build_single_message 返回类型：list[int]
                new_input_ids = tokenizer.build_single_message(
                    message['role'],    # 角色标识字符串，如 'user'、'assistant'、'system'
                    '',                 # 元数据字符串（此处为空，ChatGLM3 格式要求）
                    message['content']  # 消息文本内容，类型：str
                )
                # 为该消息的每个 token 创建对应的 loss mask 值，类型：list[bool]
                new_loss_masks = [loss_mask_val] * len(new_input_ids)

            # 将新消息的 token ID 追加到当前样本的 input_ids 末尾
            input_ids += new_input_ids
            # 将新消息的 loss mask 追加到当前样本的 loss_masks 末尾
            loss_masks += new_loss_masks

        # 在序列末尾追加 EOS token，标记整个对话序列结束
        input_ids.append(tokenizer.eos_token_id)
        # loss_masks 在开头插入 False，实现标签序列右移一位
        # 目的：labels[i] 预测 input_ids[i+1]（语言模型的自回归目标）
        loss_masks = [False, *loss_masks]
        labels = []  # 初始化当前样本的标签列表
        for input_id, mask in zip(input_ids, loss_masks):  # 根据 mask 构建标签序列
            if mask:
                labels.append(input_id)    # 参与 loss 的位置保留真实 token ID
            else:
                labels.append(-100)        # 不参与 loss 的位置设为 -100（PyTorch 忽略该位置）
        # 计算允许的最大序列总长度（输入最大长度 + 输出最大长度 + 1 个偏移位）
        max_length = max_input_length + max_output_length + 1
        # 截断到最大长度并加入批次，input_ids 类型：list[int]
        batched_input_ids.append(input_ids[:max_length])
        # 截断到最大长度并加入批次，labels 类型：list[int]
        batched_labels.append(labels[:max_length])
    # 返回包含 input_ids 和 labels 的字典，供训练器使用
    return {'input_ids': batched_input_ids, 'labels': batched_labels}


def process_batch_eval(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    """
    对一批对话样本进行 token 编码，用于评估/推理阶段（输入/输出分开存储）。

    与训练阶段的 process_batch 的核心区别：
    - 输入（input_ids）：包含 user 问题 + assistant 的角色提示符，用于驱动模型生成
    - 输出（output_ids）：assistant 的真实回复 token ID，用于与生成结果对比计算评估指标

    参数：
        batch (Mapping[str, Sequence]): 批量原始数据，必须包含 'conversations' 键；
                                        可选包含 'tools' 键（工具调用，暂未实现）
        tokenizer (PreTrainedTokenizer): ChatGLM3 分词器
        max_input_length (int): 输入序列最大 token 长度
        max_output_length (int): 输出序列最大 token 长度

    返回值：
        dict[str, list]: 包含以下键的字典：
            - 'input_ids': list[list[int]]，驱动生成的输入序列（含 user 上下文和 assistant 提示符）
            - 'output_ids': list[list[int]]，assistant 真实回复的 token ID 序列（含 EOS token）
    """
    # 获取批次中的工具调用信息，类型：Optional[Sequence]
    batched_tools = batch.get('tools', None)
    # 获取批次中所有对话列表，类型：Sequence[list[dict]]
    batched_conv = batch['conversations']
    # 初始化批量输入 ID 列表（用于生成推理），类型：list[list[int]]
    batched_input_ids = []
    # 初始化批量输出 ID 列表（真实答案，不提供 labels 以避免意外的 loss 计算），类型：list[list[int]]
    batched_output_ids = []

    if batched_tools is None:  # 若无工具调用信息
        # 填充为 None 列表，类型：list[None]
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):  # 遍历每个样本
        # 初始化 input_ids，以 [gMASK] 和 sop 两个特殊 token 开头，类型：list[int]
        input_ids = [
            tokenizer.get_command('[gMASK]'),  # [gMASK] token ID，类型：int
            tokenizer.get_command('sop'),       # sop token ID，类型：int
        ]
        if tools is not None:  # 工具调用功能暂未实现
            raise NotImplementedError()

        for message in conv:  # 遍历当前对话中的每条消息
            if len(input_ids) >= max_input_length:  # 若已达到输入最大长度则停止处理后续消息
                break
            if message['role'] == 'tool':  # 工具角色暂未实现
                raise NotImplementedError()
            else:
                # 将当前消息编码为 token ID 序列，类型：list[int]
                new_input_ids = tokenizer.build_single_message(
                    message['role'],    # 角色标识
                    '',                 # 元数据（空字符串）
                    message['content']  # 消息内容
                )
                if message['role'] == 'assistant':  # 若是 assistant 的回复消息
                    # 将 assistant 的编码序列分为：提示符（第一个 token）和实际回复内容
                    output_prompt, output_ids = (
                        new_input_ids[:1],   # 角色提示符（assistant 的角色 token），类型：list[int]，长度 1
                        new_input_ids[1:],   # 实际回复内容（不含角色 token），类型：list[int]
                    )
                    # 在真实回复末尾追加 EOS token，标记生成结束位置
                    output_ids.append(tokenizer.eos_token_id)
                    # 将输入序列（截断到 max_input_length）+ assistant 提示符作为推理输入
                    # 模型将以此为上下文续写 assistant 的回复
                    batched_input_ids.append(
                        input_ids[:max_input_length] + output_prompt[:1]  # 截断并拼接提示符
                    )
                    # 将真实回复（截断到 max_output_length）存入输出列表，用于评估指标计算
                    batched_output_ids.append(output_ids[:max_output_length])
                # 将当前消息追加到 input_ids，为下一条消息提供上下文
                input_ids += new_input_ids
    # 返回包含 input_ids 和 output_ids 的字典，用于评估阶段
    return {'input_ids': batched_input_ids, 'output_ids': batched_output_ids}


def _prepare_model_for_training(model: nn.Module, use_cpu: bool):
    """
    为训练准备模型，将参与训练的参数转换为 float32 精度。

    混合精度训练中，float16 的可训练参数可能导致梯度下溢；
    CPU 训练时所有参数均需为 fp32（CPU 不支持 fp16 训练）。

    参数：
        model (nn.Module): 需要准备的神经网络模型
        use_cpu (bool): 是否使用 CPU 训练；为 True 时将所有参数转为 fp32
    """
    for param in model.parameters():  # 遍历模型的所有参数张量
        if param.requires_grad or use_cpu:  # 若参数可训练或使用 CPU 训练
            # 将参数数据原地转换为 float32 类型，确保梯度计算的数值稳定性
            # param.data 类型：torch.Tensor，转换后 dtype 为 torch.float32
            param.data = param.data.to(torch.float32)


def load_tokenizer_and_model(
        model_dir: str,
        peft_config: Optional[PeftConfig] = None,
) -> tuple[PreTrainedTokenizer, nn.Module]:
    """
    从指定目录加载分词器和模型，根据 PEFT 配置决定微调方式。

    支持三种模式：
    - 全参数微调：peft_config 为 None，加载完整模型，所有参数可训练
    - PREFIX_TUNING（P-Tuning v2）：修改配置加载模型，仅训练前缀编码器参数
    - LORA：加载基础模型后使用 get_peft_model 注入 LoRA adapter，仅训练低秩矩阵

    参数：
        model_dir (str): 预训练模型目录路径（含模型权重文件和 config.json）
        peft_config (Optional[PeftConfig]): PEFT 配置对象；为 None 时全参数微调

    返回值：
        tuple[PreTrainedTokenizer, nn.Module]:
            - tokenizer: 加载好的 ChatGLM3 分词器，类型：PreTrainedTokenizer
            - model: 加载（并可能封装）好的模型，类型：nn.Module
    """
    # 加载分词器，trust_remote_code=True 允许执行模型仓库中的自定义分词器代码
    # 返回类型：PreTrainedTokenizer（ChatGLM3 自定义分词器）
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if peft_config is not None:  # 若指定了 PEFT 配置，使用参数高效微调
        if peft_config.peft_type.name == "PREFIX_TUNING":  # P-Tuning v2 模式
            # 加载模型基础配置，以便在加载模型前修改 pre_seq_len 等参数
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            # 设置前缀虚拟 token 的数量（影响 prefix_encoder 的大小），类型：int
            config.pre_seq_len = peft_config.num_virtual_tokens
            # 禁用 KV cache，避免 prefix_encoder 与 past_key_values 的形状冲突
            config.use_cache = False
            # 使用修改后的配置加载模型，类型：PreTrainedModel
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,  # 允许自定义代码
                config=config,           # 注入修改后的配置（含 pre_seq_len）
            )
        if peft_config.peft_type.name == "LORA":  # LoRA 模式
            # 加载基础模型，empty_init=False 确保权重正常初始化而非随机（不使用空初始化加速）
            # use_cache=False 禁用 KV cache 以兼容梯度检查点，类型：PreTrainedModel
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,  # 允许自定义代码
                empty_init=False,        # 禁用空初始化，确保加载完整预训练权重
                use_cache=False          # 禁用 KV cache（梯度检查点需要）
            )
            # 使用 PEFT 库将基础模型封装为 LoRA 模型，注入低秩适配矩阵
            # 封装后只有 LoRA 参数（A、B 矩阵）需要训练，基础权重冻结
            # model 类型从 PreTrainedModel 升级为 PeftModelForCausalLM
            model = get_peft_model(model, peft_config)
            # 打印 LoRA 可训练参数占全部参数的比例，验证微调配置
            model.print_trainable_parameters()
    else:  # 全参数微调模式，加载完整模型，所有参数均可训练
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,  # 允许自定义代码
            empty_init=False,        # 不使用空初始化
            use_cache=False          # 禁用 KV cache
        )
    # 打印模型可训练参数的总量
    print_model_size(model)
    # 返回分词器和模型，类型：tuple[PreTrainedTokenizer, nn.Module]
    return tokenizer, model


def compute_metrics(eval_preds: EvalPrediction, tokenizer: PreTrainedTokenizer):
    """
    计算文本生成评估指标，包括 ROUGE-1/2/L 和 BLEU-4。

    使用 jieba 对中文文本进行分词后，分别计算 ROUGE 和 BLEU 指标，
    最终返回所有样本各指标的均值。

    参数：
        eval_preds (EvalPrediction): HuggingFace 评估预测封装对象，包含：
            - predictions: numpy 数组，形状 (num_samples, pred_len)，预测 token ID
            - label_ids: numpy 数组，形状 (num_samples, label_len)，参考答案 token ID
        tokenizer (PreTrainedTokenizer): 用于将 token ID 解码为文本的分词器

    返回值：
        dict[str, float]: 包含各评估指标均值的字典：
            - 'rouge-1': unigram 重叠的 F1 分数均值（乘以 100 转为百分比）
            - 'rouge-2': bigram 重叠的 F1 分数均值（乘以 100 转为百分比）
            - 'rouge-l': 最长公共子序列的 F1 分数均值（乘以 100 转为百分比）
            - 'bleu-4': 4-gram 精确匹配的 BLEU-4 分数均值（含 Method3 平滑处理）
    """
    # 从 EvalPrediction 对象中解包预测 ID 数组和标签 ID 数组
    # batched_pred_ids 类型：numpy.ndarray，形状：(num_samples, pred_len)
    # batched_label_ids 类型：numpy.ndarray，形状：(num_samples, label_len)
    batched_pred_ids, batched_label_ids = eval_preds

    # 初始化指标收集字典，每个键对应一个样本级别分数的列表
    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):  # 逐样本计算
        # 将预测 token ID 解码为文本，strip() 去除首尾空白字符，类型：str
        pred_txt = tokenizer.decode(pred_ids).strip()
        # 将参考标签 token ID 解码为文本，类型：str
        label_txt = tokenizer.decode(label_ids).strip()
        # 使用 jieba 对预测文本进行中文分词，返回词语列表，类型：list[str]
        pred_tokens = list(jieba.cut(pred_txt))
        # 使用 jieba 对参考文本进行中文分词，类型：list[str]
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()  # 创建 ROUGE 评估器实例
        # 计算 ROUGE 分数，输入为以空格连接的词语字符串（ROUGE 库要求空格分隔）
        # scores 类型：list[dict]，包含 'rouge-1'、'rouge-2'、'rouge-l' 三个子字典
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        for k, v in scores[0].items():  # 遍历三种 ROUGE 指标
            # v 包含 'f'（F1）、'p'（精确率）、'r'（召回率），取 F1 乘以 100 转为百分比
            metrics_dct[k].append(round(v['f'] * 100, 4))
        # 计算 BLEU-4 分数，Method3 平滑适合短文本（避免 0 分问题）
        # sentence_bleu 返回类型：float，范围 [0, 1]
        metrics_dct['bleu-4'].append(
            sentence_bleu(
                [label_tokens],                              # 参考文本词语列表（外层 list 支持多参考）
                pred_tokens,                                 # 预测文本词语列表
                smoothing_function=SmoothingFunction().method3,  # Method3 平滑，适合短文本
            )
        )
    # 对每个指标计算所有样本的均值，返回最终指标字典，值类型：numpy.float64
    return {k: np.mean(v) for k, v in metrics_dct.items()}


# @app.command() 将 main 函数注册为 CLI 子命令
# 执行 `python finetune_hf.py <data_dir> <model_dir> ...` 时，Typer 自动完成：
#   1. 解析并校验命令行参数（类型由函数签名中的 Annotated 注解决定）
#   2. 自动生成 --help 帮助文档（帮助文本来自各参数的 help= 字段）
#   3. 将解析后的参数注入 main 函数并执行
@app.command()
def main(
        data_dir: Annotated[str, typer.Argument(help='训练数据目录路径，包含格式化后的对话数据文件')],
        model_dir: Annotated[
            str,
            typer.Argument(
                help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
            ),
        ],
        config_file: Annotated[str, typer.Argument(help='微调配置文件路径（YAML 格式），包含训练参数和 PEFT 配置')],
        auto_resume_from_checkpoint: str = typer.Argument(
            default='',
            help='If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training'
        ),
):
    """
    主训练入口函数：加载配置与模型、准备数据集、构建训练器并启动微调。

    支持三种启动模式：
    1. 从头训练：auto_resume_from_checkpoint 为空或 None
    2. 自动从最新检查点恢复：auto_resume_from_checkpoint 为 'YES'
    3. 从指定编号检查点恢复：auto_resume_from_checkpoint 为正整数字符串（如 '4000'）

    参数：
        data_dir (str): 训练数据目录路径，包含 train/validation/test 数据文件
        model_dir (str): 预训练模型目录路径或 HuggingFace Hub 模型 ID
        config_file (str): YAML 格式的微调配置文件路径（如 configs/lora.yaml）
        auto_resume_from_checkpoint (str): 断点恢复策略字符串
    """
    # 从 YAML 配置文件解析并构建微调配置对象，类型：FinetuningConfig
    ft_config = FinetuningConfig.from_file(config_file)
    # 根据配置加载分词器和模型（自动处理 PEFT 封装逻辑）
    # tokenizer 类型：PreTrainedTokenizer，model 类型：nn.Module
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    # 创建数据管理器，加载并管理训练/验证/测试数据集
    data_manager = DataManager(data_dir, ft_config.data_config)

    # 获取训练集：应用 process_batch 将多轮对话转为带 loss mask 的 input_ids/labels
    # train_dataset 类型：Optional[Dataset]，含 'input_ids' 和 'labels' 两列
    train_dataset = data_manager.get_dataset(
        Split.TRAIN,  # 训练集划分
        functools.partial(  # 使用 partial 预绑定分词器和长度参数，减少重复传参
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,  # 批处理模式，大幅加速数据映射操作
    )
    print('train_dataset:', train_dataset)  # 打印训练集结构和样本数
    # 获取验证集：应用 process_batch_eval 将对话分为输入/输出（用于生成评估）
    # val_dataset 类型：Optional[Dataset]，含 'input_ids' 和 'output_ids' 两列
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,  # 验证集划分
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if val_dataset is not None:  # 若存在验证集则打印其结构信息
        print('val_dataset:', val_dataset)
    # 获取测试集：与验证集处理逻辑相同，用于最终性能评估
    # test_dataset 类型：Optional[Dataset]，含 'input_ids' 和 'output_ids' 两列
    test_dataset = data_manager.get_dataset(
        Split.TEST,  # 测试集划分
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:  # 若存在测试集则打印其结构信息
        print('test_dataset:', test_dataset)

    # 对训练集第一个样本进行健全性检查，验证编码正确性和 loss mask 的设置
    _sanity_check(
        train_dataset[0]["input_ids"],  # 第一个样本的输入 ID，类型：list[int]
        train_dataset[0]["labels"],     # 第一个样本的标签，类型：list[int]（-100 为屏蔽位置）
        tokenizer
    )

    # 将模型的可训练参数转换为 fp32，确保梯度计算的数值稳定性
    _prepare_model_for_training(model, ft_config.training_args.use_cpu)

    # 设置生成时使用的 pad token ID，批量生成时用于填充短序列
    ft_config.training_args.generation_config.pad_token_id = (
        tokenizer.pad_token_id  # 分词器的 padding token ID，类型：int
    )
    # 设置多个终止 token ID：遇到任意一个即停止生成，避免模型生成超出角色范围的内容
    ft_config.training_args.generation_config.eos_token_id = [
        tokenizer.eos_token_id,                      # 标准序列结束 token（正常对话结束）
        tokenizer.get_command('<|user|>'),            # 用户轮次开始符（防止模型代替用户发言）
        tokenizer.get_command('<|observation|>'),     # 工具观察结果开始符（工具调用场景）
    ]
    # 启用梯度检查点技术：丢弃中间激活值，反向传播时重新计算，以显存换计算速度
    model.gradient_checkpointing_enable()
    # 允许输入嵌入层接收梯度（梯度检查点要求输入层可微分）
    model.enable_input_require_grads()

    # 默认情况下 Trainer 使用分词器（用于生成和评估）
    use_tokenizer = True
    if ft_config.peft_config is not None:  # 若使用 PEFT 微调
        # LoRA 模式下分词器已内置于模型，不需要单独传给 Trainer
        use_tokenizer = False if ft_config.peft_config.peft_type == "LORA" else True

    # 创建自定义 Seq2Seq 训练器，整合模型、训练参数、数据集和评估逻辑
    trainer = Seq2SeqTrainer(
        model=model,                      # 要训练的模型（可能是 PEFT 封装后的）
        args=ft_config.training_args,     # 训练参数配置（学习率、批次大小、epoch 等）
        data_collator=DataCollatorForSeq2Seq(  # 自定义数据批次整理器
            tokenizer=tokenizer,
            padding='longest',      # 动态填充到批次内最长序列（节省计算资源）
            return_tensors='pt',    # 返回 PyTorch 张量格式
        ),
        train_dataset=train_dataset,  # 完整训练集
        eval_dataset=val_dataset.select(list(range(50))),  # 仅取前 50 条验证样本，加速评估
        tokenizer=tokenizer if use_tokenizer else None,    # LoRA 模式下不传分词器
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),  # 绑定分词器
        callbacks=[
            # 早停回调：连续 3 次评估（即 3 × eval_steps = 1500 步）监控指标无改善则停止训练
            # early_stopping_patience：触发停止所需的连续无改善评估次数，类型：int
            # early_stopping_threshold：指标提升幅度需超过该值才算"有改善"，类型：float
            #   设为 0.0 表示只要严格大于历史最佳即算改善，防止阈值过松导致过早停止
            EarlyStoppingCallback(
                early_stopping_patience=5,    # 连续 3 次评估（1500 步）无改善则停止
                early_stopping_threshold=0.0, # 严格大于历史最佳才算改善
            )
        ],
    )

    if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint is None:
        # 未指定断点恢复，直接从头开始训练
        trainer.train()
    else:
        def do_rf_checkpoint(sn):
            """
            从指定编号的检查点目录恢复训练。

            参数：
                sn (str): 检查点编号字符串，如 '4000'，对应目录名 'checkpoint-4000'
            """
            # 恢复训练时需重新启用梯度检查点（Trainer 内部会重置部分设置）
            model.gradient_checkpointing_enable()
            # 重新允许输入嵌入层接收梯度（梯度检查点的必要条件）
            model.enable_input_require_grads()
            # 拼接检查点目录的完整路径，类型：str
            checkpoint_directory = os.path.join(output_dir, "checkpoint-" + sn)
            print("resume checkpoint from  checkpoint-" + sn)  # 打印恢复来源信息
            # 从指定检查点目录恢复训练（包括优化器状态、调度器状态等）
            trainer.train(resume_from_checkpoint=checkpoint_directory)

        # 获取模型输出目录，用于查找检查点，类型：str
        output_dir = ft_config.training_args.output_dir

        if auto_resume_from_checkpoint.upper() == "YES":  # 自动选择最新检查点
            # 列出输出目录下的所有文件和子目录，类型：list[str]
            dirlist = os.listdir(output_dir)
            checkpoint_sn = 0  # 初始化最大检查点编号为 0，类型：int
            for checkpoint_str in dirlist:  # 遍历目录内容，找到编号最大的检查点
                # 筛选条件：目录名含 'checkpoint'（通过 'eckpoint' 匹配）且不含 'tmp'（排除临时目录）
                if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                    # 提取检查点编号（去掉 'checkpoint-' 前缀后转为整数），类型：int
                    checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                    if checkpoint > checkpoint_sn:  # 若找到更大的编号则更新
                        checkpoint_sn = checkpoint
            if checkpoint_sn > 0:  # 若找到有效的检查点（编号大于 0）
                # 从最新检查点恢复训练
                do_rf_checkpoint(str(checkpoint_sn))
            else:  # 未找到任何检查点，从头开始训练
                trainer.train()
        else:
            # 检查输入是否为有效的正整数字符串（指定特定检查点编号）
            if auto_resume_from_checkpoint.isdigit() and int(auto_resume_from_checkpoint) > 0:
                # 从用户指定编号的检查点恢复训练
                do_rf_checkpoint(auto_resume_from_checkpoint)
            else:
                # 输入的字符串既不是 'YES' 也不是有效数字，打印错误提示
                print(auto_resume_from_checkpoint,
                      "The specified checkpoint sn(" + auto_resume_from_checkpoint + ") has not been saved. Please search for the correct chkeckpoint in the model output directory")

    # 若存在测试集，在训练完成后使用最终模型进行预测评估，输出测试指标
    if test_dataset is not None:
        trainer.predict(test_dataset)  # 在测试集上运行推理并打印评估结果


if __name__ == '__main__':
    app()  # 启动 typer 命令行应用，解析命令行参数并路由到 main 函数
