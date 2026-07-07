# 标准库导入：面向对象的文件系统路径操作类，支持路径拼接、解析和判断等
from pathlib import Path
# 类型注解相关：Annotated 用于为参数附加 CLI 帮助元数据，Union 表示多类型联合
from typing import Annotated, Union

# typer：基于类型注解的命令行接口构建库，自动生成 --help 文档和参数解析
import typer
# peft：参数高效微调库，AutoPeftModelForCausalLM 能自动识别并加载含 adapter 的 PEFT 模型
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
# transformers：HuggingFace Transformers 库，提供预训练模型和分词器
from transformers import (
    AutoModelForCausalLM,         # 自动识别并加载标准因果语言模型（GPT、ChatGLM 等）的类
    AutoTokenizer,                # 自动识别并加载对应分词器的类
    PreTrainedModel,              # 标准预训练模型基类，定义通用接口
    PreTrainedTokenizer,          # 慢速预训练分词器基类（纯 Python 实现）
    PreTrainedTokenizerFast,      # 快速预训练分词器基类（基于 Rust tokenizers 库）
)

# ModelType：类型别名，模型可以是标准预训练模型或 PEFT 封装后的 adapter 模型
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
# TokenizerType：类型别名，分词器可以是慢速或快速版本
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# 创建 typer 命令行应用实例，pretty_exceptions_show_locals=False 异常时不显示局部变量
app = typer.Typer(pretty_exceptions_show_locals=False)


def _resolve_path(path: Union[str, Path]) -> Path:
    """
    将输入路径解析为绝对路径。

    参数：
        path (Union[str, Path]): 输入路径，可以是字符串或 Path 对象，支持 '~' 用户目录展开

    返回值：
        Path: 展开用户目录后解析为绝对路径的 Path 对象
    """
    # expanduser() 将 '~' 展开为用户主目录路径
    # resolve() 将相对路径转换为绝对路径，并解析所有符号链接
    return Path(path).expanduser().resolve()


def load_model_and_tokenizer(model_dir: Union[str, Path]) -> tuple[ModelType, TokenizerType]:
    """
    从指定目录自动加载模型和分词器，同时支持标准模型和 PEFT 微调后的模型。

    自动检测模型目录中是否存在 'adapter_config.json' 文件：
    - 若存在：认为是经过 PEFT（如 LoRA）微调的模型，使用 AutoPeftModelForCausalLM 加载
    - 若不存在：认为是标准预训练模型，使用 AutoModelForCausalLM 加载

    分词器从基础模型路径加载（PEFT 模型从 adapter 配置中读取基础模型路径）。

    参数：
        model_dir (Union[str, Path]): 模型目录路径，可以是字符串或 Path 对象，
                                      支持相对路径和 '~' 展开

    返回值：
        tuple[ModelType, TokenizerType]:
            - model: 加载好的模型，类型为 ModelType（PreTrainedModel 或 PeftModelForCausalLM）
            - tokenizer: 加载好的分词器，类型为 TokenizerType
    """
    # 将输入路径解析为绝对 Path 对象，统一处理不同格式的输入，类型：Path
    model_dir = _resolve_path(model_dir)
    if (model_dir / 'adapter_config.json').exists():  # 检测是否为 PEFT 微调后的模型目录
        # 使用 AutoPeftModelForCausalLM 加载带 adapter 权重的因果语言模型
        # trust_remote_code=True 允许执行模型仓库中的自定义代码（ChatGLM3 需要）
        # device_map='auto' 自动将模型层分配到可用设备（GPU/CPU），支持多卡和 CPU offload
        # 返回类型：PeftModelForCausalLM（PEFT 封装的模型）
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        # 从 PEFT adapter 配置中获取基础模型的名称或路径，用于加载匹配的分词器
        # peft_config['default'] 访问默认 adapter 的配置，类型：PeftConfig
        # base_model_name_or_path 类型：str，可以是本地路径或 HuggingFace Hub 模型 ID
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:  # 标准预训练模型（无 adapter_config.json，未经 PEFT 微调）
        # 使用 AutoModelForCausalLM 加载标准因果语言模型
        # device_map='auto' 自动分配模型到可用设备
        # 返回类型：PreTrainedModel
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        # 标准模型的分词器与模型存储在同一目录下，类型：Path
        tokenizer_dir = model_dir
    # 从确定的分词器目录加载分词器
    # trust_remote_code=True 允许执行模型仓库中的自定义分词器代码（ChatGLM3 需要）
    # 返回类型：PreTrainedTokenizer 或 PreTrainedTokenizerFast
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=True
    )
    # 返回模型和分词器的二元元组，类型：tuple[ModelType, TokenizerType]
    return model, tokenizer


@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='模型目录路径，支持标准预训练模型和 PEFT 微调后的模型（含 adapter_config.json）')],
        prompt: Annotated[str, typer.Option(help='输入给模型的文本提示（prompt），模型将基于此生成一次对话回复')],
):
    """
    主推理函数：加载指定模型，对输入的 prompt 执行一次对话推理并打印生成结果。

    参数：
        model_dir (str): 模型目录路径（命令行位置参数），支持标准模型或微调后的 PEFT 模型
        prompt (str): 用户输入的提示文本（命令行选项 --prompt），模型将基于此生成回复
    """
    # 根据模型目录自动加载模型和分词器，自动检测是否为 PEFT 微调模型
    # model 类型：ModelType，tokenizer 类型：TokenizerType
    model, tokenizer = load_model_and_tokenizer(model_dir)
    # 调用 ChatGLM3 模型的 chat 方法进行单轮对话推理
    # chat 方法返回 tuple[str, list]：
    #   - 第一个元素 response 为生成的回复文本，类型：str
    #   - 第二个元素为对话历史列表，此处用 _ 忽略（单轮推理不需要历史）
    response, _ = model.chat(tokenizer, prompt)
    # 将模型生成的回复文本打印到标准输出，类型：str
    print(response)


if __name__ == '__main__':
    app()  # 启动 typer 命令行应用，解析命令行参数并调用 main 函数
