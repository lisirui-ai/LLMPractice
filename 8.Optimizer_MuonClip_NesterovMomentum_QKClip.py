# ============================================================
# 一、导入依赖
# ============================================================
import math  # 提供 sqrt()，用于注意力分数缩放：除以 sqrt(d_k) 可防止高维内积过大导致 softmax 饱和、梯度消失
import torch  # PyTorch 核心库，提供 Tensor、自动微分引擎（autograd）和设备抽象（CPU/CUDA）
import torch.nn as nn  # 神经网络模块，封装了 Linear（仿射变换）、MSELoss 等常用层；所有可学习层均继承自 nn.Module
from torch.optim.optimizer import Optimizer  # 优化器抽象基类；继承后可自动获得 param_groups 管理、zero_grad、state_dict 等标准接口


# ============================================================
# 二、MuonClip 优化器
# ============================================================
class MuonClip(Optimizer):
    """
    教学简化版 MuonClip 优化器，融合了 Muon 和 QK-Clip 两项技术。

    【Muon】
        来自 2024 年同名论文，核心思想是对梯度矩阵做 Newton-Schulz 正交化，
        使每步更新方向接近正交矩阵，避免梯度方向冗余。
        本实现为教学简化版，省略正交化步骤，仅保留其 Nesterov 动量框架。
        名称借用物理学中的 μ 介子（muon），暗示其"穿透力"更强。

    【QK-Clip】
        注意力机制中，Q 和 K 矩阵的更新量直接影响注意力分数的量级。
        若 Q/K 权重更新过大，点积 QK^T 的绝对值膨胀，softmax 趋近 one-hot，
        导致梯度稀疏、训练震荡。QK-Clip 将 Q/K 更新量的 L2 范数限制在阈值以内。

    参数
    ----------
    params : iterable
        待优化参数的迭代器，通常为 model.parameters()，内部元素为 nn.Parameter（即 Tensor 子类）
    lr : float，默认 3e-3
        学习率 α，控制每步参数变化的幅度；偏大有助于 demo 中观察到明显 loss 下降
    weight_decay : float，默认 0.0
        L2 正则化系数 λ；启用后等价于在损失上加 (λ/2)*||w||²，梯度中增加 λ*w 项，抑制参数过大
    momentum : float，默认 0.9
        动量系数 β ∈ (0,1)；β 越大，历史梯度方向保留越多，越能穿越平坦区域但也可能过冲
    nesterov : bool，默认 True
        是否使用 Nesterov 加速梯度（NAG）；普通动量先更新位置再算梯度，
        NAG 先"预判"动量方向再算梯度，修正更及时，同等轮数下收敛更快，与 Muon 原论文默认一致
    clip_threshold : float，默认 1.0
        QK-Clip 范数上限 τ；仅对挂载了 is_qk_projection=True 属性的参数生效；
        设为 1.0 较小，demo 中更容易触发裁剪以展示效果
    eps : float，默认 1e-8
        数值稳定项 ε，加在裁剪缩放因子的分母上防止 update_norm 为 0 时除零
    """

    # --------------------------------------------------------
    # 2.1 初始化
    # --------------------------------------------------------
    def __init__(
        self,
        params,            # 待优化参数集合，类型 iterable[Tensor]，通常传入 model.parameters()
        lr=3e-3,           # 学习率 α，类型 float，梯度下降步长；过小收敛慢，过大震荡甚至发散
        weight_decay=0.0,  # L2 正则系数 λ，类型 float；0.0 表示不施加正则，默认关闭以保持 demo 简洁
        momentum=0.9,      # 动量系数 β，类型 float，范围 (0,1)；0.9 意味着保留 90% 的历史动量方向
        nesterov=True,     # NAG 开关，类型 bool；True 时用"超前梯度"替代当前梯度，收敛速度快于普通动量
        clip_threshold=1.0,  # QK-Clip 阈值 τ，类型 float；更新量 L2 范数超过此值才裁剪，过大则等同于不裁剪
        eps=1e-8,          # 稳定项 ε，类型 float；防止裁剪时分母为零引发 NaN 或 Inf
    ):
        defaults = dict(           # 将全部超参打包为 dict，Optimizer 基类会将其绑定到每个参数组，支持按组设置不同超参
            lr=lr,                 # 存入学习率 α，step() 中通过 group["lr"] 读取，类型 float
            weight_decay=weight_decay,  # 存入 L2 正则系数 λ，类型 float；为 0.0 时不施加正则
            momentum=momentum,     # 存入动量系数 β，类型 float；控制历史梯度方向的保留比例
            nesterov=nesterov,     # 存入 NAG 开关，类型 bool；决定 step() 中是否使用超前梯度估计
            clip_threshold=clip_threshold,  # 存入 QK-Clip 范数上限 τ，类型 float；超过此值才裁剪
            eps=eps,               # 存入数值稳定项 ε，类型 float；防止裁剪时分母为零
        )
        super().__init__(params, defaults)  # 调用 Optimizer.__init__：解析 params 为 param_groups 列表，并将 defaults 写入每个组

    # --------------------------------------------------------
    # 2.2 单步参数更新
    # --------------------------------------------------------
    @torch.no_grad()  # 关闭自动微分图构建：step() 只做参数原地修改，不需要梯度，关掉可节省显存并加速约 10~30%
    def step(self, closure=None):
        """
        执行一次参数更新，完整更新规则如下：
            g ← g + λ·w                      （L2 weight_decay，可选）
            m ← β·m + g                       （动量累积，原地）
            u ← β·m + g  （Nesterov）          （超前位置梯度估计）
              或 u ← m   （普通动量）
            若 is_qk_projection 且 ‖u‖ > τ：
                u ← u · τ / (‖u‖ + ε)         （QK-Clip，保方向缩范数）
            w ← w - α·u                        （梯度下降）

        参数
        ----------
        closure : callable 或 None，默认 None
            可选闭包；某些算法（如 LBFGS）需要在 step 内多次重算 loss，通过闭包实现；
            普通训练循环无需传入

        返回值
        -------
        loss : Tensor（标量，shape=[]，dtype=float32） 或 None
            若传入 closure 则返回其计算的 loss 值；否则返回 None
        """
        loss = None  # 预设返回值为 None；只有调用方传入 closure 时才会被赋值为实际 loss Tensor
        if closure is not None:  # closure 存在时需要在其内部正向传播计算 loss，所以要临时恢复梯度
            with torch.enable_grad():  # @no_grad 已在外层关闭梯度，with enable_grad 仅在此块内重新开启，出块后自动恢复关闭状态
                loss = closure()       # 调用闭包执行前向传播，返回 loss Tensor，shape=[]（标量）

        for group in self.param_groups:  # param_groups 是 list[dict]，每个 dict 包含一组参数及其专属超参；支持不同层使用不同学习率
            lr              = group["lr"]              # 当前组的学习率 α，类型 float
            weight_decay    = group["weight_decay"]    # 当前组的 L2 系数 λ，类型 float
            momentum        = group["momentum"]        # 当前组的动量系数 β，类型 float
            nesterov        = group["nesterov"]        # 当前组的 NAG 开关，类型 bool
            clip_threshold  = group["clip_threshold"]  # 当前组的 QK-Clip 阈值 τ，类型 float
            eps             = group["eps"]             # 当前组的稳定项 ε，类型 float

            for p in group["params"]:  # 遍历当前参数组的每个参数张量 p，shape 与对应层权重矩阵相同，dtype=float32
                if p.grad is None:     # 未参与计算图的参数（如 frozen 层）不会有梯度，跳过可避免后续空指针
                    continue

                grad  = p.grad        # 当前参数的梯度，shape 与 p 完全相同；.backward() 后由 autograd 填入
                state = self.state[p] # 该参数专属的持久化状态字典 dict；首次访问时为空 {}，跨 step 调用保持数据

                if len(state) == 0:   # 空字典说明是该参数第一次更新，需要初始化所有状态变量
                    state["exp_avg"] = torch.zeros_like(p)  # 一阶动量缓冲区 m，初始为全零，shape 与 p 相同；zeros_like 保证设备和 dtype 与 p 一致

                exp_avg = state["exp_avg"]  # 读取动量缓冲区引用，shape 与 p 相同；后续原地操作会直接修改 state 中的张量

                if weight_decay != 0.0:  # L2 正则：在损失上加 (λ/2)||w||² 等价于在梯度上加 λ·w；只有 λ≠0 才执行，避免无意义运算
                    grad = grad.add(p, alpha=weight_decay)  # grad ← grad + λ·p；.add() 返回新张量不影响 p.grad，shape 不变

                exp_avg.mul_(momentum).add_(grad)  # 原地动量更新：m ← β·m + g
                                                   # mul_(β)：m 乘以衰减系数，保留历史方向的 β 比例
                                                   # add_(g)：叠加当前梯度，使动量向新梯度方向偏转

                if nesterov:  # NAG：在"动量预测的未来位置"重新估计梯度，等价于 u = β·m + g
                    update = exp_avg * momentum + grad  # u ← β·m + g；相比普通动量 u=m，多了一次"超前修正"，shape 与 p 相同
                else:         # 普通动量：直接用当前动量缓冲区作为更新方向
                    update = exp_avg               # u ← m，shape 与 p 相同；注意这是引用，后续 clip 操作需重新赋值

                if getattr(p, "is_qk_projection", False):  # 仅对 Q/K 投影权重执行 QK-Clip；is_qk_projection 是在模型初始化时手动挂载的自定义属性
                    update_norm = torch.norm(update)        # 计算更新量的全局 L2 范数 ‖u‖，返回标量 Tensor，shape=[]；反映本次更新的"步长大小"
                    if update_norm > clip_threshold:        # 只在超过阈值时才裁剪，避免对小更新量做不必要的缩放
                        scale  = clip_threshold / (update_norm + eps)  # 缩放因子 τ/(‖u‖+ε) ∈ (0,1)；加 ε 防止 update_norm 精确为 0 时除零
                        update = update * scale             # u ← u·scale，方向不变，L2 范数从 ‖u‖ 压缩到恰好等于 τ，shape 不变

                p.add_(update, alpha=-lr)  # 原地梯度下降：w ← w - α·u；alpha=-lr 将 update 以负学习率叠加到参数上

        return loss  # 返回 closure 产生的 loss（Tensor 标量）；无 closure 时返回 None，调用方可忽略


# ============================================================
# 三、极简注意力回归模型
# ============================================================
class TinyAttentionRegressor(nn.Module):
    """
    最小可运行的单头缩放点积注意力回归模型，专为演示 MuonClip 设计。

    结构：
        输入 x (B, L, D)
          → Q/K/V 线性投影（各自独立的 D×D 权重矩阵，无偏置）
          → 注意力分数 = Q @ K^T / sqrt(D)，shape (B, L, L)
          → softmax → 注意力权重 (B, L, L)
          → context = attn_weights @ V，shape (B, L, D)
          → 序列维平均池化 → (B, D)
          → 线性回归头（D → 1）
          → squeeze → 预测值 (B,)

    Q、K 投影的权重矩阵被挂载 is_qk_projection=True 属性，
    告知 MuonClip 优化器对这两个参数单独执行 QK-Clip 范数裁剪。

    参数
    ----------
    dim : int，默认 16
        输入特征维度 D，同时也是 Q/K/V 投影的输出维度；须与数据的特征维度一致

    输入
    ----
    x : Tensor，shape = (batch_size, seq_len, dim)，dtype = float32

    输出
    ----
    pred : Tensor，shape = (batch_size,)，dtype = float32，每个样本的回归预测标量
    """

    # --------------------------------------------------------
    # 3.1 初始化
    # --------------------------------------------------------
    def __init__(self, dim=16):  # dim：特征维度 D，类型 int，默认 16；决定 Q/K/V 投影矩阵大小为 D×D
        super().__init__()  # 调用 nn.Module 基类初始化，注册内部 _modules / _parameters / _buffers 等数据结构，使子模块可被 .parameters() 遍历到

        self.q_proj = nn.Linear(dim, dim, bias=False)  # Q 投影：仿射变换 x → xW_Q^T，权重 W_Q shape=(dim,dim)，无偏置；输入/输出均为 (...,dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)  # K 投影：仿射变换 x → xW_K^T，权重 W_K shape=(dim,dim)，无偏置；K 与 Q 的点积构成注意力分数
        self.v_proj = nn.Linear(dim, dim, bias=False)  # V 投影：仿射变换 x → xW_V^T，权重 W_V shape=(dim,dim)，无偏置；V 提供被"选取"的信息内容
        self.out    = nn.Linear(dim, 1,   bias=True)   # 回归输出头：将池化后的 dim 维特征映射为标量；输入 (...,dim)，输出 (...,1)，带偏置

        self.q_proj.weight.is_qk_projection = True  # 在 Q 权重张量上挂载自定义布尔属性；MuonClip.step() 中通过 getattr(p, "is_qk_projection", False) 读取，决定是否对该参数执行 QK-Clip
        self.k_proj.weight.is_qk_projection = True  # 同上，对 K 权重张量标记；Q 和 K 共同决定注意力分数，两者都需要被裁剪

    # --------------------------------------------------------
    # 3.2 前向传播
    # --------------------------------------------------------
    def forward(self, x):
        """
        单头缩放点积注意力前向传播。

        参数
        ----------
        x : Tensor，shape = (B, L, D)，B=batch_size，L=seq_len，D=dim
            输入序列的特征表示

        返回值
        -------
        pred : Tensor，shape = (B,)，dtype = float32
            每个样本对应的回归预测标量
        """
        q = self.q_proj(x)  # 查询矩阵：x @ W_Q^T，shape=(B,L,D)；每个位置 token 都被映射为一个"查询"向量，用于与所有 key 计算相似度
        k = self.k_proj(x)  # 键矩阵：x @ W_K^T，shape=(B,L,D)；每个位置 token 被映射为一个"键"向量，供其他位置的 query 来匹配
        v = self.v_proj(x)  # 值矩阵：x @ W_V^T，shape=(B,L,D)；每个位置 token 的"内容"，由注意力权重决定被选取多少

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(x.size(-1))
        # 缩放点积：Q @ K^T / sqrt(D)，shape=(B,L,L)
        # 除以 sqrt(D) 的原因：随机初始化时点积的方差为 D，sqrt(D) 将其归一化到方差≈1；
        # 不缩放则高维时分数绝对值大，softmax 趋近 one-hot，梯度趋近 0，训练难以进行

        attn = torch.softmax(scores, dim=-1)
        # 对 Key 维（最后一维）做 softmax，将每行分数转化为概率分布，shape=(B,L,L)
        # 每个位置的注意力权重之和为 1，表示从所有位置"借鉴"信息的比例

        context = torch.matmul(attn, v)
        # 加权聚合：attn @ V，shape=(B,L,D)
        # 每个位置的输出是所有位置 V 的加权平均，权重由注意力分数决定；实现了"动态信息选取"

        pooled = context.mean(dim=1)
        # 沿序列维（dim=1）做平均池化，shape=(B,D)
        # 将可变长度的序列级特征压缩为固定长度的样本级向量，便于后续接全连接层

        pred = self.out(pooled).squeeze(-1)
        # self.out 输出 shape=(B,1)；squeeze(-1) 去掉尾部的 1 维，最终 shape=(B,)
        # 每个样本输出一个回归标量，与标签 y shape=(B,) 对齐，方便 MSELoss 直接计算

        return pred  # 类型 Tensor，shape=(B,)，dtype=float32，每个元素为对应样本的预测回归值


# ============================================================
# 四、构造玩具数据集
# ============================================================
def build_toy_data(num_samples=128, seq_len=8, dim=16):
    """
    生成一个小规模、CPU 可快速运行的监督回归数据集。

    标签由固定"教师权重"线性生成并叠加微小高斯噪声，
    使任务有解但不完全可解，更接近真实场景。

    参数
    ----------
    num_samples : int，默认 128
        样本数量 N；控制数据集大小，128 条在 CPU 上一次前向传播不到 1ms
    seq_len : int，默认 8
        序列长度 L；每个样本包含 L 个 token，每个 token 有 dim 维特征
    dim : int，默认 16
        特征维度 D；须与 TinyAttentionRegressor 的 dim 保持一致，否则形状不匹配

    返回值
    -------
    x : Tensor，shape=(N, L, D)，dtype=float32
        从标准正态分布 N(0,1) 采样的输入特征
    y : Tensor，shape=(N,)，dtype=float32
        由教师权重线性生成并加噪的连续回归标签
    """
    torch.manual_seed(42)  # 固定全局随机种子为 42，使 x 和 y 的随机采样结果每次完全一致，保证实验可复现

    x = torch.randn(num_samples, seq_len, dim)
    # 从标准正态分布采样，shape=(N,L,D)，dtype=float32
    # randn 使各维度均值为 0、方差为 1，不需要额外归一化

    teacher_w = torch.linspace(0.1, 1.0, steps=dim).view(1, 1, dim)
    # 教师权重：在 [0.1, 1.0] 区间生成 dim 个等间距值，shape=(1,1,D)
    # view(1,1,D) 后可以广播到 (N,L,D)，高维特征的权重更大，形成有规律的目标函数

    y = (x * teacher_w).sum(dim=(1, 2)) / (seq_len * dim)
    # 逐元素加权后沿序列维和特征维求和，shape 从 (N,L,D) → (N,)
    # 除以 seq_len*dim 做归一化，防止标签量级随序列长度和维度增大而膨胀，使回归任务难度稳定

    y = y + 0.01 * torch.randn_like(y)
    # 叠加标准差为 0.01 的高斯噪声；randn_like 生成与 y 同 shape/device/dtype 的随机张量
    # 微小噪声让模型无法完全拟合（MSE 不会下降到 0），更接近真实监督学习场景

    return x, y  # x: Tensor shape=(N,L,D)；y: Tensor shape=(N,)，两者均在 CPU 上，dtype=float32


# ============================================================
# 五、训练演示
# ============================================================
def train_demo():
    """
    端到端训练演示：组装数据、模型、损失函数和 MuonClip 优化器，
    运行 120 轮训练循环并打印 loss 变化，验证优化器可正常驱动模型收敛。

    无参数，无返回值；所有结果直接打印到标准输出。
    """
    device = torch.device("cpu")  # 强制使用 CPU；torch.device 对象用于后续 .to(device) 调用；确保无 GPU 的环境也能运行

    x, y = build_toy_data(num_samples=128, seq_len=8, dim=16)  # 生成训练数据；x shape=(128,8,16)，y shape=(128,)，均在 CPU

    x = x.to(device)  # 将 x 迁移到指定设备；此处 device="cpu" 故为空操作，但写出来是规范做法，方便后续换 GPU 只改一行
    y = y.to(device)  # 同上，将 y 迁移到与 x 相同的设备，保证 MSELoss 计算时 pred 和 y 在同一设备

    model = TinyAttentionRegressor(dim=16).to(device)  # 实例化注意力回归模型并迁移到 CPU；dim=16 须与数据特征维度 D 一致

    criterion = nn.MSELoss()  # 均方误差损失：MSE = (1/N)·Σ(pred_i - y_i)²；适合连续值回归，输出为标量 Tensor shape=[]

    # --------------------------------------------------------
    # 5.1 初始化优化器
    # --------------------------------------------------------
    optimizer = MuonClip(
        model.parameters(),   # 传入模型全部可训练参数（generator of nn.Parameter）；MuonClip 内部将其整理为 param_groups
        lr=3e-3,              # 学习率 0.003；比默认 Adam 的 1e-3 稍大，使 120 轮内 loss 下降幅度肉眼可见
        momentum=0.9,         # 动量系数 0.9；每步保留 90% 的历史动量，让优化轨迹更平滑并能穿越平坦鞍点区域
        nesterov=True,        # 启用 NAG：在"动量预判的未来位置"重算梯度，修正方向比普通动量更准；
                              # 轮数仅 120 轮，每步效率至关重要；同时与 Muon 原论文默认配置保持一致
        clip_threshold=1.0,   # QK-Clip 阈值设为 1.0（偏小），Q/K 的更新范数一旦超过 1.0 立即裁剪；
                              # 设小便于 demo 中频繁触发裁剪，直观展示 QK-Clip 的实际作用
    )

    # --------------------------------------------------------
    # 5.2 训练循环
    # --------------------------------------------------------
    epochs      = 120   # 总迭代轮数，int；120 轮在 CPU 上约 < 1 秒，足以观察到明显收敛趋势
    print_every = 20    # 每隔 20 轮打印一次日志，int；加上第 1 轮共打印 7 次，覆盖训练早期和后期
    first_loss  = None  # 记录第 1 轮 loss 的 Python float；初始为 None，作为收敛对比基准
    last_loss   = None  # 记录最新一轮 loss 的 Python float；每轮覆盖，循环结束后保存最终值

    for epoch in range(1, epochs + 1):  # epoch 从 1 到 120（含），类型 int

        model.train()           # 切换训练模式；对 Dropout/BN 层有效（本模型无这些层，但保持规范写法）

        optimizer.zero_grad()   # 将所有参数的 .grad 清零或置 None；必须在每次 backward 前调用，否则梯度会跨轮次累加

        pred = model(x)         # 前向传播：x shape=(128,8,16) → pred shape=(128,)，dtype=float32

        loss = criterion(pred, y)  # 计算 MSE 损失；pred shape=(128,)，y shape=(128,) → loss shape=[]（标量 Tensor）

        loss.backward()         # 反向传播：沿计算图从 loss 向前求导，将梯度累积写入每个叶子参数的 .grad 属性

        optimizer.step()        # 调用 MuonClip.step()：读取各参数的 .grad，经动量+NAG+QK-Clip 后原地更新参数值

        if first_loss is None:      # 仅第 1 轮执行，记录训练起点的 loss 值
            first_loss = loss.item()  # .item() 将标量 Tensor 转为 Python float，脱离计算图，避免持有整个 autograd 图的引用

        last_loss = loss.item()     # 每轮都更新，循环结束后 last_loss 持有第 120 轮的 loss 值

        if epoch % print_every == 0 or epoch == 1:  # 第 1 轮（观察初始状态）和每隔 20 轮打印
            print(f"Epoch {epoch:03d} | Loss = {loss.item():.6f}")  # {:03d} 3 位补零对齐轮次，:.6f 保留 6 位小数便于观察细微变化

    # --------------------------------------------------------
    # 5.3 打印训练结果汇总
    # --------------------------------------------------------
    print(f"First Loss: {first_loss:.6f}")            # 第 1 轮损失，作为收敛基准
    print(f"Last  Loss: {last_loss:.6f}")             # 第 120 轮损失，反映优化效果
    print(f"Loss Improved: {last_loss < first_loss}") # True 表示 loss 成功下降，验证 MuonClip 能正常驱动模型收敛


# ============================================================
# 六、脚本入口
# ============================================================
if __name__ == "__main__":  # 当文件作为主程序直接运行时条件为 True；被其他模块 import 时为 False，不会自动执行训练
    train_demo()  # 启动端到端训练演示
