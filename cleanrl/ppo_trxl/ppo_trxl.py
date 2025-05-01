import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import memory_gym  # noqa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from einops import rearrange
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from pom_env import PoMEnv  # noqa
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "MortarMayhem-Grid-v0"
    """the id of the environment"""
    total_timesteps: int = 200000000
    """total timesteps of the experiments 初始学习率"""
    init_lr: float = 2.75e-4 
    """the initial learning rate of the optimizer 最终学习率"""
    final_lr: float = 1.0e-5
    """the final learning rate of the optimizer after linearly annealing"""
    num_envs: int = 32
    """the number of parallel game environments"""
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout """
    anneal_steps: int = 32 * 512 * 10000
    """the number of steps to linearly anneal the learning rate and entropy coefficient from initial to final 这里规定训练多少步到最终的学习率"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 3
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    init_ent_coef: float = 0.0001
    """initial coefficient of the entropy bonus"""
    final_ent_coef: float = 0.000001
    """final coefficient of the entropy bonus after linearly annealing"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.25
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Transformer-XL specific arguments
    trxl_num_layers: int = 3
    """the number of transformer layers"""
    trxl_num_heads: int = 4
    """the number of heads used in multi-head attention"""
    trxl_dim: int = 384
    """the dimension of the transformer"""
    trxl_memory_length: int = 119
    """the length of TrXL's sliding memory window"""
    trxl_positional_encoding: str = "absolute"
    """the positional encoding type of the transformer, choices: "", "absolute", "learned" """
    reconstruction_coef: float = 0.0
    """the coefficient of the observation reconstruction loss, if set to 0.0 the reconstruction loss is not used"""

    # To be filled on runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, render_mode="debug_rgb_array"):
    '''
    创建环境
    :param env_id: 环境id
    :param idx: 环境索引
    :param capture_video: 是否捕获视频
    :param run_name: 运行名称
    :param render_mode: 渲染模式 默认rgb模式

    return 因为是并行化环境，所以返回一个函数
    '''
    if "MiniGrid" in env_id:
        if render_mode == "debug_rgb_array":
            render_mode = "rgb_array"

    def thunk():
        if "MiniGrid" in env_id:
            # 简化的2d环境，可以通过该环境快速验证模型
            '''
            在 MiniGrid 环境中，这两个参数的作用是：

            1. `agent_view_size=3`:
            - 定义了代理（agent）的可视范围大小
            - 值为3表示代理可以看到以自己为中心的 3x3 网格区域
            - 这是一个重要的部分可观察性（Partial Observability）参数
            - 更大的值会让代理看到更多区域，更小的值会限制视野范围

            2. `tile_size=28`:
            - 定义了每个网格（tile）渲染时的像素大小
            - 值为28表示每个网格会被渲染成 28x28 像素的图像
            - 这个参数影响观察空间的最终分辨率
            - 在这个例子中，3x3 的观察窗口会被渲染成 84x84 像素（3×28 = 84）

            示意图：
            ```
            agent_view_size=3 的视野范围：
            [ ][ ][ ]  
            [ ][A][ ]  (A代表agent位置)
            [ ][ ][ ]

            每个[ ]会被渲染成 28x28 像素
            ```

            这种设计让 MiniGrid 环境既保持了简单的离散网格结构，又能提供像素级的观察值给深度学习模型使用。
            '''
            env = gym.make(env_id, agent_view_size=3, tile_size=28, render_mode=render_mode)
            '''
            RGBImgPartialObsWrapper:
            将代理的部分可观察视野转换为 RGB 图像格式
            输入参数 tile_size=28 定义了每个网格的像素大小
            将原始的符号化观察转换为 RGB 颜色表示
            输出形状为 (agent_view_size * tile_size, agent_view_size * tile_size, 3)
            在这个例子中是 (84, 84, 3) (3×28 = 84)
            ImgObsWrapper:
            简化了观察空间的结构
            移除了原始观察字典中的其他信息，只保留图像数据
            将观察空间从字典格式 {"image": box_space} 简化为直接的 box_space
            使得环境的观察值直接就是图像数据，更容易与深度学习模型集成
            '''
            env = ImgObsWrapper(RGBImgPartialObsWrapper(env, tile_size=28))
            # 限制环境的最大步数为 96 步
            env = gym.wrappers.TimeLimit(env, 96)
        else:
            env = gym.make(env_id, render_mode=render_mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # 用于记录每个回合（episode）的统计信息，并在info字典中返回
        '''
        主要功能
        回合数据记录

        记录每个回合的累计奖励（return）
        记录每个回合的长度（steps）
        记录每个回合的时间信息
        数据访问

        在回合结束时（done=True），通过 info 字典返回统计数据
        返回格式为：
        info = {
            "episode": {
                "r": episode_return,  # 回合总奖励
                "l": episode_length,  # 回合长度
                "t": episode_time    # 回合用时
            }
        }
        '''
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    # torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, min_timescale=2.0, max_timescale=1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer("inv_freqs", inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.0)
        sinusoidal_inp = rearrange(seq, "n -> n ()") * rearrange(self.inv_freqs, "d -> () d")
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim=-1)
        return pos_emb


class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 每个头的维度
        self.head_size = embed_dim // num_heads

        # 这里必须要整除
        assert self.head_size * num_heads == embed_dim, "Embedding dimension needs to be divisible by the number of heads"

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

    def forward(self, values, keys, query, mask):
        '''
        
        values: 形状为 (N, value_len, embed_dim)
        keys: 形状为 (N, key_len, embed_dim)
        query: 形状为 (N, query_len, embed_dim)

        mask: 形状为 (N, query_len, key_len) 

        return shape

        '''

        N = query.shape[0]
        # todo 这些len是输入数据的长度吗？
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # todo 输入shape （？）to （batch_size, seq_len, num_heads, head_size)
        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        # 提取 q k v的特征
        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Dot-product
        '''
        [n, q, h, d]:
        n: batch size
        q: query长度
        h: 注意力头数
        d: 每个头的维度

        [n, k, h, d]:
        n: batch size
        k: key长度
        h: 注意力头数
        d: 每个头的维度

        输出张量维度[n, h, q, k]:
        n: batch size
        h: 注意力头数
        q: query长度
        k: key长度

        这个操作实际上在计算注意力机制中的查询(query)和键(key)的点积，具体来说：
        对每个batch(n)
        对每个注意力头(h)
        计算query和key在维度d上的点积
        得到query和key之间的注意力分数矩阵

        output = torch.zeros(n, h, q, k)
        for n_idx in range(n):
            for h_idx in range(h):
                for q_idx in range(q):
                    for k_idx in range(k):
                        for d_idx in range(d):
                            output[n_idx, h_idx, q_idx, k_idx] += \
                                queries[n_idx, q_idx, h_idx, d_idx] * keys[n_idx, k_idx, h_idx, d_idx]

        实际上就是简化操作，告诉方法中输入的两个张量每个维度的标识，输出张量的维度标识是什么样子的，从而自动生成对应的计算流程
        如果不使用这个，那么就要手动先
        交换q\k和h的维度，然后在计算q和k的乘积时，需要交换k的k d维度，使得最后计算输出能够得到nhqk
        '''
        # energy 张量形状为 [n, h, q, k]
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            # mask.unsqueeze(1).unsqueeze(1) == 0  # 增加维度使其匹配 energy 的形状
            # 以下操作会将energy中对应mask对应位置为0 设置为-1e20
            # 在后续的 softmax 操作中，这些位置会变成接近 0 的值
            # 实际上就是让模型在这些位置的注意力权重趋近于 0
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20"))  # -inf causes NaN

        # Normalize energy values and apply softmax to retrieve the attention scores
        # 先正则化，然后对最后一个维度进行softmax计算
        attention = torch.softmax(
            energy / (self.embed_dim ** (1 / 2)), dim=3
        )  # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        # 依旧用相同的方法计算
        # attention shape: (N, heads, query_len, key_len)
        # values shape:  (N, value_len/seq_len, heads, head_dim)
        # 首先Value 维度变为：(N, heads，value_len/seq_len, head_dim)
        # 然后attention和value相乘后shape：(N, heads, query_len, head_dim)  因为query_len = value_lan = key_len 
        # 交换维度(N, query_len, heads，head_dim) 
        # torch.einsum("nhql,nlhd->nqhd", [attention, values]) shape （N, query_len/seq_len, heads, head_dim）
        # reshape shape (N, query_len/seq_len, heads * head_dim = embed_dim)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.num_heads * self.head_size)

        # 进一步对特征进行提取
        # out shape (N, query_len/seq_len, embed_dim)
        # attention shape (N, heads, query_len, key_len)
        return self.fc_out(out), attention


# 这里仅仅只是trandformer的Encoder部分
class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        '''
        dim: transformer的维度
        num_heads: transformer的注意力头数
        '''
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads)
        self.layer_norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.layer_norm_attn = nn.LayerNorm(dim)
        self.fc_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        '''
        return out和注意力权重
        out: 形状为 (N, query_len/seq_len, dim)
        attention_weights: 形状为 (N, heads, query_len, key_len)
        '''
        # Pre-layer normalization (post-layer normalization is usually less effective)
        # 先进行归一哈
        query_ = self.layer_norm_q(query)
        value = self.norm_kv(value)
        key = value  # K = V -> self-attention
        # 然后进行注意力计算
        # k等于v，使用q来计算注意力
        attention, attention_weights = self.attention(value, key, query_, mask)  # MHA
        # 这里实现的resnet连接
        x = attention + query  # Skip connection
        # 对结果进行归一化
        x_ = self.layer_norm_attn(x)  # Pre-layer normalization
        # 对输出的结果进行特征提取
        forward = self.fc_projection(x_)  # Forward projection
        # 又是resnet连接
        out = forward + x  # Skip connection
        return out, attention_weights


class Transformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, max_episode_steps, positional_encoding):
        '''
        num_layers: transformer层的数量
        dim: transformer的维度
        num_heads: transformer的注意力头数
        max_episode_steps: 最大步数 todo 用于什么
        positional_encoding: 位置编码的类型
        '''
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.positional_encoding = positional_encoding
        if positional_encoding == "absolute":
            # 这里应该是绝对位置编码
            self.pos_embedding = PositionalEncoding(dim)
        elif positional_encoding == "learned":
            # 这里应该是可以训练学习的位置编码器 todo
            # torch.randn(max_episode_steps, dim) shape ： (max_episode_steps, dim)
            self.pos_embedding = nn.Parameter(torch.randn(max_episode_steps, dim))
        
        # 创建transformer层
        self.transformer_layers = nn.ModuleList([TransformerLayer(dim, num_heads) for _ in range(num_layers)])

    def forward(self, x, memories, mask, memory_indices):
        # Add positional encoding to every transformer layer input
        if self.positional_encoding == "absolute":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2)
        elif self.positional_encoding == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2)

        # Forward transformer layers and return new memories (i.e. hidden states)
        out_memories = []
        for i, layer in enumerate(self.transformer_layers):
            out_memories.append(x.detach())
            x, attention_weights = layer(
                memories[:, :, i], memories[:, :, i], x.unsqueeze(1), mask
            )  # args: value, key, query, mask
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        return x, torch.stack(out_memories, dim=1)


class Agent(nn.Module):
    def __init__(self, args, observation_space, action_space_shape, max_episode_steps):
        '''
        
        args: 命令行参数
        observation_space: 观察空间
        action_space_shape: 动作空间的形状
        max_episode_steps: 最大步数
        '''
        super().__init__()
        self.obs_shape = observation_space.shape
        self.max_episode_steps = max_episode_steps

        if len(self.obs_shape) > 1:
            # 这里是处理像素观察
            # 输出的shape 是（batch_size, trxl_dim）
            self.encoder = nn.Sequential(
                layer_init(nn.Conv2d(3, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, args.trxl_dim)),
                nn.ReLU(),
            )
        else:
            # 这里是处理非像素观察
            # 输出的shape 是（batch_size, trxl_dim）
            self.encoder = layer_init(nn.Linear(observation_space.shape[0], args.trxl_dim))

        # 创建transformer层
        self.transformer = Transformer(
            args.trxl_num_layers, args.trxl_dim, args.trxl_num_heads, self.max_episode_steps, args.trxl_positional_encoding
        )

        # todo 这里的作用是什么？
        self.hidden_post_trxl = nn.Sequential(
            layer_init(nn.Linear(args.trxl_dim, args.trxl_dim)),
            nn.ReLU(),
        )

        # 这里对于每个动作空间的维度创建一个actor # todo 了解这里的
        self.actor_branches = nn.ModuleList(
            [
                layer_init(nn.Linear(args.trxl_dim, out_features=num_actions), np.sqrt(0.01))
                for num_actions in action_space_shape
            ]
        )

        # 创建观察特征评价网络
        self.critic = layer_init(nn.Linear(args.trxl_dim, 1), 1)

        # todo 这里的网络的作用是什么？
        if args.reconstruction_coef > 0.0:
            self.transposed_cnn = nn.Sequential(
                layer_init(nn.Linear(args.trxl_dim, 64 * 7 * 7)),
                nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                layer_init(nn.ConvTranspose2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(64, 32, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.ConvTranspose2d(32, 3, 8, stride=4)),
                nn.Sigmoid(),
            )

    def get_value(self, x, memory, memory_mask, memory_indices):
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        x, _ = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        return self.critic(x).flatten()

    def get_action_and_value(self, x, memory, memory_mask, memory_indices, action=None):
        if len(self.obs_shape) > 1:
            x = self.encoder(x.permute((0, 3, 1, 2)) / 255.0)
        else:
            x = self.encoder(x)
        x, memory = self.transformer(x, memory, memory_mask, memory_indices)
        x = self.hidden_post_trxl(x)
        self.x = x
        probs = [Categorical(logits=branch(x)) for branch in self.actor_branches]
        if action is None:
            action = torch.stack([dist.sample() for dist in probs], dim=1)
        log_probs = []
        for i, dist in enumerate(probs):
            log_probs.append(dist.log_prob(action[:, i]))
        entropies = torch.stack([dist.entropy() for dist in probs], dim=1).sum(1).reshape(-1)
        return action, torch.stack(log_probs, dim=1), entropies, self.critic(x).flatten(), memory

    def reconstruct_observation(self):
        x = self.transposed_cnn(self.x)
        return x.permute((0, 2, 3, 1))


if __name__ == "__main__":
    args = tyro.cli(Args)
    # batch_size 数量等于 num_envs（环境数量） * num_steps（todo 这个是控制什么）
    # minibatch_size todo 
    # num_minibatches todo
    # total_timesteps 总训练步数
    # num_iterations 实际的训练轮数
    # run_name: 由环境id、实验名称、随机种子和当前时间戳组成
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # 这里是类似tensorboard的可视化工具
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    # 控制器随机种子，用于复线
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Determine the device to be used for training and set the default tensor type
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_device(device)
    else:
        device = torch.device("cpu")

    # Environment setup
    # 并行化环境
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    observation_space = envs.single_observation_space
    # 如果是离散的动作，那么动作的空间是 n
    # 如果动作是连续的，那么动作的空间是向量空间
    action_space_shape = (
        (envs.single_action_space.n,)
        if isinstance(envs.single_action_space, gym.spaces.Discrete)
        else tuple(envs.single_action_space.nvec)
    )
    # 每个环境的id
    env_ids = range(args.num_envs)
    # 记录每个环境当前的步数？todo
    env_current_episode_step = torch.zeros((args.num_envs,), dtype=torch.long)
    # Determine maximum episode steps 环境的最大步数
    max_episode_steps = envs.envs[0].spec.max_episode_steps
    # 尝试用其他的方法获取最大的步数
    if not max_episode_steps:
        envs.envs[0].reset()  # Memory Gym envs need to be reset before accessing max_episode_steps
        max_episode_steps = envs.envs[0].max_episode_steps
    # 如果没有最大步数，则限制最大步数为 1024
    if max_episode_steps <= 0:
        max_episode_steps = 1024  # Memory Gym envs have max_episode_steps set to -1
    # Set transformer memory length to max episode steps if greater than max episode steps
    # 这里感觉是在设置每个环境的最大记忆步数 todo
    args.trxl_memory_length = min(args.trxl_memory_length, max_episode_steps)

    agent = Agent(args, observation_space, action_space_shape, max_episode_steps).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.init_lr)
    # 难怪用BCELoss，因为里面的actor输出本来就是单独针对每个动作空间的维度
    bce_loss = nn.BCELoss()  # Binary cross entropy loss for observation reconstruction

    # ALGO Logic: Storage setup
    # 这里是按照时间序列的顺序，存储每个环境采集的数据
    rewards = torch.zeros((args.num_steps, args.num_envs))
    actions = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)), dtype=torch.long)
    dones = torch.zeros((args.num_steps, args.num_envs))
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space.shape)
    log_probs = torch.zeros((args.num_steps, args.num_envs, len(action_space_shape)))
    values = torch.zeros((args.num_steps, args.num_envs))
    # The length of stored-memories is equal to the number of sampled episodes during training data sampling
    # (num_episodes, max_episode_length, num_layers, embed_dim)
    stored_memories = []
    # Memory mask used during attention
    # 这里存储着每个环境每步的掩码
    stored_memory_masks = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.bool)
    # Index to select the correct episode memory from stored_memories
    # todo 这里的参数作用
    stored_memory_index = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long)
    # Indices to slice the episode memories into windows
    stored_memory_indices = torch.zeros((args.num_steps, args.num_envs, args.trxl_memory_length), dtype=torch.long)

    # TRY NOT TO MODIFY: start the game
    global_step = 0 # 实际训练步数
    start_time = time.time()
    episode_infos = deque(maxlen=100)  # Store episode results for monitoring statistics
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs)
    # Setup placeholders for each environments's current episodic memory
    # 采集训练数据缓冲区，多少个环境，最大的步数，transformer的层户，transformer的嵌入维度 
    # todo 存储什么？
    next_memory = torch.zeros((args.num_envs, max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32)
    # Generate episodic memory mask used in attention
    # 这里创建了一个下三角掩码矩阵
    memory_mask = torch.tril(torch.ones((args.trxl_memory_length, args.trxl_memory_length)), diagonal=-1)
    """ e.g. memory mask tensor looks like this if memory_length = 6
    0, 0, 0, 0, 0, 0
    1, 0, 0, 0, 0, 0
    1, 1, 0, 0, 0, 0
    1, 1, 1, 0, 0, 0
    1, 1, 1, 1, 0, 0
    1, 1, 1, 1, 1, 0
    """
    # Setup memory window indices to support a sliding window over the episodic memory
    # torch.arange(0, args.trxl_memory_length).unsqueeze(0)：首先创建一个序列，例如 [0,1,2,3]
    # base = base.unsqueeze(0)  # 2. 增加维度 变成 [[0,1,2,3]]
    #  torch.repeat_interleave 重复这个序列trxl_memory_length - 1次
    # 那么最终会输出的repetitions的shape (args.trxl_memory_length - 1, args.trxl_memory_length)
    repetitions = torch.repeat_interleave(
        torch.arange(0, args.trxl_memory_length).unsqueeze(0), args.trxl_memory_length - 1, dim=0
    ).long()
    # # torch.arange(0, args.trxl_memory_length).unsqueeze(0)：首先创建一个序列，例如 [0,1,2,3] 、 [1,2,3,4]、 [2,3,4,5]、 [3,4,5,6] ......
    # 将上述序列重复max_episode_steps - args.trxl_memory_length + 1，由之前可知trxl_memory_length <= max_episode_steps
    # 最后stack起来 shape (max_episode_steps - args.trxl_memory_length + 1, args.trxl_memory_length)
    memory_indices = torch.stack(
        [torch.arange(i, i + args.trxl_memory_length) for i in range(max_episode_steps - args.trxl_memory_length + 1)]
    ).long()

    # 最后拼接起来，shape = （max_episode_steps, args.trxl_memory_length） todo 
    # 最后输出的内容类似如下 todo 作用是什么？
    memory_indices = torch.cat((repetitions, memory_indices))
    """ e.g. the memory window indices tensor looks like this if memory_length = 4 and max_episode_length = 7:
    0, 1, 2, 3
    0, 1, 2, 3
    0, 1, 2, 3
    0, 1, 2, 3
    1, 2, 3, 4
    2, 3, 4, 5
    3, 4, 5, 6
    """

    for iteration in range(1, args.num_iterations + 1):
        sampled_episode_infos = []

        # Annealing the learning rate and entropy coefficient if instructed to do so
        # 计算是否开启学习率动态调整 以及 是否已训练步数超过了最大学习率训练步数
        do_anneal = args.anneal_steps > 0 and global_step < args.anneal_steps
        # 如果超过了则学习率比率为0
        # 否则则根据1 - 当前步数/最大学习率调整步数的比率 计算 学习率比率
        frac = 1 - global_step / args.anneal_steps if do_anneal else 0
        # 计算当前的学习率
        lr = (args.init_lr - args.final_lr) * frac + args.final_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # todo ent_coef的作用
        ent_coef = (args.init_ent_coef - args.final_ent_coef) * frac + args.final_ent_coef

        # Init episodic memory buffer using each environments' current episodic memory
        # todo 这里在干嘛？为什么重新组建list
        stored_memories = [next_memory[e] for e in range(args.num_envs)]
        # 这边应该是设置环境的id
        for e in range(args.num_envs):
            stored_memory_index[:, e] = e

        # todo 这里在干嘛？
        for step in range(args.num_steps):
            # 因为有num_envs个环境，所以每执行一步都相当于走了num_envs步
            global_step += args.num_envs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                obs[step] = next_obs
                dones[step] = next_done
                # torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1 将env_current_episode_step步数的范围设置为0, args.trxl_memory_length - 1
                # 限制在一个合理的范围内
                # 然后根据env_current_episode_step每个环境执行到的步数提取出对应的掩码，使得环境不会看到未来，
                # 比如某个环境执行到了第4步，那么提取出来的memory_mask的掩码为1, 1, 1, 1, 0, 0
                # 将每步的掩码存储到stored_memory_masks
                stored_memory_masks[step] = memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)]
                # todo 这个是在干嘛
                stored_memory_indices[step] = memory_indices[env_current_episode_step]
                # Retrieve the memory window from the entire episodic memory
                memory_window = batched_index_select(next_memory, 1, stored_memory_indices[step])
                action, logprob, _, value, new_memory = agent.get_action_and_value(
                    next_obs, memory_window, stored_memory_masks[step], stored_memory_indices[step]
                )
                next_memory[env_ids, env_current_episode_step] = new_memory
                # Store the action, log_prob, and value in the buffer
                actions[step], log_probs[step], values[step] = action, logprob, value

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Reset and process episodic memory if done
            for id, done in enumerate(next_done):
                if done:
                    # Reset the environment's current timestep
                    env_current_episode_step[id] = 0
                    # Break the reference to the environment's episodic memory
                    mem_index = stored_memory_index[step, id]
                    stored_memories[mem_index] = stored_memories[mem_index].clone()
                    # Reset episodic memory
                    next_memory[id] = torch.zeros(
                        (max_episode_steps, args.trxl_num_layers, args.trxl_dim), dtype=torch.float32
                    )
                    if step < args.num_steps - 1:
                        # Store memory inside the buffer
                        stored_memories.append(next_memory[id])
                        # Store the reference of to the current episodic memory inside the buffer
                        stored_memory_index[step + 1 :, id] = len(stored_memories) - 1
                else:
                    # Increment environment timestep if not done
                    env_current_episode_step[id] += 1

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        sampled_episode_infos.append(info["episode"])

        # Bootstrap value if not done
        with torch.no_grad():
            start = torch.clip(env_current_episode_step - args.trxl_memory_length, 0)
            end = torch.clip(env_current_episode_step, args.trxl_memory_length)
            indices = torch.stack([torch.arange(start[b], end[b]) for b in range(args.num_envs)]).long()
            memory_window = batched_index_select(next_memory, 1, indices)  # Retrieve the memory window from the entire episode
            next_value = agent.get_value(
                next_obs,
                memory_window,
                memory_mask[torch.clip(env_current_episode_step, 0, args.trxl_memory_length - 1)],
                stored_memory_indices[-1],
            )
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape(-1, *obs.shape[2:])
        b_logprobs = log_probs.reshape(-1, *log_probs.shape[2:])
        b_actions = actions.reshape(-1, *actions.shape[2:])
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_memory_index = stored_memory_index.reshape(-1)
        b_memory_indices = stored_memory_indices.reshape(-1, *stored_memory_indices.shape[2:])
        b_memory_mask = stored_memory_masks.reshape(-1, *stored_memory_masks.shape[2:])
        stored_memories = torch.stack(stored_memories, dim=0)

        # Remove unnecessary padding from TrXL memory, if applicable
        actual_max_episode_steps = (stored_memory_indices * stored_memory_masks).max().item() + 1
        if actual_max_episode_steps < args.trxl_memory_length:
            b_memory_indices = b_memory_indices[:, :actual_max_episode_steps]
            b_memory_mask = b_memory_mask[:, :actual_max_episode_steps]
            stored_memories = stored_memories[:, :actual_max_episode_steps]

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                mb_memories = stored_memories[b_memory_index[mb_inds]]
                mb_memory_windows = batched_index_select(mb_memories, 1, b_memory_indices[mb_inds])

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], mb_memory_windows, b_memory_mask[mb_inds], b_memory_indices[mb_inds], b_actions[mb_inds]
                )

                # Policy loss
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_advantages = mb_advantages.unsqueeze(1).repeat(
                    1, len(action_space_shape)
                )  # Repeat is necessary for multi-discrete action spaces
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = torch.exp(logratio)
                pgloss1 = -mb_advantages * ratio
                pgloss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pgloss1, pgloss2).mean()

                # Value loss
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                if args.clip_vloss:
                    v_loss_clipped = b_values[mb_inds] + (newvalue - b_values[mb_inds]).clamp(
                        min=-args.clip_coef, max=args.clip_coef
                    )
                    v_loss = torch.max(v_loss_unclipped, (v_loss_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = v_loss_unclipped.mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Combined losses
                loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                # Add reconstruction loss if used
                r_loss = torch.tensor(0.0)
                if args.reconstruction_coef > 0.0:
                    r_loss = bce_loss(agent.reconstruct_observation(), b_obs[mb_inds] / 255.0)
                    loss += args.reconstruction_coef * r_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log and monitor training statistics
        episode_infos.extend(sampled_episode_infos)
        episode_result = {}
        if len(episode_infos) > 0:
            for key in episode_infos[0].keys():
                episode_result[key + "_mean"] = np.mean([info[key] for info in episode_infos])

        print(
            "{:9} SPS={:4} return={:.2f} length={:.1f} pi_loss={:.3f} v_loss={:.3f} entropy={:.3f} r_loss={:.3f} value={:.3f} adv={:.3f}".format(
                iteration,
                int(global_step / (time.time() - start_time)),
                episode_result["r_mean"],
                episode_result["l_mean"],
                pg_loss.item(),
                v_loss.item(),
                entropy_loss.item(),
                r_loss.item(),
                torch.mean(values),
                torch.mean(advantages),
            )
        )

        if episode_result:
            for key in episode_result:
                writer.add_scalar("episode/" + key, episode_result[key], global_step)
        writer.add_scalar("episode/value_mean", torch.mean(values), global_step)
        writer.add_scalar("episode/advantage_mean", torch.mean(advantages), global_step)
        writer.add_scalar("charts/learning_rate", lr, global_step)
        writer.add_scalar("charts/entropy_coefficient", ent_coef, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/reconstruction_loss", r_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": agent.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")

    writer.close()
    envs.close()
