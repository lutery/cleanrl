# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


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

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    '''
    1. 记录视频
    2. RecordEpisodeStatistics：记录每个episode的统计信息，包含奖励、长度等
    3. NoopResetEnv：在环境重置时，随机执行0到noop_max之间的no-op动作
    4. MaxAndSkipEnv：跳帧
    5. EpisodicLifeEnv：将游戏的一条生命视为一个episode游戏周期
    6. FireResetEnv：在环境重置时，执行FIRE动作，有些游戏需要执行Fire才会开始游戏
    7. ClipRewardEnv: 奖励裁剪，限制到 Bin reward to {+1, 0, -1} by its sign.
    8. ResizeObservation：将观察空间的图像大小调整为指定的大小 84 * 84
    9. GrayScaleObservation: 将RGB转换为灰度图
    10. FrameStack：帧堆叠，这里的1应该是不采用帧堆叠
    '''
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 1)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # 卷积+MLP层：最终输出512的维度
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        # LSTM层 输出128维度
        self.lstm = nn.LSTM(512, 128)
        # 初始化
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # 根据LSTM层的输出，进行动作的预测和奖励的预测
        # 如果是共用，那么动作的维度和奖励的维度应该尽可能的接近，否则可能会有问题
        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        '''
        x: 当前的环境观察数据
        lstm_state: lstm的状态，这个lstm的状态需要手动维护
        done: 当前的环境是否结束
        '''
        # 提取特征
        hidden = self.network(x / 255.0)

        # LSTM logic 也就是环境的数量
        batch_size = lstm_state[0].shape[1]
        # 上面特征提取以后的hidden的shape必须是[batch_size, 512]，而batch_size是环境的数量
        # -1的维度应该是环境的步数 
        # 在最开始时，传入的步数时1步
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = [] # 存储lstm每个环境步的预测输出，可以从历史步的环境状态和当前步的环境状态进行计算
        # 按照环境的步数进行循环迭代
        for h, d in zip(hidden, done):
            # 1 - 0 的意思是当前的环境没有结束，如果环境结束则值为0，那么传入的隐藏层状态也为0，表示不进行计算
            # lstm_state仅存储最后的状态，表示LSTM的最新的状态
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        # 将每个环境状态的步的预测输出进行拼接返回
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        '''
        x: 当前的环境观察数据
        lstm_state: lstm的状态，这个lstm的状态需要手动维护
        done: 当前的环境是否结束
        '''
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        '''
        x: 当前的环境观察数据
        lstm_state: lstm的状态，这个lstm的状态需要手动维护
        done: 当前的环境是否结束
        action: 当前的动作，可以是None不传入动作

        return: 
        action: 当前的动作/或者预测的动作
        probs.log_prob(action): 当前动作的概率log
        probs.entropy(): 当前动作的熵
        self.critic(hidden): 当前动作的价值
        lstm_state: lstm的状态，这个lstm的状态需要手动维护
        '''
        # 获取历史环境步的状态和当前步的环境状态提取的特征，LSTM的最新的状态
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        # 利用提取的特征进行动作的预测动作
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # 这种形式的话，应该返回的观察数据是连续的
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device) # 执行的动作
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device) # 执行动作的log概率
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device) # 环境当前所处与价值函数的值

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # 需要记录LSTM层的状态
    # todo 为什么会有两层 ，可以放在agent里面
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        # 学习率更新
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # 每次训练收集的数据长度
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # SyncVectorEnv 环境 reset返回的状体啊的维度是[env_num, 4, 84, 84]
                # next_done维度是[env_num]
                # 这边每次传入的都是一个时间步的数据
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # 执行，并搜集环境数据
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # 这块没有不记录
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            # 计算最后一个时间步的状态的价值
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
            # 优势函数存储缓冲区 初始化为0
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            # 逆序计算时间步
            # nextnonterminal： 1表示当前的环境没有结束，0表示当前的环境已经结束
            # nextvalues：当前时间步的环境的价值
            # delta：当前时间步的环境的奖励 + gamma * 下一个时间步的环境的价值 * nextnonterminal - 当前时间步的环境的价值，当前时间步的环境的价值
            # advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            # 这里的advantages是一个时间步的环境的优势函数时间上的累积
            # returns = advantages + values 表示bellman方程的计算的价值值
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

        # flatten the batch 展平
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches # 每轮训练的小循环内部将训练的数据分为几个环境batch组合
        envinds = np.arange(args.num_envs) # 环境的ID
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs) # 在代码刚开始时就设置了=args.num_envs * args.num_steps 这里应该是构建索引
        # 代表着所有采集数据的索引id，然后在训练时从中随机抽取id进行训练
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds) # 打乱环境的顺序
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end] # 获取本轮训练的所选取的对应环境的id
                # ravel()：将多维数组展平为一维数组
                # 这里是按照环境的ID进行索引，所以训练的数据是连续的（对应每一个环境）
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], # 选择对应环境的连续观察数据
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]), # 选择对应环境的lstm状态
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )

                # 计算最新模型的动作的log概率 和 旧模型的动作的log概率的比率差值
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # 这里是测量新旧策略之间的差异，防止过大的差异
                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # 是新旧策略之间的近似 KL 散度（Kullback-Leibler Divergence）
                    old_approx_kl = (-logratio).mean() 
                    # 是另一种近似 KL 散度的计算方式  KL ≈ (ratio - 1) - log(ratio)，同样用于衡量新旧策略的差异
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # 判断 ratio 是否超出了裁剪范围 [1 - clip_coef, 1 + clip_coef]。
                    # 统计超出裁剪范围的比例
                    # 如果 clipfracs 比例过高，说明裁剪限制了大部分更新，可能需要调整超参数（如 clip_coef）
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # 获取对应环境的优势函数并进行标准化
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss 计算动作优势损失
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    # 如果开启了裁剪价值损失
                    # v_loss_unclipped：未裁剪的价值损失
                    # v_clipped：裁剪后的价值损失，裁剪的方法就是将旧值加上 （新旧差值的裁剪），模拟模型预测的新值
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    # 然后将裁剪的价值损失和未裁剪的价值损失进行比较，取较大值
                    # TODO 这里还是不太明白
                    # 我自己的理解就是：类似价值的熵，训练预测的价值接近真实的价值，但是又不能过于接近
                    # 否则会导致模型过拟合，所以这里的裁剪就是为了防止模型过拟合才使用torch.max选择更大的值
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # 动作的熵损失，避免动作过拟合
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # 如果训练时新旧差异过则不进行训练
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        
        # y_pred 预测的价值
        # y_true 真实的价值
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # 真实价值的方差
        var_y = np.var(y_true)
        # var_y == 0：表示所有的值都相同，那么说明计算的价值有问题
        # explained_var：解释方差，表示模型预测的价值和真实价值之间的相关性
        # 1.0: 完美预测，值函数完全准确
        # 0.0: 预测效果等同于始终预测平均值
        # < 0: 预测效果比预测平均值还差
        '''
        当 var_y = 0 时返回 np.nan，说明所有回报值都相同，这种情况可能表示：
        环境奖励设计过于简单
        训练出现问题    
        奖励信号异常
        '''
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
