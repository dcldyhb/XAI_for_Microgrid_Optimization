import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
DEFAULT_GAMMA_STRENGTH = 0.25


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def sanitize_gamma(gamma):
    gamma = torch.nan_to_num(gamma.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if gamma.dim() == 1:
        gamma = gamma.unsqueeze(0)
    scale = gamma.abs().mean(dim=1, keepdim=True).clamp_min(1.0)
    gamma = gamma / scale
    return torch.clamp(gamma, min=-5.0, max=5.0)


def build_gamma_proj(in_features, out_features):
    gamma_proj = nn.Linear(int(in_features), int(out_features))
    torch.nn.init.constant_(gamma_proj.bias, 0.0)
    torch.nn.init.xavier_uniform_(gamma_proj.weight, gain=0.05)
    return gamma_proj


def apply_gamma_modulation(x, gamma, gamma_proj, strength):
    gamma = sanitize_gamma(gamma).to(x.device)
    gamma_delta = torch.tanh(gamma_proj(gamma))
    gamma_scale = 1.0 + float(strength) * gamma_delta
    return x * gamma_scale


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, gamma_dim=0):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.hidden_dim = hidden_dim
        self.gamma_dim = int(max(0, gamma_dim or 0))
        self.gamma_strength = DEFAULT_GAMMA_STRENGTH
        self.q1_gamma_proj = None
        self.q2_gamma_proj = None

        self.apply(weights_init_)
        if self.gamma_dim > 0:
            self.q1_gamma_proj = build_gamma_proj(self.gamma_dim, self.hidden_dim)
            self.q2_gamma_proj = build_gamma_proj(self.gamma_dim, self.hidden_dim)

    def ensure_gamma_proj(self, gamma_dim, device=None):
        gamma_dim = int(max(0, gamma_dim or 0))
        if gamma_dim <= 0:
            return False

        same_dim = (
            self.q1_gamma_proj is not None
            and self.q2_gamma_proj is not None
            and self.q1_gamma_proj.in_features == gamma_dim
            and self.q2_gamma_proj.in_features == gamma_dim
        )
        if same_dim:
            if device is not None:
                self.q1_gamma_proj = self.q1_gamma_proj.to(device)
                self.q2_gamma_proj = self.q2_gamma_proj.to(device)
            self.gamma_dim = gamma_dim
            return False

        self.q1_gamma_proj = build_gamma_proj(gamma_dim, self.hidden_dim)
        self.q2_gamma_proj = build_gamma_proj(gamma_dim, self.hidden_dim)
        if device is not None:
            self.q1_gamma_proj = self.q1_gamma_proj.to(device)
            self.q2_gamma_proj = self.q2_gamma_proj.to(device)
        self.gamma_dim = gamma_dim
        return True

    def forward(self, state, action, gamma=None):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x2 = F.relu(self.linear4(xu))

        if gamma is not None:
            gamma = gamma.to(xu.device).float()
            if gamma.dim() == 1:
                gamma = gamma.unsqueeze(0)
            if self.q1_gamma_proj is None or self.q1_gamma_proj.in_features != gamma.size(1):
                self.ensure_gamma_proj(gamma.size(1), device=xu.device)
            x1 = apply_gamma_modulation(x1, gamma, self.q1_gamma_proj, self.gamma_strength)
            x2 = apply_gamma_modulation(x2, gamma, self.q2_gamma_proj, self.gamma_strength)

        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    """
    改进版 GaussianPolicy：
    引入结构调制向量 gamma（来自结构调制网络 PySR 的输出），
    作用：动态调制 actor 的隐藏层特征，实现安全约束对动作生成的影响。
    """

    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, gamma_dim=0):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.hidden_dim = hidden_dim
        self.gamma_dim = int(max(0, gamma_dim or 0))
        self.gamma_strength = DEFAULT_GAMMA_STRENGTH
        self.gamma_proj = None

        self.apply(weights_init_)
        if self.gamma_dim > 0:
            self.gamma_proj = build_gamma_proj(self.gamma_dim, self.hidden_dim)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)

    def ensure_gamma_proj(self, gamma_dim, device=None):
        gamma_dim = int(max(0, gamma_dim or 0))
        if gamma_dim <= 0:
            return False
        if self.gamma_proj is not None and getattr(self.gamma_proj, "in_features", None) == gamma_dim:
            if device is not None:
                self.gamma_proj = self.gamma_proj.to(device)
            self.gamma_dim = gamma_dim
            return False
        self.gamma_proj = build_gamma_proj(gamma_dim, self.hidden_dim)
        if device is not None:
            self.gamma_proj = self.gamma_proj.to(device)
        self.gamma_dim = gamma_dim
        return True

    def forward(self, state, gamma=None, evaluate=False):
        x = F.relu(self.linear1(state))
        if gamma is not None:
            gamma = gamma.to(x.device).float()
            if gamma.dim() == 1:
                gamma = gamma.unsqueeze(0)
            if self.gamma_proj is None or self.gamma_proj.in_features != gamma.size(1):
                self.ensure_gamma_proj(gamma.size(1), device=x.device)
            x = apply_gamma_modulation(x, gamma, self.gamma_proj, self.gamma_strength)

        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, gamma=None, evaluate=False):
        mean, log_std = self.forward(state, gamma)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.0)

    def forward(self, state, gamma=None):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state, gamma=None):
        mean = self.forward(state, gamma=gamma)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0, device=state.device), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
