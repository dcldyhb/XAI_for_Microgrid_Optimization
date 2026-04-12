import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from SAC_gama.model_gamma import DeterministicPolicy, GaussianPolicy, QNetwork
from SAC_gama.utils import hard_update, soft_update


class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_lr = args.lr
        self.critic_lr = args.lr

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = getattr(args, "device", torch.device("cpu"))

        self.action_dim = int(action_space.shape[0])
        self.action_scale = torch.as_tensor(
            (action_space.high - action_space.low) / 2.0,
            dtype=torch.float32,
            device=self.device,
        )
        self.action_bias = torch.as_tensor(
            (action_space.high + action_space.low) / 2.0,
            dtype=torch.float32,
            device=self.device,
        )

        self.gamma_actor_init_strength = float(getattr(args, "gamma_actor_init_strength", 0.0))
        self.gamma_actor_final_strength = float(getattr(args, "gamma_actor_final_strength", 0.25))
        self.gamma_actor_warmup_updates = int(max(1, getattr(args, "gamma_actor_warmup_updates", 1)))

        self.gamma_critic_init_strength = float(getattr(args, "gamma_critic_init_strength", 0.0))
        self.gamma_critic_final_strength = float(getattr(args, "gamma_critic_final_strength", 0.15))
        self.gamma_critic_delay_updates = int(max(0, getattr(args, "gamma_critic_delay_updates", 0)))
        self.gamma_critic_warmup_updates = int(max(1, getattr(args, "gamma_critic_warmup_updates", 1)))

        self.gamma_prior_weight_init = float(getattr(args, "gamma_prior_weight_init", 0.0))
        self.gamma_prior_weight_final = float(getattr(args, "gamma_prior_weight_final", 0.0))
        self.gamma_prior_decay_updates = int(max(1, getattr(args, "gamma_prior_decay_updates", 1)))
        self.gamma_prior_action_scale = float(getattr(args, "gamma_prior_action_scale", 1.0))

        self.last_policy_prior_loss = 0.0
        self.last_policy_prior_confidence = 0.0
        self.last_actor_gamma_strength = 0.0
        self.last_critic_gamma_strength = 0.0

        gamma_input_dim = int(max(0, getattr(args, "gamma_input_dim", 0) or 0))

        self.critic = QNetwork(
            num_inputs,
            self.action_dim,
            args.hidden_size,
            gamma_dim=gamma_input_dim,
        ).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

        self.critic_target = QNetwork(
            num_inputs,
            self.action_dim,
            args.hidden_size,
            gamma_dim=gamma_input_dim,
        ).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(
                num_inputs,
                self.action_dim,
                args.hidden_size,
                action_space,
                gamma_dim=gamma_input_dim,
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.policy_lr)
        else:
            self.alpha = 0.0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs,
                self.action_dim,
                args.hidden_size,
                action_space,
            ).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.policy_lr)

        self._apply_gamma_schedule(0)

    def _refresh_policy_optimizer(self):
        self.policy_optim = Adam(self.policy.parameters(), lr=self.policy_lr)

    def _refresh_critic_optimizer(self):
        self.critic_optim = Adam(self.critic.parameters(), lr=self.critic_lr)

    def _linear_schedule(self, step, start_step, duration, start_value, end_value):
        if duration <= 0:
            return float(end_value)
        if step <= start_step:
            return float(start_value)
        if step >= start_step + duration:
            return float(end_value)
        ratio = float(step - start_step) / float(duration)
        return float(start_value + ratio * (end_value - start_value))

    def _apply_gamma_schedule(self, updates):
        actor_strength = self._linear_schedule(
            updates,
            start_step=0,
            duration=self.gamma_actor_warmup_updates,
            start_value=self.gamma_actor_init_strength,
            end_value=self.gamma_actor_final_strength,
        )
        critic_strength = self._linear_schedule(
            updates,
            start_step=self.gamma_critic_delay_updates,
            duration=self.gamma_critic_warmup_updates,
            start_value=self.gamma_critic_init_strength,
            end_value=self.gamma_critic_final_strength,
        )

        self.last_actor_gamma_strength = actor_strength
        self.last_critic_gamma_strength = critic_strength

        if hasattr(self.policy, "set_gamma_strength"):
            self.policy.set_gamma_strength(actor_strength)
        if hasattr(self.critic, "set_gamma_strength"):
            self.critic.set_gamma_strength(critic_strength)
        if hasattr(self.critic_target, "set_gamma_strength"):
            self.critic_target.set_gamma_strength(critic_strength)

    def _policy_prior_weight(self, updates):
        return self._linear_schedule(
            updates,
            start_step=0,
            duration=self.gamma_prior_decay_updates,
            start_value=self.gamma_prior_weight_init,
            end_value=self.gamma_prior_weight_final,
        )

    def initialize_gamma_module(self, gamma_dim):
        gamma_dim = int(max(0, gamma_dim or 0))
        if gamma_dim <= 0:
            return

        critic_rebuilt = self.critic.ensure_gamma_proj(gamma_dim, device=self.device)
        target_rebuilt = self.critic_target.ensure_gamma_proj(gamma_dim, device=self.device)
        if critic_rebuilt or target_rebuilt:
            hard_update(self.critic_target, self.critic)
            self._refresh_critic_optimizer()

        if self.policy_type == "Gaussian" and self.policy.ensure_gamma_proj(gamma_dim, device=self.device):
            self._refresh_policy_optimizer()

    def _prepare_gamma_batch(self, gamma_batch):
        if gamma_batch is None:
            return None

        if isinstance(gamma_batch, torch.Tensor):
            gamma_tensor = gamma_batch.detach().to(self.device, dtype=torch.float32)
            if gamma_tensor.dim() == 1:
                gamma_tensor = gamma_tensor.unsqueeze(0)
        else:
            gamma_array = np.asarray(gamma_batch, dtype=np.float32)
            if gamma_array.ndim == 1:
                gamma_array = np.expand_dims(gamma_array, axis=0)
            gamma_tensor = torch.as_tensor(gamma_array, dtype=torch.float32, device=self.device)

        if gamma_tensor.dim() != 2 or gamma_tensor.size(1) == 0:
            return None

        self.initialize_gamma_module(gamma_tensor.size(1))
        return gamma_tensor

    def _alpha_log_value(self):
        if torch.is_tensor(self.alpha):
            return self.alpha.detach().clone()
        return torch.tensor(float(self.alpha), dtype=torch.float32, device=self.device)

    def select_action(self, state, gamma=None, evaluate=False):
        state = np.asarray(state, dtype=np.float32)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        gamma = self._prepare_gamma_batch(gamma)

        action, log_prob, mean = self.policy.sample(state, gamma=gamma)
        chosen_action = mean if evaluate else action
        return chosen_action.detach().cpu().numpy()[0], log_prob, mean

    def update_parameters(self, memory, batch_size, updates):
        self._apply_gamma_schedule(updates)

        batch = memory.sample(batch_size=batch_size)
        if len(batch) == 5:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = batch
            gamma_batch = None
            next_gamma_batch = None
        else:
            (
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                mask_batch,
                gamma_batch,
                next_gamma_batch,
            ) = batch

        state_batch = torch.as_tensor(state_batch, dtype=torch.float32, device=self.device)
        next_state_batch = torch.as_tensor(next_state_batch, dtype=torch.float32, device=self.device)
        action_batch = torch.as_tensor(action_batch, dtype=torch.float32, device=self.device)
        reward_batch = torch.as_tensor(reward_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        mask_batch = torch.as_tensor(mask_batch, dtype=torch.float32, device=self.device).unsqueeze(1)
        gamma_batch = self._prepare_gamma_batch(gamma_batch)
        next_gamma_batch = self._prepare_gamma_batch(next_gamma_batch)

        critic_gamma_batch = gamma_batch if updates >= self.gamma_critic_delay_updates else None
        critic_next_gamma_batch = next_gamma_batch if updates >= self.gamma_critic_delay_updates else None

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch,
                gamma=next_gamma_batch,
            )
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch,
                next_state_action,
                gamma=critic_next_gamma_batch,
            )
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch, gamma=critic_gamma_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()

        pi, log_pi, mean_pi = self.policy.sample(state_batch, gamma=gamma_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi, gamma=critic_gamma_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        self.last_policy_prior_loss = 0.0
        self.last_policy_prior_confidence = 0.0

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.detach().clone()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)
            alpha_tlogs = self._alpha_log_value()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return (
            qf1_loss.item(),
            qf2_loss.item(),
            policy_loss.item(),
            alpha_loss.item(),
            alpha_tlogs.item(),
        )

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")
        if ckpt_path is None:
            ckpt_path = f"checkpoints/sac_checkpoint_{env_name}_{suffix}"
        print(f"Saving models to {ckpt_path}")
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "critic_target_state_dict": self.critic_target.state_dict(),
                "critic_optimizer_state_dict": self.critic_optim.state_dict(),
                "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            },
            ckpt_path,
        )

    def load_checkpoint(self, ckpt_path, evaluate=False):
        print(f"Loading models from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        gamma_dims = []
        for state_dict, key in (
            (checkpoint["policy_state_dict"], "gamma_proj.weight"),
            (checkpoint["critic_state_dict"], "q1_gamma_proj.weight"),
        ):
            weight = state_dict.get(key)
            if weight is not None:
                gamma_dims.append(int(weight.shape[1]))
        if gamma_dims:
            self.initialize_gamma_module(max(gamma_dims))

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self._apply_gamma_schedule(
            max(
                self.gamma_actor_warmup_updates,
                self.gamma_critic_delay_updates + self.gamma_critic_warmup_updates,
                self.gamma_prior_decay_updates,
            )
        )

        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train()
            self.critic.train()
            self.critic_target.train()
