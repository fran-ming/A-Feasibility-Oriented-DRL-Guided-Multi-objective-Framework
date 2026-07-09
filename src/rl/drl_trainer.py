import copy
import random
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

@dataclass
class DRLResult:
    best_chromosome: Solution
    best_metrics: Dict[str, float]
    best_fitness: float

class DRLTrainer:
    def __init__(self, cfg: Config, seed: int = 0, pretrained_seed: Optional[int] = None):
        self.cfg = cfg
        self.seed = seed
        self.pretrained_seed = pretrained_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 网络尺寸
        hidden_dim = cfg.drl_hidden_dim
        num_vehicles = cfg.num_vehicles
        device_name = cfg.drl_device if hasattr(cfg, 'drl_device') else 'cpu'
        if device_name not in ['cpu', 'cuda']:
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device_name if device_name == 'cpu' or torch.cuda.is_available() else 'cpu')
        self.policy = TaskPolicyNet(feat_dim, hidden_dim, num_vehicles).to(self.device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.drl_lr)
      
    def save_model(self):
        """保存模型和优化器状态"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'seed': self.seed,
        }
        torch.save(checkpoint, self.model_save_path)
        print(f"[DRL] Model saved to {self.model_save_path}")

    def load_model(self):
        """加载已有的模型，如果存在则返回 True"""
        if os.path.exists(self.model_save_path):
            try:
                checkpoint = torch.load(self.model_save_path, map_location=self.device)
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
                self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"[DRL] Model loaded from {self.model_save_path}")
                return True
            except Exception as e:
                print(f"[DRL] Failed to load model: {e}")
                return False
        else:
            print(f"[DRL] No previous model found at {self.model_save_path}, training from scratch")
            return False

    def _select_action(self, x: torch.Tensor) -> tuple[int, int, torch.Tensor]:
        priority_scores, vehicle_logits, _ = self.policy(x)
        task_probs = F.softmax(priority_scores, dim=0)
        task_dist = Categorical(task_probs)
        task_idx = task_dist.sample()
        logp_task = task_dist.log_prob(task_idx)

        vehicle_logits_for_task = vehicle_logits[task_idx]
        vehicle_dist = Categorical(logits=vehicle_logits_for_task)
        vehicle_idx = vehicle_dist.sample()
        logp_vehicle = vehicle_dist.log_prob(vehicle_idx)

        return int(task_idx.cpu().item()), int(vehicle_idx.cpu().item()), logp_task + logp_vehicle

    def _discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        gamma = getattr(self.cfg, 'drl_gamma', 1.0)
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # Removed normalization to preserve reward scale for differential rewards
        return returns

    def _sample_chromosome(self, instance: ProblemInstance) -> Tuple[Solution, float]:
        """采样一个完整的 chromosome（解）
        Returns: (solution, episode_reward)
        """
        state = init_drl_state(instance)
        total_reward = 0.0

        while state['remaining']:
            x = build_task_state_features(instance, state, self.cfg)
            if x.shape[0] == 0:
                break
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
            task_pos, vehicle_idx, _ = self._select_action(x_tensor)
            task_idx = state['remaining'][task_pos]
            state, step_reward = apply_drl_action(instance, state, task_idx, vehicle_idx, self.cfg)
            total_reward += step_reward

        solution = build_solution_from_state(state)
        return solution, total_reward

    def train(self, instance: ProblemInstance) -> DRLResult:
        if self.pretrained_model_path is not None:
            if os.path.exists(self.pretrained_model_path):
                try:
                    checkpoint = torch.load(self.pretrained_model_path, map_location=self.device)
                    self.policy.load_state_dict(checkpoint['policy_state_dict'])
                    print(f"[DRL] Loaded pretrained model from {self.pretrained_model_path}")
                except Exception as e:
                    print(f"[DRL] Failed to load pretrained model: {e}")
            else:
                print(f"[DRL] No pretrained model found at {self.pretrained_model_path}, training from scratch")
        else:
            self.load_model()

        best_chrom = None
        best_metrics = None
        best_fit = float("inf")

        for ep in range(self.cfg.drl_max_episodes):
            state = init_drl_state(instance)
            # Initialize previous_J for differential rewards
            from algorithms.drl_env import compute_current_J
            state['previous_J'] = compute_current_J(state, self.cfg)
            log_probs = []
            rewards = []

            while state['remaining']:
                x = build_task_state_features(instance, state, self.cfg)
                if x.shape[0] == 0:
                    break
                x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
                task_pos, vehicle_idx, log_prob = self._select_action(x_tensor)
                task_idx = state['remaining'][task_pos]
                state, step_reward = apply_drl_action(instance, state, task_idx, vehicle_idx, self.cfg)
                log_probs.append(log_prob)
                rewards.append(step_reward)

            solution = build_solution_from_state(state)
            episode_reward, metrics = evaluate_policy_chromosome(solution, instance, self.cfg)
            fit = metrics.get('J', float('inf'))
            if fit < best_fit:
                best_fit = fit
                best_chrom = copy.deepcopy(solution)
                best_metrics = copy.deepcopy(metrics)

            returns = self._discounted_returns(rewards)
            if len(log_probs) > 0:
                log_probs_tensor = torch.stack(log_probs)
                loss = -(log_probs_tensor * returns).mean()
                if hasattr(self.cfg, 'drl_entropy_coef') and self.cfg.drl_entropy_coef > 0:
                    _, vehicle_logits, _ = self.policy(x_tensor)
                    entropy_loss = -Categorical(logits=vehicle_logits).entropy().sum()
                    loss = loss - self.cfg.drl_entropy_coef * entropy_loss

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optim.step()

            if self.cfg.verbose and ((ep + 1) % 10 == 0 or ep == 0):
                print(f"[DRL REINFORCE] Ep{ep+1}/{self.cfg.drl_max_episodes} "
                      f"EpisodeReward={episode_reward:.4f} BestFit={best_fit:.4f}")

        self.save_model()
        return DRLResult(best_chrom, best_metrics, best_fit)
