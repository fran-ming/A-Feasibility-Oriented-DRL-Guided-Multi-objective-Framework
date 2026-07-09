import torch
import torch.nn as nn

class TaskPolicyNet(nn.Module):
    """
    输入: 每个任务的特征 [num_tasks, feat_dim]
    输出:
      - priority scores [num_tasks]
      - vehicle logits [num_tasks, num_vehicles]
      - state value (scalar)
    """
    def __init__(self, feat_dim: int, hidden_dim: int, num_vehicles: int):
        super().__init__()
        # Backbone MLP
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor heads
        self.priority_head = nn.Linear(hidden_dim, 1)
        self.vehicle_head = nn.Linear(hidden_dim, num_vehicles)
        # Critic head (state value)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [N, feat_dim]
        返回:
          priority_scores: [N]
          vehicle_logits: [N, K]
          state_value: scalar
        """
        h = self.backbone(x)
        priority = self.priority_head(h).squeeze(-1)     # [N]
        vehicle_logits = self.vehicle_head(h)            # [N, K]
        value = self.value_head(h).mean()                # 标量
        return priority, vehicle_logits, value
