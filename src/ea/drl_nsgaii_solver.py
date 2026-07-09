import copy
import torch

class DRLNSGA2Solver(NSGA2Solver):
    """
    DRL-NSGAII:
    1) 训练 DRL policy
    2) 用 policy 采样高质量初始种群
    3) 再交给 NSGA-II 精细搜索
    """

    def __init__(self, cfg: Config, seed: int = 0, drl_seed_ratio: float = 0.5):
        super().__init__(cfg, seed)
        self.drl_seed_ratio = drl_seed_ratio
        self.drl = DRLTrainer(
            cfg,
            seed=seed,
            pretrained_seed=getattr(cfg, 'drl_pretrained_seed', None)
        )

    def _initialize(self, instance: ProblemInstance) -> List[Individual]:
        # 训练 DRL
        drl_result = self.drl.train(instance)
        pop = []
        n_drl = int(self.cfg.pop_size * self.drl_seed_ratio)
        # 多次采样
        self.drl.policy.eval()
        with torch.no_grad():
            for _ in range(n_drl):
                chrom, _ = self.drl._sample_chromosome(instance)
                chrom = mutate_solution(
                    chrom,
                    num_tasks=len(instance.tasks),
                    num_vehicles=self.cfg.num_vehicles,
                    seed=self.rng.randint(0, 10 ** 9)
                )
                pop.append(self._evaluate(chrom, instance))
        pop.append(self._evaluate(drl_result.best_chromosome, instance))

        while len(pop) < self.cfg.pop_size:
            chrom = random_solution(
                num_tasks=len(instance.tasks),
                num_vehicles=self.cfg.num_vehicles,
                seed=self.rng.randint(0, 10 ** 9) 
            )
            pop.append(self._evaluate(chrom, instance))

        return pop[:self.cfg.pop_size]