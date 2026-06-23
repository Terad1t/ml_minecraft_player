"""
Factory para criar o ambiente MineRL com todos os wrappers aplicados.

Pipeline completo:
  gym.make("MineRLTreechop-v0")
    → MineRLObservationWrapper   # dict obs → array (3,64,64)
    → RewardShapingWrapper       # reward esparso → denso
    → FrameStack(n=4)            # (3,64,64) → (12,64,64)
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import FrameStack

from .wrappers import (
    MineRLObservationWrapper,
    RewardShapingWrapper,
)


def make_env(cfg=None, reward_shaping: bool = True, seed: int = 42):
    """
    Cria e retorna o ambiente MineRL com pipeline de wrappers.

    Args:
        cfg: Namespace de configuração (de load_config())
        reward_shaping: Se True, aplica RewardShapingWrapper
        seed: Seed para reprodutibilidade

    Returns:
        gym.Env pronto para uso com SB3
    """
    env_id = cfg.environment.env_id if cfg else "MineRLTreechop-v0"
    frame_stack = cfg.environment.frame_stack if cfg else 4

    # 1. Ambiente base MineRL
    env = gym.make(env_id)

    # 2. Extrai frame RGB do dict de observações
    env = MineRLObservationWrapper(env)

    # 3. Reward shaping (Fase 2+)
    if reward_shaping and (cfg is None or cfg.reward_shaping.enabled):
        env = RewardShapingWrapper(env, cfg=cfg)

    # 4. Empilha frames — (3,64,64) × 4 → (12,64,64)
    env = FrameStack(env, num_stack=frame_stack)

    # Seed para reprodutibilidade
    env.reset(seed=seed)

    return env


def make_eval_env(cfg=None, seed: int = 0):
    """
    Ambiente de avaliação — sem reward shaping para medir
    performance real com reward original.
    """
    return make_env(cfg=cfg, reward_shaping=False, seed=seed)