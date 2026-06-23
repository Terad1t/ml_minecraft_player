"""
Wrappers Gymnasium para o ambiente MineRL.

Pipeline de wrappers aplicados na ordem:
  MineRL env
    → MineRLObservationWrapper  (extrai frame RGB do dict de obs)
    → RewardShapingWrapper      (recompensas densas intermediárias)
    → gymnasium.FrameStack      (empilha 4 frames)

Conceito: wrappers são decoradores de ambiente — cada um modifica
um aspecto (obs, reward, action) sem alterar o env original.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MineRLObservationWrapper(gym.ObservationWrapper):
    """
    MineRL retorna observações como dicionário:
    {"pov": array(64,64,3), "inventory": {...}, ...}

    Este wrapper extrai apenas o frame RGB "pov" e
    transpõe de (H, W, C) para (C, H, W) — formato PyTorch.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Define novo observation_space: (3, 64, 64) uint8
        h, w = env.observation_space["pov"].shape[:2]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, h, w),   # C, H, W
            dtype=np.uint8,
        )

    def observation(self, obs: dict) -> np.ndarray:
        """Extrai POV e transpõe para formato PyTorch (C, H, W)."""
        frame = obs["pov"]  # (64, 64, 3)
        return np.transpose(frame, (2, 0, 1))  # (3, 64, 64)


class MineRLActionWrapper(gym.ActionWrapper):
    """
    MineRL usa espaço de ação Dict com muitas teclas.
    Para Fase 1-2, simplificamos para as ações relevantes
    para coletar madeira:
      - attack (1/0)
      - forward (1/0)
      - back (1/0)
      - left (1/0)
      - right (1/0)
      - jump (1/0)
      - camera [pitch, yaw] contínuo
      - sprint (1/0)

    O wrapper aceita um dict simplificado e preenche o resto.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Mantém o action_space original do MineRL por enquanto
        # Na Fase 3 podemos discretizar para simplificar

    def action(self, action):
        """Passa a ação diretamente (sem transformação ainda)."""
        return action


class RewardShapingWrapper(gym.Wrapper):
    """
    Adiciona recompensas densas ao reward esparso do MineRL.

    Reward original: +1 ao coletar um log de madeira.
    Com shaping: recompensas intermediárias guiam o agente
    antes de ele conseguir coletar o primeiro log.

    Baseado em potential-based shaping (Ng et al., 1999):
    r'(s,a,s') = r + γΦ(s') − Φ(s)
    onde Φ(s) é o potencial do estado.
    """

    def __init__(self, env: gym.Env, cfg=None):
        super().__init__(env)
        # Configuração de recompensas (do config.yaml)
        if cfg is not None:
            rs = cfg.reward_shaping
            self.reward_look_at_log = rs.look_at_log
            self.reward_attack_log = rs.attack_log
            self.reward_collect_log = rs.collect_log
            self.reward_approach_tree = rs.approach_tree
            self.penalty_idle = rs.penalty_idle
        else:
            # Defaults
            self.reward_look_at_log = 0.02
            self.reward_attack_log = 0.05
            self.reward_collect_log = 1.0
            self.reward_approach_tree = 0.01
            self.penalty_idle = -0.001

        self._steps_idle = 0
        self._max_idle = 100  # steps sem ação útil = penalidade

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = self._shape_reward(obs, action, reward, info)
        return obs, shaped_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._steps_idle = 0
        return self.env.reset(**kwargs)

    def _shape_reward(
        self,
        obs: np.ndarray,
        action: dict,
        reward: float,
        info: dict,
    ) -> float:
        """
        Calcula reward total com shaping.
        info do MineRL contém: 'equipped_items', 'inventory', etc.
        """
        shaped = reward * self.reward_collect_log  # escala reward original

        # Atacando = possivelmente quebrando madeira
        if isinstance(action, dict) and action.get("attack", 0):
            shaped += self.reward_attack_log
            self._steps_idle = 0
        else:
            self._steps_idle += 1

        # Penalidade por inatividade prolongada
        if self._steps_idle > self._max_idle:
            shaped += self.penalty_idle

        # Recompensa por madeira coletada (reward original > 0)
        if reward > 0:
            shaped += self.reward_collect_log

        return shaped


class NormalizeObservation(gym.ObservationWrapper):
    """
    Normaliza pixels de [0, 255] para [0, 1].
    Aplicado APÓS o FrameStack, antes de entrar na rede.

    Nota: o SB3 CnnPolicy divide por 255 internamente,
    então este wrapper é opcional — use apenas se criar
    CNN customizada que não normaliza.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=old_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0