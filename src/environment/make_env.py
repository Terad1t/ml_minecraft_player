import minerl
import gym
from gym.wrappers import FrameStack

from .wrappers import (
    MineRLObservationWrapper,
    MineRLActionDiscretizer,
    RewardShapingWrapper,
    FlattenFrameStack,
)


def make_env(cfg=None, reward_shaping: bool = True, seed: int = 42):
    env_id = cfg.environment.env_id if cfg else "MineRLTreechop-v0"
    frame_stack = cfg.environment.frame_stack if cfg else 4

    env = gym.make(env_id)
    env = MineRLObservationWrapper(env)
    env = MineRLActionDiscretizer(env)  # ← adicionar aqui

    if reward_shaping and (cfg is None or cfg.reward_shaping.enabled):
        env = RewardShapingWrapper(env, cfg=cfg)

    env = FrameStack(env, num_stack=frame_stack)
    env = FlattenFrameStack(env)

    return env


def make_eval_env(cfg=None, seed: int = 0):
    return make_env(cfg=cfg, reward_shaping=False, seed=seed)
