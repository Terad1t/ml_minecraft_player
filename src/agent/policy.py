"""
Factory do agente PPO com suporte a DirectML.

CnnPolicy do SB3:
  Input (12, 64, 64) → NatureDQN CNN → features (512,) → Actor + Critic

O Actor gera π(a|s): distribuição sobre ações.
O Critic estima V(s): valor do estado atual.
Ambos compartilham o extrator CNN (shared backbone).
"""

from __future__ import annotations

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)




from src.utils.device import get_device


def make_ppo_agent(env, cfg, eval_env=None) -> PPO:
    """
    Cria e retorna um agente PPO configurado.

    Args:
        env: Ambiente de treino (gym.Env ou VecEnv)
        cfg: Namespace de configuração
        eval_env: Ambiente de avaliação (opcional)

    Returns:
        PPO agent pronto para .learn()
    """
    device = get_device(cfg.device.backend)
    ppo_cfg = cfg.ppo

    # SB3 aceita torch.device ou string
    # DirectML usa device "privateuseone" — passamos o objeto
    device_arg = device if isinstance(device, str) else str(device)

    agent = PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=ppo_cfg.learning_rate,
    n_steps=ppo_cfg.n_steps,
    batch_size=ppo_cfg.batch_size,
    n_epochs=ppo_cfg.n_epochs,
    gamma=ppo_cfg.gamma,
    gae_lambda=ppo_cfg.gae_lambda,
    clip_range=ppo_cfg.clip_range,
    clip_range_vf=ppo_cfg.clip_range_vf,
    ent_coef=ppo_cfg.ent_coef,
    vf_coef=ppo_cfg.vf_coef,
    max_grad_norm=ppo_cfg.max_grad_norm,
    normalize_advantage=ppo_cfg.normalize_advantage,
    tensorboard_log=cfg.paths.tensorboard,
    policy_kwargs={"normalize_images": False},
    device=device_arg,
    verbose=1,
)

    return agent


def make_callbacks(cfg, eval_env=None) -> CallbackList:
    """
    Cria callbacks para checkpointing e avaliação.

    CheckpointCallback: salva modelo a cada N steps.
    EvalCallback: avalia o agente e salva o melhor modelo.
    """
    callbacks = []

    # Salva checkpoints periodicamente
    checkpoint_cb = CheckpointCallback(
        save_freq=cfg.training.save_freq,
        save_path=cfg.paths.checkpoints,
        name_prefix="ppo_minecraft",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )
    callbacks.append(checkpoint_cb)

    # Avaliação periódica com ambiente separado
    if eval_env is not None:
        eval_cb = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=cfg.paths.models,
            log_path=cfg.paths.logs,
            eval_freq=cfg.training.eval_freq,
            n_eval_episodes=cfg.training.eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )
        callbacks.append(eval_cb)

    return CallbackList(callbacks)


def load_agent(path: str, env, cfg) -> PPO:
    """
    Carrega agente salvo de checkpoint.

    Args:
        path: Caminho para o arquivo .zip do modelo
        env: Ambiente para vincular ao agente carregado
        cfg: Configuração (para device)

    Returns:
        PPO agent carregado
    """
    device = get_device(cfg.device.backend)
    device_arg = str(device)

    agent = PPO.load(path, env=env, device=device_arg)
    return agent