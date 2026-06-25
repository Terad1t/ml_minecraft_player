"""
Avalia um agente treinado por N episódios e imprime estatísticas.

Uso:
    python -m src.training.evaluate --model models/ppo_minecraft_final
"""

from __future__ import annotations

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    

from rich.console import Console
from rich.table import Table

from src.utils import load_config
from src.environment import make_eval_env
from src.agent import load_agent

console = Console()


def evaluate(
    model_path: str,
    n_episodes: int = 10,
    config_path: str = "configs/config.yaml",
    render: bool = False,
) -> dict:
    """
    Avalia o agente por n_episodes e retorna métricas.

    Args:
        model_path: Caminho para o .zip do modelo SB3
        n_episodes: Número de episódios de avaliação
        config_path: Caminho para config.yaml
        render: Se True, renderiza o ambiente (requer display)

    Returns:
        dict com mean_reward, std_reward, mean_length, n_logs_collected
    """
    cfg = load_config(config_path)

    console.print(f"\n[bold]Carregando modelo:[/bold] {model_path}")
    env = make_eval_env(cfg=cfg, seed=999)
    agent = load_agent(model_path, env=env, cfg=cfg)

    episode_rewards = []
    episode_lengths = []
    logs_collected = []

    console.print(f"[bold]Avaliando por {n_episodes} episódios...[/bold]\n")

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        ep_logs = 0

        while not done:
            # deterministic=True: sem exploração, usa argmax da política
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_length += 1
            ep_logs += int(reward > 0)  # reward > 0 = coletou madeira
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        logs_collected.append(ep_logs)

        console.print(
            f"  Ep {ep+1:>2}/{n_episodes} | "
            f"reward={ep_reward:>7.2f} | "
            f"length={ep_length:>5} | "
            f"logs={ep_logs}"
        )

    env.close()

    # Estatísticas finais
    metrics = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_logs": float(np.mean(logs_collected)),
        "total_logs": int(np.sum(logs_collected)),
    }

    _print_results(metrics, n_episodes)
    return metrics


def _print_results(metrics: dict, n_episodes: int) -> None:
    table = Table(title=f"Resultado da Avaliação ({n_episodes} episódios)")
    table.add_column("Métrica", style="dim")
    table.add_column("Valor", style="bold green")

    table.add_row("Reward médio", f"{metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    table.add_row("Reward mín/máx", f"{metrics['min_reward']:.2f} / {metrics['max_reward']:.2f}")
    table.add_row("Comprimento médio", f"{metrics['mean_length']:.0f} steps")
    table.add_row("Logs coletados (média)", f"{metrics['mean_logs']:.1f} por episódio")
    table.add_row("Logs coletados (total)", str(metrics["total_logs"]))

    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia agente MineRL treinado")
    parser.add_argument("--model", required=True, help="Caminho para o modelo .zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        config_path=args.config,
    )