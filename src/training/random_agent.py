"""
Fase 1 — Agente Aleatório.

Objetivo: entender o ambiente MineRL antes de treinar qualquer coisa.
  - Confirma que o env está funcionando
  - Mostra o formato de obs e action
  - Mede FPS e duração de episódio
  - Loga reward total (esparso — provavelmente 0 na maioria dos eps)

Rode com:
    python -m src.training.random_agent
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from turtle import done

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from rich.console import Console
from rich.table import Table

from src.utils import load_config, print_header, device_info
from src.environment import make_env

console = Console()


def run_random_agent(
    n_episodes: int = 3,
    max_steps: int = 500,
    config_path: str = "configs/config.yaml",
) -> None:
    """
    Roda o agente aleatório por n_episodes.
    Útil para verificar o ambiente e entender o MDP.
    """
    cfg = load_config(config_path)
    info = device_info()
    print_header(cfg.project.name, cfg.project.phase, info)

    console.print("[bold]Modo:[/bold] Agente Aleatório (Fase 1)\n")
    console.print("[dim]Inicializando MineRL...[/dim]")

    env = make_env(cfg=cfg, reward_shaping=False)  # sem shaping — reward puro

    # Inspeciona o ambiente
    _print_spaces(env)

    results = []

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        ep_steps = 0
        start = time.time()

        console.print(f"\n[bold]Episódio {ep + 1}/{n_episodes}[/bold]")

        for step in range(max_steps):
            # Ação aleatória — amostra do espaço de ações
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break

            ep_reward += reward
            ep_steps += 1

            # Loga quando coleta madeira (reward esparso)
            if reward > 0:
                console.print(
                    f"  [green]✓ Madeira coletada![/green] "
                    f"step={step}, reward={reward:.2f}"
                )

        elapsed = time.time() - start
        fps = ep_steps / elapsed if elapsed > 0 else 0

        results.append({
            "ep": ep + 1,
            "reward": ep_reward,
            "steps": ep_steps,
            "fps": fps,
        })

        console.print(
            f"  reward={ep_reward:.2f} | "
            f"steps={ep_steps} | "
            f"fps={fps:.1f}"
        )

    env.close()
    _print_summary(results)


def _print_spaces(env) -> None:
    """Imprime observation_space e action_space do ambiente."""
    table = Table(title="Espaços do Ambiente MineRL")
    table.add_column("Espaço", style="dim")
    table.add_column("Tipo", style="cyan")
    table.add_column("Shape / Valores")

    obs_space = env.observation_space
    table.add_row(
        "observation_space",
        type(obs_space).__name__,
        str(obs_space.shape),
    )

    action_space = env.action_space
    table.add_row(
        "action_space",
        type(action_space).__name__,
        str(action_space),
    )

    console.print(table)
    console.print(
        f"\n[dim]Frame stack ativo — obs shape: {obs_space.shape}[/dim]\n"
        f"[dim]Interpretação: ({obs_space.shape[0]} canais = "
        f"{obs_space.shape[0] // 3} frames × 3 RGB, "
        f"{obs_space.shape[1]}×{obs_space.shape[2]} pixels)[/dim]\n"
    )


def _print_summary(results: list[dict]) -> None:
    table = Table(title="Resumo — Agente Aleatório")
    table.add_column("Episódio", style="dim")
    table.add_column("Reward Total", style="green")
    table.add_column("Steps", style="cyan")
    table.add_column("FPS")

    for r in results:
        table.add_row(
            str(r["ep"]),
            f"{r['reward']:.2f}",
            str(r["steps"]),
            f"{r['fps']:.1f}",
        )

    rewards = [r["reward"] for r in results]
    table.add_row(
        "[bold]Média[/bold]",
        f"[bold]{np.mean(rewards):.2f}[/bold]",
        f"[bold]{np.mean([r['steps'] for r in results]):.0f}[/bold]",
        f"[bold]{np.mean([r['fps'] for r in results]):.1f}[/bold]",
    )

    console.print("\n")
    console.print(table)

    if all(r["reward"] == 0 for r in results):
        console.print(
            "\n[yellow]Reward = 0 em todos os episódios — normal para agente aleatório.[/yellow]\n"
            "[dim]O reward esparso do MineRL raramente é atingido por ações aleatórias.\n"
            "Isso é exatamente o problema que o PPO + reward shaping vai resolver.[/dim]"
        )


if __name__ == "__main__":
    run_random_agent()