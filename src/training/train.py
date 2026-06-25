"""
Loop principal de treino — Fase 1 e 2.

Fluxo:
  1. Carrega config
  2. Cria ambiente + wrappers
  3. Cria agente PPO
  4. model.learn() → coleta rollouts → updates PPO → loga TensorBoard
  5. Salva modelo final
"""

from __future__ import annotations

import sys
from pathlib import Path

# Garante que src/ está no path quando rodado da raiz do projeto
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import (
    load_config,
    device_info,
    print_header,
    print_env_info,
    print_training_config,
)
from src.environment import make_env, make_eval_env
from src.agent import make_ppo_agent, make_callbacks


def train(config_path: str = "configs/config.yaml") -> None:
    """Executa o loop de treino completo."""

    # ── 1. Configuração ──────────────────────────────────────
    cfg = load_config(config_path)
    info = device_info()
    print_header(cfg.project.name, cfg.project.phase, info)
    print_training_config(cfg)

    # ── 2. Ambiente ──────────────────────────────────────────
    from rich.console import Console
    console = Console()
    console.print("\n[bold]Inicializando ambiente MineRL...[/bold]")
    console.print("[dim]Isso pode demorar na primeira vez (download do Minecraft)[/dim]\n")

    env = make_env(cfg=cfg, seed=cfg.environment.seed)
    eval_env = make_eval_env(cfg=cfg, seed=cfg.environment.seed + 100)

    print_env_info(env)

    # ── 3. Agente ────────────────────────────────────────────
    console.print("\n[bold]Criando agente PPO...[/bold]")
    agent = make_ppo_agent(env, cfg, eval_env=eval_env)
    callbacks = make_callbacks(cfg, eval_env=eval_env)

    # ── 4. Treino ────────────────────────────────────────────
    console.print(
        f"\n[bold green]Iniciando treino:[/bold green] "
        f"{cfg.ppo.total_timesteps:,} timesteps\n"
    )
    console.print(
        "Acompanhe o progresso:\n"
        f"  [cyan]tensorboard --logdir {cfg.paths.tensorboard}[/cyan]\n"
    )

    agent.learn(
        total_timesteps=cfg.ppo.total_timesteps,
        callback=callbacks,
        tb_log_name=cfg.training.tb_log_name,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # ── 5. Salvar modelo final ───────────────────────────────
    final_path = Path(cfg.paths.models) / "ppo_minecraft_final"
    agent.save(str(final_path))
    console.print(f"\n[bold green]✓ Modelo salvo:[/bold green] {final_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()