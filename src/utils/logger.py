"""
Logger centralizado com Rich (console bonito) + TensorBoard.
"""

import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

console = Console()


def print_header(project_name: str, phase: int, device_info: dict) -> None:
    """Imprime cabeçalho do projeto no início do treino."""
    console.rule(f"[bold blue]{project_name}[/bold blue]")
    console.print(f"  [dim]Fase {phase}[/dim]")
    console.print(f"  Torch: [cyan]{device_info['torch_version']}[/cyan]")
    console.print(
        f"  Device: [green]{device_info.get('gpu_name', device_info['device'])}[/green]"
    )
    console.rule()


def print_env_info(env) -> None:
    """Imprime informações do ambiente Gymnasium."""
    table = Table(title="Ambiente MineRL", show_header=True, header_style="bold")
    table.add_column("Propriedade", style="dim")
    table.add_column("Valor", style="cyan")

    table.add_row("Observation Space", str(env.observation_space))
    table.add_row("Action Space", str(env.action_space))

    console.print(table)


def print_training_config(cfg) -> None:
    """Imprime configuração de treino em tabela."""
    table = Table(title="Configuração PPO", show_header=True, header_style="bold")
    table.add_column("Parâmetro", style="dim")
    table.add_column("Valor", style="green")

    ppo = cfg.ppo
    table.add_row("Total Timesteps", f"{ppo.total_timesteps:,}")
    table.add_row("Learning Rate", str(ppo.learning_rate))
    table.add_row("N Steps", str(ppo.n_steps))
    table.add_row("Batch Size", str(ppo.batch_size))
    table.add_row("N Epochs", str(ppo.n_epochs))
    table.add_row("Gamma (γ)", str(ppo.gamma))
    table.add_row("GAE Lambda (λ)", str(ppo.gae_lambda))
    table.add_row("Clip Range (ε)", str(ppo.clip_range))
    table.add_row("Entropy Coef", str(ppo.ent_coef))

    console.print(table)


class TrainingLogger:
    """Logger simples para métricas de treino."""

    def __init__(self, log_dir: str = "logs/"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.log_file = self.log_dir / "training.log"

    def log(self, step: int, metrics: dict) -> None:
        """Loga métricas no console e em arquivo."""
        elapsed = time.time() - self.start_time
        line = f"[step={step:>8,}] elapsed={elapsed:.0f}s | " + " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        console.print(f"[dim]{line}[/dim]")
        with open(self.log_file, "a") as f:
            f.write(line + "\n")