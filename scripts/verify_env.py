"""
Verifica se toda a stack está funcionando corretamente.
Rode ANTES de iniciar o treino.

    uv run python scripts/verify_env.py
"""

import sys
from rich.console import Console
from rich.table import Table

console = Console()

results = []


def check(name: str, fn):
    try:
        value = fn()
        results.append((name, "OK", str(value), "green"))
    except Exception as e:
        results.append((name, "FALHOU", str(e)[:80], "red"))


# ── Verificações ─────────────────────────────────────────────

check("Python version", lambda: sys.version.split()[0])

check("PyTorch", lambda: __import__("torch").__version__)

check(
    "DirectML",
    lambda: (
        lambda tdml: f"{tdml.device_name(0)}"
    )(__import__("torch_directml")),
)

check(
    "DirectML tensor op",
    lambda: (
        lambda tdml, torch: str(
            (torch.tensor([2.0]).to(tdml.device()) * 3).item()
        )
    )(__import__("torch_directml"), __import__("torch")),
)

check("Stable-Baselines3", lambda: __import__("stable_baselines3").__version__)

check("Gymnasium", lambda: __import__("gymnasium").__version__)

check("OpenCV", lambda: __import__("cv2").__version__)

check("NumPy", lambda: __import__("numpy").__version__)

check("TensorBoard", lambda: __import__("tensorboard").__version__)

check(
    "MineRL",
    lambda: __import__("minerl").__version__,
)

check(
    "Java (MineRL dep)",
    lambda: __import__("subprocess")
    .check_output(["java", "-version"], stderr=__import__("subprocess").STDOUT)
    .decode()
    .split("\n")[0],
)

check("Config YAML", lambda: (
    lambda cfg: f"phase={cfg.project.phase}, env={cfg.environment.env_id}"
)(sys.path.insert(0, ".") or __import__("src.utils", fromlist=["load_config"]).load_config()))

# ── Resultado ────────────────────────────────────────────────

table = Table(title="Verificação do Ambiente minecraft-rl", show_lines=True)
table.add_column("Componente", style="dim", min_width=20)
table.add_column("Status", min_width=8)
table.add_column("Detalhe")

for name, status, detail, color in results:
    table.add_row(name, f"[{color}]{status}[/{color}]", detail)

console.print("\n")
console.print(table)

failed = [r for r in results if r[1] == "FALHOU"]
if failed:
    console.print(
        f"\n[red bold]{len(failed)} verificação(ões) falharam.[/red bold] "
        "Resolva antes de iniciar o treino."
    )
    if any("DirectML" in r[0] for r in failed):
        console.print(
            "\n[yellow]DirectML não encontrado:[/yellow] rode o setup:\n"
            "  [cyan]uv pip install torch-directml[/cyan]"
        )
    if any("MineRL" in r[0] for r in failed):
        console.print(
            "\n[yellow]MineRL não encontrado:[/yellow] instale:\n"
            "  [cyan]uv pip install minerl==0.4.4[/cyan]"
        )
    sys.exit(1)
else:
    console.print(
        "\n[bold green]✓ Tudo verificado! Ambiente pronto.[/bold green]\n"
        "\nPróximo passo — agente aleatório (Fase 1):\n"
        "  [cyan]uv run python -m src.training.random_agent[/cyan]"
    )