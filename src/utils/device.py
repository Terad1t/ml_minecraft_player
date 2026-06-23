"""
Detecção automática de backend: DirectML (AMD GPU) ou CPU.

DirectML usa o device string "privateuseone" no PyTorch.
O SB3 aceita qualquer device via th.device().
"""

import torch
from rich.console import Console

console = Console()


def get_device(backend: str = "auto") -> torch.device:
    """
    Retorna o device correto baseado na configuração.

    Args:
        backend: "auto" | "cpu" | "privateuseone" (DirectML)

    Returns:
        torch.device configurado
    """
    if backend == "cpu":
        console.print("[yellow]Device:[/yellow] CPU (forçado por config)")
        return torch.device("cpu")

    if backend == "privateuseone":
        return _try_directml()

    # auto: tenta DirectML, cai para CPU
    return _try_directml()


def _try_directml() -> torch.device:
    """Tenta inicializar DirectML. Retorna CPU se não disponível."""
    try:
        import torch_directml

        dml_device = torch_directml.device()
        # Teste rápido para confirmar que funciona
        test = torch.tensor([1.0]).to(dml_device)
        _ = test + 1
        console.print(
            f"[green]Device:[/green] DirectML — {torch_directml.device_name(0)} "
            f"(AMD GPU acelerada)"
        )
        return dml_device
    except ImportError:
        console.print(
            "[yellow]Device:[/yellow] CPU "
            "(torch-directml não instalado — rode scripts/setup_directml.ps1)"
        )
        return torch.device("cpu")
    except Exception as e:
        console.print(f"[yellow]Device:[/yellow] CPU (DirectML falhou: {e})")
        return torch.device("cpu")


def device_info() -> dict:
    """Retorna informações do device para logging."""
    info = {
        "torch_version": torch.__version__,
        "directml_available": False,
        "device": "cpu",
    }
    try:
        import torch_directml

        info["directml_available"] = True
        info["device"] = "privateuseone"
        info["gpu_name"] = torch_directml.device_name(0)
    except ImportError:
        pass
    return info