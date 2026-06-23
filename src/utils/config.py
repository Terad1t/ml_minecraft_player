"""
Carrega e valida o arquivo de configuração YAML central.
Uso: cfg = load_config() → acesso via cfg.ppo.learning_rate
"""

from __future__ import annotations

import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Converte dict aninhado em SimpleNamespace para acesso por atributo."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def load_config(path: str | Path = "configs/config.yaml") -> SimpleNamespace:
    """
    Carrega o config YAML e retorna como namespace navegável.

    Exemplo:
        cfg = load_config()
        cfg.ppo.learning_rate      # 0.0003
        cfg.environment.frame_stack  # 4
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config não encontrado: {config_path.resolve()}\n"
            "Certifique-se de rodar o projeto a partir da raiz."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = _dict_to_namespace(raw)

    # Garante que diretórios de saída existem
    for attr in ["logs", "models", "checkpoints", "tensorboard"]:
        path_val = getattr(cfg.paths, attr, None)
        if path_val:
            Path(path_val).mkdir(parents=True, exist_ok=True)

    return cfg