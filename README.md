# minecraft-rl

Agente de Reinforcement Learning para Minecraft usando **MineRL + PPO (Stable-Baselines3)**.  
Stack: Python 3.10 · PyTorch · torch-directml (AMD RX 7600) · Gymnasium · TensorBoard · uv

---

## Roadmap

| Fase | Objetivo | Status |
|------|----------|--------|
| 1 | Setup + agente aleatório (entender o ambiente) | 🔧 atual |
| 2 | PPO básico + reward shaping (coleta madeira) | ⏳ |
| 3 | CNN customizada + curriculum learning (crafta ferramentas) | ⏳ |
| 4 | Imitation learning com dataset VPT (sobrevive sozinho) | ⏳ |

---

## Estrutura

```
minecraft-rl/
├── configs/
│   └── config.yaml          # Todos os hiperparâmetros centralizados
├── scripts/
│   ├── setup.ps1            # Setup Windows (uv + DirectML)
│   └── verify_env.py        # Verifica instalação
├── src/
│   ├── agent/
│   │   └── policy.py        # Factory do agente PPO
│   ├── environment/
│   │   ├── wrappers.py      # Wrappers Gymnasium customizados
│   │   └── make_env.py      # Pipeline de ambiente
│   ├── training/
│   │   ├── train.py         # Loop principal de treino
│   │   ├── evaluate.py      # Avaliação de agente treinado
│   │   └── random_agent.py  # Fase 1: agente aleatório
│   └── utils/
│       ├── config.py        # Carrega config.yaml
│       ├── device.py        # Detecta DirectML / CPU
│       └── logger.py        # Logger Rich + TensorBoard
├── tests/
│   └── test_environment.py  # Testes unitários dos wrappers
├── logs/                    # Logs de treino e TensorBoard
├── models/                  # Modelos salvos e checkpoints
└── pyproject.toml           # Dependências (uv)
```

---

## Setup

```powershell
# 1. Clone o projeto
git clone <repo>
cd minecraft-rl

# 2. Rode o script de setup (PowerShell como Admin)
.\scripts\setup.ps1

# 3. Verifique a instalação
uv run python scripts/verify_env.py
```

---

## Uso

### Fase 1 — Agente Aleatório
```powershell
uv run python -m src.training.random_agent
```
Confirma que o ambiente MineRL está funcionando e mostra o formato das observações e ações.

### Fase 2 — Treino PPO
```powershell
uv run python -m src.training.train
```

### Monitoramento (TensorBoard)
```powershell
uv run tensorboard --logdir logs/tensorboard
# Abra http://localhost:6006
```

### Métricas importantes no TensorBoard
| Métrica | O que significa |
|---------|----------------|
| `rollout/ep_rew_mean` | Reward médio por episódio — métrica principal |
| `train/value_loss` | Erro do critic ao estimar V(s) — deve diminuir |
| `train/entropy_loss` | Diversidade da política — muito baixo = policy collapsed |
| `train/approx_kl` | Divergência entre política nova e antiga — ideal < 0.1 |
| `train/explained_variance` | Qualidade do critic — ideal próximo de 1 |

### Avaliação de modelo treinado
```powershell
uv run python -m src.training.evaluate --model models/ppo_minecraft_final
```

### Testes
```powershell
uv run pytest tests/ -v
```

---

## Hiperparâmetros

Todos os hiperparâmetros ficam em `configs/config.yaml`. Edite ali sem tocar no código.

Parâmetros mais importantes para tunar na Fase 2:

- `ppo.ent_coef` — aumentar se o agente parar de explorar (entropy cair muito)
- `ppo.learning_rate` — diminuir se o treino for instável
- `ppo.n_steps` — aumentar para rollouts mais longos (mais memória)
- `reward_shaping.*` — ajustar densidades das recompensas intermediárias

---

## Device

O projeto detecta automaticamente:
1. **DirectML** (AMD RX 7600 via `torch-directml`) — preferido
2. **CPU** — fallback automático

Para forçar CPU: mude `device.backend: "cpu"` no `config.yaml`.