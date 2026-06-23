"""
Testes do pipeline de ambiente.
Rodar com: uv run pytest tests/ -v
"""

import numpy as np
import pytest


class TestObservationWrapper:
    """Testa MineRLObservationWrapper sem precisar do MineRL real."""

    def test_observation_shape_transpose(self):
        """Frame (64,64,3) deve virar (3,64,64)."""
        from unittest.mock import MagicMock, patch
        import gymnasium as gym
        from gymnasium import spaces
        from src.environment.wrappers import MineRLObservationWrapper

        # Mock do env base
        mock_env = MagicMock()
        mock_env.observation_space = spaces.Dict({
            "pov": spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
        })

        wrapper = MineRLObservationWrapper(mock_env)

        # Testa transformação
        fake_obs = {"pov": np.zeros((64, 64, 3), dtype=np.uint8)}
        result = wrapper.observation(fake_obs)

        assert result.shape == (3, 64, 64), f"Esperado (3,64,64), got {result.shape}"
        assert result.dtype == np.uint8

    def test_observation_space_updated(self):
        """observation_space deve refletir o novo shape."""
        from unittest.mock import MagicMock
        from gymnasium import spaces
        from src.environment.wrappers import MineRLObservationWrapper

        mock_env = MagicMock()
        mock_env.observation_space = spaces.Dict({
            "pov": spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)
        })

        wrapper = MineRLObservationWrapper(mock_env)
        assert wrapper.observation_space.shape == (3, 64, 64)


class TestRewardShaping:
    """Testa o reward shaping sem precisar do MineRL real."""

    def _make_wrapper(self):
        from unittest.mock import MagicMock
        from gymnasium import spaces
        from src.environment.wrappers import RewardShapingWrapper

        mock_env = MagicMock()
        mock_env.observation_space = spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
        mock_env.action_space = spaces.Discrete(5)
        mock_env.step.return_value = (
            np.zeros((3, 64, 64), dtype=np.uint8),
            0.0,  # reward original
            False, False, {}
        )
        mock_env.reset.return_value = (np.zeros((3, 64, 64), dtype=np.uint8), {})
        return RewardShapingWrapper(mock_env)

    def test_attack_gives_bonus(self):
        """Atacar deve dar reward positivo além do original."""
        wrapper = self._make_wrapper()
        wrapper.reset()

        action = {"attack": 1, "forward": 0}
        _, reward, _, _, _ = wrapper.step(action)
        assert reward > 0.0, "Atacar deveria dar reward positivo"

    def test_idle_penalized_after_threshold(self):
        """Ficar parado por muitos steps deve dar penalidade."""
        wrapper = self._make_wrapper()
        wrapper.reset()
        wrapper._max_idle = 5  # baixa o threshold para o teste

        action = {"attack": 0, "forward": 0}
        reward = 0.0
        for _ in range(10):  # mais que _max_idle
            _, reward, _, _, _ = wrapper.step(action)

        assert reward < 0.0, "Inatividade prolongada deveria ter penalidade"

    def test_original_reward_preserved(self):
        """Reward original > 0 deve ser mantido e amplificado."""
        from unittest.mock import MagicMock
        from gymnasium import spaces
        from src.environment.wrappers import RewardShapingWrapper

        mock_env = MagicMock()
        mock_env.observation_space = spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
        mock_env.action_space = spaces.Discrete(5)
        mock_env.step.return_value = (
            np.zeros((3, 64, 64), dtype=np.uint8),
            1.0,  # reward original = coletou madeira
            False, False, {}
        )
        mock_env.reset.return_value = (np.zeros((3, 64, 64), dtype=np.uint8), {})

        wrapper = RewardShapingWrapper(mock_env)
        wrapper.reset()
        _, reward, _, _, _ = wrapper.step({"attack": 0})

        assert reward > 1.0, "Coletar madeira deveria dar reward aumentado"


class TestDeviceDetection:
    """Testa detecção de device."""

    def test_cpu_forced(self):
        """Backend 'cpu' deve retornar device CPU."""
        import torch
        from src.utils.device import get_device

        device = get_device("cpu")
        assert str(device) == "cpu"

    def test_auto_returns_device(self):
        """Backend 'auto' deve retornar algum device válido."""
        import torch
        from src.utils.device import get_device

        device = get_device("auto")
        # Deve ser cpu ou privateuseone (DirectML)
        assert device is not None


class TestConfig:
    """Testa carregamento de configuração."""

    def test_load_config(self, tmp_path):
        """Config deve ser carregado corretamente do YAML."""
        import yaml
        from src.utils.config import load_config

        cfg_data = {
            "project": {"name": "test", "phase": 1},
            "environment": {"env_id": "MineRLTreechop-v0", "frame_stack": 4,
                           "frame_size": 64, "grayscale": False,
                           "max_episode_steps": 8000, "n_envs": 1, "seed": 42},
            "ppo": {"total_timesteps": 1000, "learning_rate": 3e-4,
                   "n_steps": 128, "batch_size": 32, "n_epochs": 4,
                   "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2,
                   "clip_range_vf": None, "ent_coef": 0.01, "vf_coef": 0.5,
                   "max_grad_norm": 0.5, "normalize_advantage": True},
            "reward_shaping": {"enabled": True, "look_at_log": 0.02,
                              "attack_log": 0.05, "collect_log": 1.0,
                              "approach_tree": 0.01, "penalty_idle": -0.001},
            "training": {"log_interval": 10, "save_freq": 1000,
                        "eval_freq": 500, "eval_episodes": 2,
                        "tb_log_name": "test"},
            "paths": {"logs": str(tmp_path / "logs") + "/",
                     "models": str(tmp_path / "models") + "/",
                     "checkpoints": str(tmp_path / "checkpoints") + "/",
                     "tensorboard": str(tmp_path / "tb") + "/"},
            "device": {"backend": "cpu"},
        }

        cfg_file = tmp_path / "config.yaml"
        with open(cfg_file, "w") as f:
            yaml.dump(cfg_data, f)

        cfg = load_config(cfg_file)
        assert cfg.project.name == "test"
        assert cfg.ppo.learning_rate == 3e-4
        assert cfg.environment.frame_stack == 4