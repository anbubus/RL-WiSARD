trained_model_path = r"G:/Documentos/codes/IC_codes_studies/NTN_katopodis/ntn_neurocomputing_v0/wnn/cartpole/circular/individual_run/experiment/PPO_CartPole-v1_4219d_00001_1_2026-03-09_19-35-22/checkpoint_000000"


import os
import sys
import random
import numpy as np
import ray
from ray import tune
from ray.tune import Tuner, RunConfig, CheckpointConfig
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog

sys.path.append(sys.path.append("G:/Documentos/codes/IC_codes_studies/NTN_katopodis"))
from ntn_neurocomputing_v0.wnn.ntn_model import NTNModel

# import os
# import sys

# MODULE_PATH = "G:/Documentos/codes/IC_codes_studies/NTN_katopodis"
# sys.path.insert(0, MODULE_PATH)
# os.environ["PYTHONPATH"] = MODULE_PATH

# # ── Função que roda em CADA worker ao iniciar ──────────────────────
# def setup_workers():
#     sys.path.insert(0, "G:/Documentos/codes/IC_codes_studies/NTN_katopodis")
#     os.environ["PYTHONPATH"] = "G:/Documentos/codes/IC_codes_studies/NTN_katopodis"
    
#     from ray.rllib.models import ModelCatalog
#     from ntn_neurocomputing_v0.wnn.ntn_model import NTNModel # ajuste aqui
#     ModelCatalog.register_custom_model("ntn_model", NTNModel)

# ── Ray com worker_setup ───────────────────────────────────────────

# if ray.is_initialized():
#     ray.shutdown()

# ray.init(
#     ignore_reinit_error=True,
#     include_dashboard=False,
#     runtime_env={
#         "worker_process_setup_hook": setup_workers,
#         "env_vars": {"PYTHONPATH": MODULE_PATH}
#     }
# )

# ── Registrar no processo principal também ─────────────────────────

ModelCatalog.register_custom_model("ntn_model", NTNModel)
print("Modelo registrado!")

# ── Restaurar ─────────────────────────────────────────────────────

algo = PPO.from_checkpoint(trained_model_path)
print("Checkpoint restaurado!")

# ── Renderizar ────────────────────────────────────────────────────
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="rgb_array")

for episodio in range(5):
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = algo.compute_single_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episódio {episodio + 1} — Reward: {total_reward}")

env.close()