import os
import random
import numpy as np
import ray
from ray import tune
from ray.tune import Tuner, RunConfig, CheckpointConfig
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog

from ntn_neurocomputing_v0.wnn.ntn_model import NTNModel


ModelCatalog.register_custom_model("ntn_model", NTNModel)

ray.shutdown()
ray.init()
# registre o seu modelo antigo (ModelV2)


if __name__ == "__main__":
    # inicia o ray (pode usar local_mode=True se quiser)

    seed = 12345678
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # --- CRIA CONFIG COM PPOConfig E DESATIVA O NOVO API STACK ---
    config = (
        PPOConfig()
        .framework("torch")
        .environment(env="CartPole-v1")
        .training(
            lr=0.003,
            num_sgd_iter=1,
            minibatch_size=128,
        )
        .env_runners(num_env_runners=1)  # num_workers equivalent
        # desativa novo API stack (compatibilidade com custom_model)
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        # configura o modelo (ModelV2 style)
        .training(model={
            "custom_model": "ntn_model",
            "custom_model_config": {
                "seed": 123,
                "tuple_size": 8,
                "encoding": {
                    "enc_type": "circular",
                    "resolution": 64,
                    "min": -1.5,
                    "max": 1.5
                }
            }
        })
    )

    # Transforme para dict para passar ao tune.run
    config_dict = config.to_dict()

    config_dict.update({
      "checkpoint_trainable_policies_only": False,
      "keep_per_episode_data": True,
      "keep_checkpoints_num": 5,
      "export_native_model_files": True,  # garante que .pt será exportado
    })

    storage_path = os.path.abspath(os.path.dirname(__file__))
    
    tuner = Tuner(
        PPO,
        param_space=config_dict,
        run_config=RunConfig(
            name="experiment",
            storage_path=storage_path,
            stop={"timesteps_total":5000},
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=5,
                checkpoint_at_end=True,
                num_to_keep=4,
            ),
        ),
        tune_config=tune.TuneConfig(
            num_samples=2,
            metric="env_runners/episode_reward_mean",
            mode="max",
        )
    )
    ## path=G:/Documentos/codes/IC_codes_studies/NTN_katopodis/ntn_neurocomputing_v0/wnn/cartpole/circular/individual_run/experiment/PPO_CartPole-v1_78063_00001_1_2026-03-09_19-01-04/checkpoint_000000
    # Evite resumir experimentos antigos enquanto testa:
    # analysis = tune.run(
    #     PPO,  # use a classe PPO aqui
    #     name="experiment",
    #     storage_path=os.path.abspath(current_directory),
    #     resume=False,            # <- importantíssimo para testar a mudança
    #     num_samples=2, # <- 20
    #     config=config_dict,
    #     stop={'timesteps_total': 5_000}, # <- 1_000_000
    #     checkpoint_freq=5,
    #     checkpoint_at_end=True,
    #     keep_checkpoints_num=5,
    #     metric="env_runners/episode_reward_mean",
    #     mode="max"
    # )
    results = tuner.fit()

best_result = results.get_best_result(
    metric="env_runners/episode_reward_mean",
    mode="max",
)

best_check = best_result.checkpoint

algo = PPO(
    config=config_dict,
)
algo.restore(best_check.path)
print(storage_path)
ray.shutdown()

# if __name__ == "__main__":
#     ray.init()

#     ModelCatalog.register_custom_model("ntn_model", NTNModel)

#     seed = 12345678

#     random.seed(seed)
#     np.random.seed(seed)
#     rng = np.random.default_rng(seed)

#     analysis = tune.run(
#         "PPO",
#         name="experiment",
#         local_dir=os.path.abspath(os.path.dirname(__file__)),
#         resume="AUTO",
#         num_samples=20,
#         config={
#             "env": "CartPole-v1",
#             "framework": "torch",
#             "num_workers": 2,
#             "seed": tune.sample_from(lambda _: int(rng.integers(1_000, int(1e6)))),
#             "lr": 0.003,
#             "observation_filter": "MeanStdFilter",
#             "num_sgd_iter": 1,
#             "sgd_minibatch_size": 128,
#             "model": {
#                 "custom_model": "ntn_model",
#                 "custom_model_config": {
#                     "seed": tune.sample_from(lambda _: int(rng.integers(1_000, int(1e6)))),
#                     "tuple_size": 8,
#                     "encoding": {
#                         "enc_type": "circular",
#                         "resolution": 64,
#                         "min": -1.5,
#                         "max": 1.5
#                     }
#                 },
#             },
#         },
#         stop={ 'timesteps_total': 1_000_000 },
#         checkpoint_freq=5,
#         checkpoint_at_end=True
#     )

#     ray.shutdown()