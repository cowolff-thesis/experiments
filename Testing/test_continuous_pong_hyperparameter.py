# from ThesisPackage.Environments.pong.multi_pong_language_continuous_vectorized import PongEnv
from ThesisPackage.Environments.pong.multi_pong_language_continuous_vectorized_onehot import PongEnv
# from ThesisPackage.Environments.pong.multi_pong_language_continuous import PongEnv
from ThesisPackage.RL.Decentralized_PPO_vectorized.multi_ppo import PPO_Multi_Agent
import torch
import numpy as np
import time

if __name__ == "__main__":
    num_envs = 256
    seed = 1
    total_timesteps = 2000000000
    for lr in [5e-3, 1e-3, 5e-4, 2.5e-4]:
        for num_minibatches in [128, 256, 512]:
            envs = PongEnv(num_envs=num_envs, sequence_length=2, vocab_size=3)
            # envs = [make_env() for i in range(num_envs)]
            agent = PPO_Multi_Agent(envs, device="cuda", normalize_obs=False)
            agent.train(total_timesteps, tensorboard_folder="continuous", exp_name="continuous vectorized", lr_auto_adjust=False, anneal_lr=True, num_minibatches=num_minibatches, learning_rate=lr, num_checkpoints=20)