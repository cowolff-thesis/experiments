from ThesisPackage.Environments.pong.multi_pong_language_continuous_vectorized import PongEnv
# from ThesisPackage.Environments.pong.multi_pong_language_continuous_vectorized_onehot import PongEnv
# from ThesisPackage.Environments.pong.multi_pong_language_continuous import PongEnv
from ThesisPackage.RL.Decentralized_PPO_vectorized.multi_ppo import PPO_Multi_Agent
import torch
import numpy as np
import time

def make_env():
    sequence_length = 2
    vocab_size = 3
    max_episode_steps = 2048
    env = PongEnv(width=15, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    # env = ParallelFrameStack(env, 4)
    return env

if __name__ == "__main__":
    num_envs = 64
    seed = 1
    for sequence in [2]:
        total_timesteps = 150000000
        envs = PongEnv(num_envs=num_envs, sequence_length=sequence, vocab_size=3, max_episode_steps=2048)
        # envs = [make_env() for i in range(num_envs)]
        agent = PPO_Multi_Agent(envs, device="cpu", normalize_obs=False)
        agent.train(total_timesteps, tensorboard_folder="continuous", exp_name="continuous vectorized", lr_auto_adjust=False, anneal_lr=True, learning_rate=5e-3, num_checkpoints=20)