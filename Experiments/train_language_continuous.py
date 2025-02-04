from ThesisPackage.Environments.pong.multi_pong_language_continuous import PongEnv
from ThesisPackage.RL.Decentralized_PPO.multi_ppo import PPO_Multi_Agent
from ThesisPackage.RL.Decentralized_PPO.util import flatten_list, reverse_flatten_list_with_agent_list
from ThesisPackage.Wrappers.vecWrapper import PettingZooVectorizationParallelWrapper
import torch
import numpy as np
import time

def make_env():
    sequence_length = 1
    vocab_size = 4
    max_episode_steps = 1024
    env = PongEnv(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    # env = ParallelFrameStack(env, 4)
    return env

if __name__ == "__main__":
    num_envs = 64
    seed = 1
    total_timesteps = 500000000
    # envs = PettingZooVectorizationParallelWrapper(make_env, num_envs)
    envs = [make_env() for i in range(num_envs)]
    agent = PPO_Multi_Agent(envs, device="cpu", normalize_obs=False)
    agent.train(total_timesteps, tensorboard_folder="continuous", exp_name="continuous", lr_auto_adjust=False, anneal_lr=True, num_checkpoints=20, learning_rate=0.001)
    agent.save("models/pong_continuous")