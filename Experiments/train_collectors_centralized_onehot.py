from ThesisPackage.Environments.collectors.collectors_env_discrete_onehot import Collectors
from ThesisPackage.RL.Centralized_PPO.multi_ppo import PPO_Multi_Agent_Centralized
from ThesisPackage.Wrappers.frame_stack import ParallelFrameStack
from ThesisPackage.Wrappers.vecWrapper import PettingZooVectorizationParallelWrapper
import torch
import time

def make_env(sequence_length=0):
    vocab_size = 4
    max_episode_steps = 2048
    env = Collectors(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_timesteps=max_episode_steps, timestep_countdown=15)
    # env = ParallelFrameStack(env, 4)
    return env

if __name__ == "__main__":
    num_envs = 64
    seed = 1
    total_timesteps = 6000000000
    
    for sequence_length in [1, 2, 3, 4, 0]:
        envs = [make_env(sequence_length) for i in range(num_envs)]

        agent = PPO_Multi_Agent_Centralized(envs, device="cpu")

        agent.train(total_timesteps, tensorboard_folder="OneHot", exp_name=f"collect_seq_{sequence_length}", anneal_lr=True, learning_rate=0.001, num_checkpoints=60)

        agent.save(f"models/collectors_seq_{sequence_length}")