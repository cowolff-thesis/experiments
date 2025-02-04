from ThesisPackage.Environments.collectors.collectors_env_discrete import Collectors
from ThesisPackage.RL.Decentralized_PPO.multi_ppo import PPO_Multi_Agent
from ThesisPackage.Wrappers.frame_stack import ParallelFrameStack
from ThesisPackage.Wrappers.vecWrapper import PettingZooVectorizationParallelWrapper
import torch
import time

def make_env():
    sequence_length = 2
    vocab_size = 3
    max_episode_steps = 2048
    env = Collectors(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_timesteps=max_episode_steps, timestep_countdown=16)
    # env = ParallelFrameStack(env, 4)
    return env

def make_env_no_language():
    sequence_length = 0
    max_episode_steps = 2048
    env = Collectors(width=20, height=20, vocab_size=0, sequence_length=sequence_length, max_timesteps=max_episode_steps, timestep_countdown=16)
    # env = ParallelFrameStack(env, 4)
    return env

if __name__ == "__main__":
    num_envs = 64
    seed = 1
    total_timesteps = 300000000

    envs = PettingZooVectorizationParallelWrapper(make_env, num_envs)
    agent = PPO_Multi_Agent(envs, device="cpu")
    agent.agent.load_state_dict(torch.load("models/collectors_discrete.pt"))
    agent.train(total_timesteps, tensorboard_folder="collectors", exp_name="collect", anneal_lr=True, learning_rate=0.0005)
    agent.save("models/collectors_discrete")