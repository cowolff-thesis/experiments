import os
import time
from ThesisPackage.Environments.multi_pong_language import PongEnv
from ThesisPackage.RL.Decentralized_PPO.multi_ppo import PPO_Multi_Agent
from ThesisPackage.Wrappers.frame_stack import ParallelFrameStack

def make_env(seed, vocab_size, sequence_length, max_episode_steps):
    env = PongEnv(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    env = ParallelFrameStack(env, 4)
    return env

def benchmark(num_envs, device, seed, sequence_length, vocab_size, max_episode_steps, total_timesteps, tensorboard_folder):
    envs = [make_env(seed, vocab_size, sequence_length, max_episode_steps) for _ in range(num_envs)]
    agent = PPO_Multi_Agent(envs, device=device)
    
    exp_name = f"{device}_{num_envs}_envs"
    start_time = time.time()
    agent.train(total_timesteps, tensorboard_folder=tensorboard_folder, exp_name=exp_name)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Training on {device} with {num_envs} environments took {duration:.2f} seconds.")

if __name__ == "__main__":
    seed = 1
    sequence_length = 3
    vocab_size = 3
    max_episode_steps = 2048
    total_timesteps = 8000000
    tensorboard_folder = "benchmark"

    # Define the configurations to test
    num_envs_list = [1024]
    devices = ["mps", "cpu"]

    for device in devices:
        for num_envs in num_envs_list:
            benchmark(num_envs, device, seed, sequence_length, vocab_size, max_episode_steps, total_timesteps, tensorboard_folder)
