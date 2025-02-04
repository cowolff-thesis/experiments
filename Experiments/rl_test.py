from pettingzoo.butterfly import pistonball_v6
from ThesisPackage.RL.Decentralized_PPO.multi_ppo import PPO_Multi_Agent
from ThesisPackage.Wrappers.frame_stack import ParallelFrameStack
from ThesisPackage.Wrappers.vecWrapper import PettingZooVectorizationParallelWrapper
import torch
import time

def make_env():
    env = pistonball_v6.parallel_env()
    # env = ParallelFrameStack(env, 4)
    return env

if __name__ == "__main__":
    num_envs = 64
    seed = 1
    total_timesteps = 1000000

    envs = [make_env() for i in range(num_envs)]

    agent = PPO_Multi_Agent(envs, device="cpu")

    agent.train(total_timesteps, tensorboard_folder="RLTest", exp_name=f"pistonball", anneal_lr=True, learning_rate=0.001, num_checkpoints=0)