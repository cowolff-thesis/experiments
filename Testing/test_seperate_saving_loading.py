from ThesisPackage.Environments.soccer.soccer_env_multi import SoccerGame
from ThesisPackage.RL.Seperated_PPO_soccer.multi_ppo import PPO_Multi_Agent
from ThesisPackage.RL.Decentralized_PPO.util import flatten_list, reverse_flatten_list_with_agent_list
from ThesisPackage.Wrappers.frame_stack import ParallelFrameStack
from ThesisPackage.Wrappers.vecWrapper import PettingZooVectorizationParallelWrapper
import torch
import time

def make_env():
    env = SoccerGame(20, 12, sequence_length=2, vocab_size=3, noise_team=-1)
    env = ParallelFrameStack(env, 4)
    return env

def train():
    num_envs = 64
    seed = 1
    total_timesteps = 1000000
    envs = PettingZooVectorizationParallelWrapper(make_env, num_envs)

    test_env = SoccerGame(20, 12, sequence_length=2, vocab_size=3, noise_team=1)
    test_env = ParallelFrameStack(test_env, 4)
    # envs = [make_env(seed, vocab_size, sequence_length, max_episode_steps) for i in range(num_envs)]
    # agent = PPO_Multi_Agent(envs, test_env=test_env, device="cpu")
    agent = PPO_Multi_Agent(envs, test_env=test_env, device="cpu")
    # agent.agent.load_state_dict(torch.load("models/soccer_2vs2.pt"))
    # agent.train(total_timesteps, learn_team=1, lr_auto_adjust=True, anneal_lr=False)
    agent.train(total_timesteps, tensorboard_folder="soccerResults", exp_name="soccer2vs2_separated", learn_team=1, lr_auto_adjust=True, anneal_lr=False)

    agent.save_models_to_zip("models/soccer_2vs2_separated.zip")

def load():
    num_envs = 8
    seed = 1
    total_timesteps = 1000000
    envs = PettingZooVectorizationParallelWrapper(make_env, num_envs)

    test_env = SoccerGame(20, 12, sequence_length=2, vocab_size=3, noise_team=1)
    test_env = ParallelFrameStack(test_env, 4)
    # envs = [make_env(seed, vocab_size, sequence_length, max_episode_steps) for i in range(num_envs)]
    # agent = PPO_Multi_Agent(envs, test_env=test_env, device="cpu")
    agent = PPO_Multi_Agent(envs, test_env=test_env, device="cpu")
    agent.load_models_from_zip("models/soccer_2vs2_separated.zip")
    agent.train(total_timesteps, tensorboard_folder="soccerResults", exp_name="soccer2vs2_separated", learn_team=1, lr_auto_adjust=True, anneal_lr=False)

    # agent.save_models_to_zip("models/soccer_2vs2_separated.zip")

if __name__ == "__main__":
    load()