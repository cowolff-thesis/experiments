from ThesisPackage.Environments.pong.multi_pong_language_continuous import PongEnv
from ThesisPackage.RL.Decentralized_PPO.multi_ppo import PPO_Multi_Agent
from ThesisPackage.RL.Decentralized_PPO.util import flatten_list, reverse_flatten_list_with_agent_list
from ThesisPackage.Wrappers.vecWrapper import PettingZooVectorizationParallelWrapper
import torch
import numpy as np
import time

def make_env():
    sequence_length = 0
    vocab_size = 3
    max_episode_steps = 2048
    env = PongEnv(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    # env = ParallelFrameStack(env, 4)
    return env

# Inference
env = make_env()
obs, infos = env.reset()
for i in range(10):
    while True:
        obs = flatten_list([obs])
        env.render_console()
        time.sleep(0.1)
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = {agent: np.concatenate(env.action_space(agent).sample()) for agent in env.agents}
        obs, reward, truncations, terminations, infos = env.step(actions)
        if any([truncations[agent] or terminations[agent] for agent in env.agents]):
            break
    env.reset()