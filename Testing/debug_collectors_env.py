from ThesisPackage.Environments.collectors.collectors_env_discrete import Collectors
from ThesisPackage.RL.Decentralized_PPO.util import flatten_list
import torch
import numpy as np
import time

def make_env():
    sequence_length = 1
    vocab_size = 3
    max_episode_steps = 2048
    env = Collectors(width=20, height=12, vocab_size=vocab_size, sequence_length=sequence_length, max_timesteps=max_episode_steps)
    # env = ParallelFrameStack(env, 4)
    return env

# Inference
env = make_env()
obs, infos = env.reset()
for i in range(10):
    while True:
        obs = np.array(flatten_list([obs]))
        
        time.sleep(1)
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, reward, truncations, terminations, infos = env.step(actions)
        env.render()
        if any([truncations[agent] or terminations[agent] for agent in env.agents]):
            continue
    env.reset()