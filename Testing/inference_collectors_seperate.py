from ThesisPackage.Environments.collectors.collectors_env import Collectors
from ThesisPackage.RL.Decentralized_PPO.util import flatten_list, reverse_flatten_list_with_agent_list
import torch
import numpy as np
import time
from ThesisPackage.RL.Centralized_PPO.multi_ppo import PPO_Multi_Agent_Centralized

def make_env():
    sequence_length = 2
    vocab_size = 3
    max_episode_steps = 2048
    env = Collectors(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_timesteps=max_episode_steps, timestep_countdown=13)
    # env = ParallelFrameStack(env, 4)
    return env

# Inference
if __name__ == "__main__":
    num_envs = 64
    seed = 1
    total_timesteps = 800000000
    env = make_env()

    agent = PPO_Multi_Agent_Centralized(env, device="cpu")
    agent.agent.load_state_dict(torch.load("models/soccer_2vs2_randomized.pt"))

    obs, infos = env.reset()
    state = env.state()
    for i in range(10240):
        obs = flatten_list([obs])
        state = flatten_list([state])
        env.render()
        time.sleep(0.3)
        obs = torch.tensor(obs, dtype=torch.float32)
        state = torch.tensor(state, dtype=torch.float32)
        actions, _, _, _ = agent.agent.get_action_and_value(obs,state )
        actions = actions.detach().numpy()
        actions = reverse_flatten_list_with_agent_list(actions, env.agents)
        obs, reward, truncations, terminations, infos = env.step(actions[0])
        state = env.state()
        if any([truncations[agent] or terminations[agent] for agent in env.agents]):
            print("Episode done ", env.timestep)
            env.reset()