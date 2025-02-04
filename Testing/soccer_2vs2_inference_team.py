from ThesisPackage.Environments.soccer.soccer_env_multi import SoccerGame
from ThesisPackage.RL.Decentralized_PPO_soccer.multi_ppo import PPO_Multi_Agent
from ThesisPackage.RL.Decentralized_PPO.util import flatten_list, reverse_flatten_list_with_agent_list
from ThesisPackage.Wrappers.frame_stack import ParallelFrameStack
from ThesisPackage.Wrappers.vecWrapper import PettingZooVectorizationParallelWrapper
import torch
import time

def make_env():
    env = SoccerGame(20, 12, sequence_length=2, vocab_size=3, noise_team=1)
    env = ParallelFrameStack(env, 4)
    return env

envs = make_env()
agent = PPO_Multi_Agent(envs, device="cpu")

agent.agent.load_state_dict(torch.load("models/soccer_2vs2.pt"))

env = make_env()
obs, infos = env.reset()
wins = []
for j in range(1000):
    while True:
        obs = flatten_list([obs])
        obs = torch.tensor(obs, dtype=torch.float32)
        actions, _, _, _ = agent.agent.get_action_and_value(obs)
        actions = actions.detach().numpy()
        actions = reverse_flatten_list_with_agent_list(actions, env.agents)
        obs, reward, truncations, terminations, infos = env.step(actions[0])
        if any([truncations[agent] or terminations[agent] for agent in env.agents]):
            teams = [infos[agent]["team"] for agent in env.agents if "team" in infos[agent]]
            if len(teams) > 0:
                wins.append(teams[0])
            else:
                wins.append(0)
            obs, infos = env.reset()
            break

print("Team 1 wins: ", wins.count(1))
print("Team 2 wins: ", wins.count(-1))
print("Noise Team", 1)