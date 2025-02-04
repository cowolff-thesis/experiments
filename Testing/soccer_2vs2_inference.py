from ThesisPackage.Environments.soccer.soccer_env_multi import SoccerGame
from ThesisPackage.RL.Decentralized_PPO_soccer.multi_ppo import PPO_Multi_Agent
from ThesisPackage.RL.Decentralized_PPO.util import flatten_list, reverse_flatten_list_with_agent_list
from ThesisPackage.Wrappers.frame_stack import ParallelFrameStack
from ThesisPackage.Wrappers.vecWrapper import PettingZooVectorizationParallelWrapper
import torch
import time

def make_env():
    env = SoccerGame(12, 7, sequence_length=2, vocab_size=3, noise_team=-1)
    return env

envs = PettingZooVectorizationParallelWrapper(make_env, 1)
agent = PPO_Multi_Agent(envs, device="cpu")

agent.agent.load_state_dict(torch.load("models/soccer_2vs2.pt"))

env = make_env()
obs, infos = env.reset()
env.render()
for i in range(10240):
    obs = flatten_list([obs])
    env.render()
    time.sleep(0.3)
    obs = torch.tensor(obs, dtype=torch.float32)
    actions, _, _, _ = agent.agent.get_action_and_value(obs)
    actions = actions.detach().numpy()
    actions = reverse_flatten_list_with_agent_list(actions, env.agents)
    obs, reward, truncations, terminations, infos = env.step(actions[0])
    if any([truncations[agent] or terminations[agent] for agent in env.agents]):
        print("Episode done ", env.timestep)
        env.reset()