from ThesisPackage.Environments.soccer.soccer_env import SoccerGame
from ThesisPackage.RL.Decentralized_PPO.multi_ppo import PPO_Multi_Agent
from ThesisPackage.RL.Decentralized_PPO.util import flatten_list, reverse_flatten_list_with_agent_list
from ThesisPackage.Wrappers.frame_stack import ParallelFrameStack
from ThesisPackage.Wrappers.vecWrapper import PettingZooVectorizationParallelWrapper
import torch
import time

def make_env():
    env = SoccerGame(9, 5, sequence_length=2)
    return env

if __name__ == "__main__":
    num_envs = 32
    seed = 1
    total_timesteps = 2000000
    envs = PettingZooVectorizationParallelWrapper(make_env, num_envs)
    # envs = [make_env(seed, vocab_size, sequence_length, max_episode_steps) for i in range(num_envs)]
    agent = PPO_Multi_Agent(envs, device="cpu", normalize_obs=True, target_kl=0.01)
    agent.train(total_timesteps, tensorboard_folder="soccerResults", exp_name="debugging")

    # Inference
    env = make_env()
    obs, infos = env.reset()
    for i in range(1024):
        obs = flatten_list([obs])
        env.render()
        time.sleep(0.3)
        obs = torch.tensor(obs, dtype=torch.float32)
        actions, _, _, _ = agent.agent.get_action_and_value(obs)
        actions = actions.detach().numpy()
        actions = reverse_flatten_list_with_agent_list(actions, env.agents)
        obs, reward, truncations, terminations, infos = env.step(actions[0])
        if any([truncations[agent] or terminations[agent] for agent in env.agents]):
            env.reset()