# from ThesisPackage.Environments.pong.multi_pong_language_continuous_vectorized import PongEnv
from ThesisPackage.Environments.pong.multi_pong_language_continuous_vectorized_onehot import PongEnv
from ThesisPackage.Environments.pong.multi_pong_language_continuous import PongEnv as ContPongEnv
from ThesisPackage.RL.Decentralized_PPO_vectorized.multi_ppo import PPO_Multi_Agent
from ThesisPackage.RL.Decentralized_PPO_vectorized.util import *
import torch
import numpy as np
import time

if __name__ == "__main__":
    num_envs = 1
    seed = 1
    total_timesteps = 2000000000
    env = PongEnv(num_envs=1, sequence_length=2, height=20, width=10, vocab_size=3)
    # env = ContPongEnv(sequence_length=2, height=20, width=20, vocab_size=3)
    # envs = [make_env() for i in range(num_envs)]
    agent = PPO_Multi_Agent(env, device="cpu", normalize_obs=False)
    agent.agent.load_state_dict(torch.load("/Users/cowolff/Documents/GitHub/ma.pong_rl/models/checkpoints/checkpoint_1463.pt"))

    next_obs, info = env.reset()
    lengths = []
    for i in range(200):
        while True:
            next_obs = torch.tensor(concatenate_agent_observations(next_obs), dtype=torch.float32)
            # time.sleep(0.2)
            actions, _, _, _ = agent.agent.get_action_and_value(next_obs)
            actions = split_agent_actions(actions.cpu().numpy(), agent.agents)
            # print(actions)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            terminated = terminated[list(terminated.keys())[0]][0]
            truncated = truncated[list(truncated.keys())[0]][0]

            if truncated or terminated:
                cur_reward = rewards["paddle_1"][0]
                lengths.append(env.timestep[0])
                next_obs, info = env.reset(index=0)
                break
            # try:
            #     env.render()
            # except:
            #     print(env.balls)
            #     exit()

    print(np.mean(lengths))