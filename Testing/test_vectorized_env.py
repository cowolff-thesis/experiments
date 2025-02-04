from ThesisPackage.Environments.pong.multi_pong_language_continuous_vectorized import PongEnv
import numpy as np
import time

env = PongEnv(num_envs=4, sequence_length=2)
obs, info = env.reset()
env.render()
for i in range(1024):
    time.sleep(0.3)
    actions = {agent: np.array([np.concatenate(env.action_space(agent).sample(), axis=0) for n in range(env.num_envs)]) for agent in obs.keys()}
    obs, rewards, terminated, truncated, info = env.step(actions)
    print("\n")
    
    terminated = terminated[list(terminated.keys())[0]]
    for i, current_term in enumerate(terminated):
        if current_term:
            print(f"Env {i} terminated")
            cur_reward = rewards["paddle_1"][i]
            print(f"Reward: {cur_reward}")
            env.reset(index=i)
    try:
        env.render()
    except:
        print(env.balls)
        exit()