from ThesisPackage.Environments.pong.multi_pong_language import PongEnv
from ThesisPackage.RL.Decentralized_PPO.multi_ppo import PPO_Multi_Agent
from ThesisPackage.Wrappers.frame_stack import ParallelFrameStack
from ThesisPackage.Wrappers.vecWrapper import PettingZooVectorizationParallelWrapper

def make_env():
    sequence_length = 2
    vocab_size = 3
    max_episode_steps = 1024
    env = PongEnv(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    # env = ParallelFrameStack(env, 4)
    return env

if __name__ == "__main__":
    # Create Single Env and test observation_space with actual observation
    num_envs = 64
    seed = 1
    sequence_length = 2
    vocab_size = 3
    max_episode_steps = 512
    total_timesteps = 500000000
    envs = PettingZooVectorizationParallelWrapper(make_env, num_envs)
    # envs = [make_env(seed, vocab_size, sequence_length, max_episode_steps) for i in range(num_envs)]
    agent = PPO_Multi_Agent(envs, device="cpu", normalize_obs=False)
    agent.train(total_timesteps, tensorboard_folder="normalizeResults", exp_name="pong")

    agent.save("models/pong")