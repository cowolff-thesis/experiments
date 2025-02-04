from ThesisPackage.Environments.multi_pong_language import PongEnv
from ThesisPackage.RL.Reinforce.multi_reinforce import Multi_Reinforce
from gym_peetingzoo_wrapper import GymToPettingZooWrapper

def make_env(seed, vocab_size, sequence_length, max_episode_steps):
    env = PongEnv(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    return env

def make_classic_env():
    import gym
    env = gym.make("CartPole-v1")
    env = GymToPettingZooWrapper(env)
    return env

if __name__ == "__main__":
    num_envs = 1
    seed = 1
    sequence_length = 3
    vocab_size = 3
    max_episode_steps = 2048
    total_timesteps = 20000000
    for num_steps in [256, 512, 1024, 2048, 4096]:
        for lr in [1e-3, 1e-4, 1e-5]:
            envs = [make_env(seed, vocab_size, sequence_length, max_episode_steps) for i in range(num_envs)]
            # envs = [make_classic_env() for i in range(num_envs)]
            agent = Multi_Reinforce(envs, num_steps=num_steps)
            agent.train(total_timesteps, exp_name=f"gym_test_{num_steps}_{lr}", tensorboard_folder="Reinforce", learning_rate=lr)