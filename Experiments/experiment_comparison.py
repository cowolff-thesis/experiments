from ThesisPackage.Environments.multi_pong_sender_receiver_ball_onehot import PongEnvSenderReceiverBallOneHot
from ThesisPackage.RL.Decentralized_PPO.multi_ppo import PPO_Multi_Agent
from ThesisPackage.Environments.multi_pong_language import PongEnv
from ThesisPackage.RL.Centralized_PPO.multi_ppo import PPO_Multi_Agent_Centralized

def make_env_one_hot(seed, vocab_size, sequence_length, max_episode_steps):
    env = PongEnvSenderReceiverBallOneHot(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    return env

def make_env(seed, vocab_size, sequence_length, max_episode_steps):
    env = PongEnv(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    return env

if __name__ == "__main__":
    num_envs = 64
    seed = 1
    sequence_length = [2, 1, 0]
    vocab_size = 3
    max_episode_steps = 2048
    total_timesteps = 600000000

    for length in sequence_length:

        # Experiment with decentralized critic and one-hot encoding
        envs = [make_env_one_hot(seed, vocab_size, length, max_episode_steps) for i in range(num_envs)]
        agent = PPO_Multi_Agent_Centralized(envs)
        agent.train(total_timesteps, exp_name=f"PPO_Centralized_OneHot_{length}", tensorboard_folder="Results")
        
        # Experiment with decentralized critic and one-hot encoding
        envs = [make_env_one_hot(seed, vocab_size, length, max_episode_steps) for i in range(num_envs)]
        agent = PPO_Multi_Agent(envs)
        agent.train(total_timesteps, exp_name=f"PPO_Decentralized_OneHot_{length}", tensorboard_folder="Results")