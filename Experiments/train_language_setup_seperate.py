from ThesisPackage.Environments.multi_pong_language import PongEnv
from ThesisPackage.RL.Seperated_PPO.multi_ppo import PPO_Separate_Multi_Agent
from ThesisPackage.Environments.multi_pong_sender_receiver import PongEnvSenderReceiver
from ThesisPackage.RL.Decentralized_PPO.multi_ppo import PPO_Multi_Agent

def make_env(seed, vocab_size, sequence_length, max_episode_steps):
    env = PongEnv(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    return env

def make_env_sender_receiver(seed, vocab_size, sequence_length, max_episode_steps, mute_method="zero"):
    env = PongEnvSenderReceiver(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps, self_play=True, receiver="paddle_2", mute_method=mute_method)
    return env

if __name__ == "__main__":
    num_envs = 64
    seed = 1
    sequence_length = 2
    vocab_size = 3
    max_episode_steps = 2048
    total_timesteps = 20000000

    envs = [make_env(seed, vocab_size, sequence_length, max_episode_steps) for i in range(num_envs)]
    agent = PPO_Separate_Multi_Agent(envs)
    agent.train(total_timesteps, exp_name="seperate_network", tensorboard_folder="Results")
    agent.save("models/seperate_network")