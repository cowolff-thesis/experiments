from ThesisPackage.Environments.multi_pong_sender_receiver import PongEnvSenderReceiver
from ThesisPackage.RL.Decentralized_PPO.multi_ppo import PPO_Multi_Agent

def make_env(seed, vocab_size, sequence_length, max_episode_steps):
    env = PongEnvSenderReceiver(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps, self_play=True, receiver="paddle_2", mute_method="zero")
    return env

if __name__ == "__main__":
    i = 4
    num_envs = 64
    seed = 1
    sequence_length = i
    vocab_size = 3
    max_episode_steps = 2048
    total_timesteps = 15000000
    envs = [make_env(seed, vocab_size, sequence_length, max_episode_steps) for i in range(num_envs)]
    agent = PPO_Multi_Agent(envs)
    agent.train(total_timesteps, exp_name="multi_pong_sender_receiver")
    agent.save(f"models/multi_pong_test_sender_receiver_{i}")