from ThesisPackage.Environments.pong.multi_pong_sender_receiver_ball_onehot import PongEnvSenderReceiverBallOneHot
from ThesisPackage.RL.Centralized_PPO.multi_ppo import PPO_Multi_Agent_Centralized

def make_env(seed, vocab_size, sequence_length, max_episode_steps):
    env = PongEnvSenderReceiverBallOneHot(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    return env

if __name__ == "__main__":
    num_envs = 64
    seed = 1
    for sequence_length in [2, 3, 1, 0]:
        vocab_size = 3
        max_episode_steps = 2048
        total_timesteps = 3000000000
        envs = [make_env(seed, vocab_size, sequence_length, max_episode_steps) for i in range(num_envs)]
        agent = PPO_Multi_Agent_Centralized(envs, device="cpu")
        agent.train(total_timesteps, exp_name=f"multi_pong_{sequence_length}", tensorboard_folder="Training_Long", learning_rate=0.001) 
        agent.save(f"models/final_model_multi_pong_{sequence_length}.pt")