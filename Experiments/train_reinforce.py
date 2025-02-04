from ThesisPackage.Environments.multi_pong_sender_receiver_ball_onehot import PongEnvSenderReceiverBallOneHot
from ThesisPackage.RL.Reinforce.multi_reinforce import Multi_Reinforce

def make_env(seed, vocab_size, sequence_length, max_episode_steps):
    env = PongEnvSenderReceiverBallOneHot(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    return env

if __name__ == "__main__":
    device = "cpu"
    for sequence_lenght in [0, 1, 2, 3]:
        num_envs = 16
        seed = 1
        sequence_length = 3
        vocab_size = 3
        max_episode_steps = 2048
        total_timesteps = 600000000
        envs = [make_env(seed, vocab_size, sequence_length, max_episode_steps) for i in range(num_envs)]
        agent = Multi_Reinforce(envs, num_steps=256, device=device)
        agent.train(total_timesteps, exp_name="reinforce_pong", tensorboard_folder="Results/", seed=seed)