from ThesisPackage.Environments.multi_pong_language import PongEnv
from ThesisPackage.RL.Worldmodel_PPO.multi_ppo import PPO_World_Model
from ThesisPackage.RL.Wrappers.normalizeObservation import NormalizeObservation

def make_env(seed, vocab_size, sequence_length, max_episode_steps):
    env = PongEnv(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    env = NormalizeObservation(env)
    return env

if __name__ == "__main__":
    num_envs = 64
    seed = 1
    sequence_length = 3
    vocab_size = 3
    max_episode_steps = 2048
    total_timesteps = 800000000
    envs = [make_env(seed, vocab_size, sequence_length, max_episode_steps) for i in range(num_envs)]
    agent = PPO_World_Model(envs, vae_latent_dim=20, vae_beta=10, total_vae_updates=20, vae_stack=4)
    agent.train(total_timesteps, vae_path="models/vae.pth")