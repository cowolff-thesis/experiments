import torch
import torch.nn as nn
import torch.optim as optim
from ThesisPackage.Environments.multi_pong_language import PongEnv
from ThesisPackage.RL.Wrappers.normalizeObservation import NormalizeObservation
import numpy as np
from ThesisPackage.RL.Worldmodel_PPO.vae import VAE, vae_loss_function
from ThesisPackage.RL.Worldmodel_PPO.util import flatten_list, reverse_flatten_list_with_agent_list, normalize_batch_observations
from progressbar import progressbar

def flatten_list(data):
    flat_list = []
    for item in data:
        flat_list.extend(item.values())
    return flat_list

def reverse_flatten_list_with_agent_list(flat_list, agent_names):
    reversed_data = []
    for i in range(0, len(flat_list), len(agent_names)):
        # Creating a dictionary for each set of lists corresponding to the agent names
        entry = {agent_names[j]: flat_list[i + j] for j in range(len(agent_names))}
        reversed_data.append(entry)
    return reversed_data

def make_env(seed, vocab_size, sequence_length, max_episode_steps):
    env = PongEnv(width=20, height=20, vocab_size=vocab_size, sequence_length=sequence_length, max_episode_steps=max_episode_steps)
    return env

if __name__ == "__main__":
    num_envs = 64
    seed = 1
    sequence_length = 3
    num_agents = 2
    vocab_size = 3
    vae_stack = 4
    batch_size = 256
    vae_epochs = 3
    vae_kld_weight = 0.05
    max_episode_steps = 2048
    dataset_size = 2000
    device = "mps"
    learning_rate = 1e-3
    envs = [make_env(seed, vocab_size, sequence_length, max_episode_steps) for i in range(num_envs)]
    
    vae_obs = torch.zeros((dataset_size * vae_stack, num_agents * num_envs) + envs[0].observation_space(envs[0].agents[0]).shape).to(device)

    for sample_step in progressbar(range(0, dataset_size), redirect_stdout=True):
        next_obs = []
        current_rewards = []
        current_dones = []
        current_trunacted = []
        info = []
        for current_env in envs:
            action = {agent: current_env.action_space(agent).sample() for agent in current_env.agents}
            new_obs, reward, terminated, truncated, info = current_env.step(action)
            current_done = {key: terminated.get(key, False) or truncated.get(key, False) for key in set(terminated) | set(truncated)}
            next_obs.append(new_obs)
            current_rewards.append(reward)
            current_dones.append(current_done)
            current_trunacted.append(truncated)

        next_obs = np.array(flatten_list(next_obs))
        current_rewards = np.array(flatten_list(current_rewards))
        current_dones = np.array(flatten_list(current_dones))

        next_obs = np.array(next_obs)
        next_obs = torch.Tensor(next_obs).to(device)

        vae_obs[sample_step] = next_obs

        if any(current_dones):
            true_indices = np.nonzero(current_dones)[0]
            for index in true_indices:
                if index % num_agents == 0:
                    envs[int(index / num_agents)].reset()

    vae = VAE(envs[0], vae_stack, 32).to(device)
    vae_optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    low_values = envs[0].observation_space(envs[0].agents[0]).low
    high_values = envs[0].observation_space(envs[0].agents[0]).high
    vae_obs_normalized = normalize_batch_observations(vae_obs.cpu().numpy(), low_values, high_values)

    vae_obs_normalized = torch.Tensor(vae_obs_normalized).to(device)
    vae_obs_normalized = vae_obs_normalized.view(-1, vae_obs_normalized.shape[-1])
    vae_obs = vae_obs.view(-1, vae_obs.shape[-1])
    
    # Print batches with a value smaller than 0
    for i, batch in enumerate(vae_obs_normalized):
        if torch.any(batch < 0):
            print(batch, vae_obs[i])