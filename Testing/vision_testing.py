from ThesisPackage.Environments.multi_pong_language_vision import PongEnv
import matplotlib.pyplot as plt

env = PongEnv()
for i in range(5):
    obs, reward, terminated, truncated, infos = env.step({agent: env.action_space(env.agents[0]).sample() for agent in env.agents})
print(env.observation_space(env.agents[0]))
print(obs[env.agents[0]]["vision"].shape)

vision = obs[env.agents[1]]["vision"]
plt.imshow(vision)
plt.savefig('Testing/image.png')
plt.show()