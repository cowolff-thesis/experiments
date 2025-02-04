from ThesisPackage.Environments.soccer.soccer_env import SoccerGame
import time

env = SoccerGame(25, 15)
env.reset()

while True:
    env.render()
    actions = {}
    for agent in env.agents:
        action = env.action_space(agent).sample()
        actions[agent] = action

    obs, reward, truncations, terminations, info = env.step(actions)
    time.sleep(0.1)
    if any([truncations[agent] or terminations[agent] for agent in env.agents]):
        break