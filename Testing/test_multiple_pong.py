from ThesisPackage.Environments.multi_pong_language import PongEnv

def get_user_action(agent):
    action = input(f"Enter action (w=Up, s=Down, q=Quit) for agent {agent}: ")
    if action == 'w':
        return 1  # Move paddle up
    elif action == 's':
        return 2  # Move paddle down
    elif action == 'q':
        return 'quit'  # Quit game
    else:
        return 0  # No actionw

# Initialize the Pong environment
env = PongEnv(sequence_length=0)

# Start a new game
observation = env.reset()
done = False

while not done:
    env.render()  # Render the current state

    actions = {}
    for agent in env.agents:
        action = get_user_action(agent)
        actions[agent] = action

    if action == 'quit':
        break

    # Perform the action and get new state
    observation, reward, dones, info = env.step(actions)

    done = any([dones[agent] for agent in dones.keys()])

    print(observation, reward, dones)

env.close()