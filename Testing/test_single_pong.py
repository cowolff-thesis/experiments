from single_pong_multiple_balls import PongEnv

def get_user_action():
    action = input("Enter action (w=Up, s=Down, q=Quit): ")
    if action == 'w':
        return 1  # Move paddle up
    elif action == 's':
        return 2  # Move paddle down
    elif action == 'q':
        return 'quit'  # Quit game
    else:
        return 0  # No actionw

# Initialize the Pong environment
env = PongEnv()

# Start a new game
observation = env.reset()
done = False

while not done:
    env.render()  # Render the current state
    action = get_user_action()

    if action == 'quit':
        break

    # Perform the action and get new state
    observation, reward, done, info = env.step(action)

env.close()