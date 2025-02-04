def player_position_relative(player_position, goal_position, field_height, team=1.0):
    # player_position is a tuple (x_player, y_player)
    # goal_position is a tuple (x_goal, y_goal) which is the center of the goal
    # field_width is the width of the goal line from one end to the other

    if team not in [-1, 1]:
        raise ValueError("team must be either -1 or 1")
    
    # Calculate the midpoint of the goal line as the new origin
    x = (player_position[0] - goal_position) * team
    y = (player_position[1] - (field_height / 2)) * team
    
    return (x, y)

def get_direction_relative(direction, team=1.0):
    # direction is a tuple (dx, dy)
    # team is either -1 or 1
    return (direction[0] * team, direction[1] * team)

# Example usage:
player_pos = (0, 2.5)  # Absolute position of the player
goal_pos = 10     # Position of the center of the goal
field_height = 5      # Standard goal width in meters

relative_position = player_position_relative(player_pos, goal_pos, field_height, team=1)
print("Player's position relative to the goal line:", relative_position)


player_pos = (10, 2.5)
goal_pos = 0

relative_position = player_position_relative(player_pos, goal_pos, field_height, team=-1)
print("Player's position relative to the goal line:", relative_position)