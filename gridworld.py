import numpy as np
import gym

# Defining the special states with higher rewards
GOAL_STATES = [(1,0), (3, 0)]

# Defining the transition from the special states
GOAL_NEXT_STATES = {
        (1,0): (1,4),
        (3,0): (3,2),
        }

# Defining the special rewards
TRANSITION_REWARDS = {
        ((1,0), (1,4)): 10.0,
        ((3,0), (3,2)): 5.0
        }

# Defining the actions
ACTIONS  = [
        (1,0),  #EAST, increase column by 1
        (-1,0), #WEST, decrease column by 1
        (0,-1),  #NORTH, decrease row by 1
        (0,1)  #SOUTH, increase row by 1
         ]

# Utility function to print the grid in an ordered way
def print_grid(grid, dim=5):
    for j in range(dim):
        row_str = ''
        for i in range(dim):
            row_str += '\t{} '.format(np.round(grid[i,j], 2))
        print(row_str)


# Utility function to print the policy in an ordered way
def print_policy(policy, dim=5):
    pol_enum = {(1,0): 'E',
            (-1,0): 'W',
            (0,-1): 'N',
            (0,1): 'S',}
    for j in range(dim):
        row_str = ''
        for i in range(dim):
            row_str += '\t{} '.format(pol_enum[policy[(i,j)]])
        print(row_str)

# An OpenAI Gym environment for the gridworld described in A3
# Developed for MMAI-845
# Note the two functions which we allow our agent to access, transition and get_rew
# These functions allow us to use dynamic programming, as the agent has knowledge of the environment
# dynamics
# Also note that we don't actually set initial states or transitions, since we don't need to actually
# interact with the environment with a perfect (known) model
class gridworld(gym.Env):
    def __init__(self,
            grid_dim=5,
            ):

        self.grid_dim = grid_dim 
        # We must set the size of our observations and actions so an agent
        # can be created for the environment
        self.observation_space = gym.spaces.MultiDiscrete((self.grid_dim, self.grid_dim))
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

    # Place us in the initial state
    # This does not need to be deterministic
    # Returns:
    #   obs: an observation of our current state after the reset
    def reset(self):
        pass


    # This function returns the next state ocurring in response to an action in the previous
    # state. This is used to update the values in DP.
    def transition(self, state, action):        
        # First, we update the state if we are not in a goal state
        if state not in GOAL_STATES:
            # Apply the action
            next_state = (state[0] + action[0], state[1] + action[1]) 
            # We modify the value here to make sure we are still in the grid
            next_x = max(min(next_state[0], self.grid_dim - 1), 0)
            next_y = max(min(next_state[1], self.grid_dim - 1), 0)
            next_state = (next_x, next_y)
        # If we are in the goal state, we move to a fixed state regardless of the action
        else:
            next_state = GOAL_NEXT_STATES[state]

        return next_state

    def get_reward(self, state, action, next_state):
        # Our default reward is 0
        reward = 0
        # If we are in a goal state before acting, we change the reward based on the problem definition
        if state in GOAL_STATES:
            reward = TRANSITION_REWARDS[(state, next_state)]
        # If we hit a wall, our reward is -1
        if state == next_state:
            reward = -1
        return reward

    def step(self, action):
        pass

# We define a simple random policy for the assignment.
# This chooses each action with probability 0.25 for each state
class assignmentUniformPolicy():
    def __init__(self):
        pass

    # Return a list of the actions and probabilities
    def __call__(self, state):
        action_probabilities = {}
        for action in ACTIONS:
            action_probabilities[action] = 1.0 / len(ACTIONS)
        return action_probabilities 

# This function implements the policy evaluation algorithm
# It returns a table of the values for each state
def policy_evaluation(environment, tolerance, policy, gamma):
    # We create a list of zeros of dimensions nxn, where n is the dimensionality of the grid
    # Note that this could be randomly instantiated, as per the algorithm states
    value_table = np.zeros((environment.grid_dim, environment.grid_dim)) 

    # Set the delta high so we enter the loop initially
    delta = float('inf')

    # We loop until the difference in values is smaller than the tolerance we define
    while delta > tolerance:
        delta = 0 
        # Since our state is a grid, we loop through every combination of x and y position
        # First is the x position
        for i in range(environment.grid_dim):
            # Second is the y position
            for j in range(environment.grid_dim):
                # Store our value to check the delta
                value_old = value_table[i, j]
                state = (i,j)
                value = 0

                # Loop through every action to find the overall value
                for action in ACTIONS:

                    # Get the actions and its probabilities given the state
                    actions_probabilities = policy(state) 

                    # Get the probability of taking action a given state s
                    pi = actions_probabilities[action]

                    # Get the next state according to the selected action and given the current state
                    next_state = environment.transition(state, action)

                    # Get the value of the next state
                    V_next_state = value_table[next_state[0], next_state[1]]

                    # Get the reward of arriving in next state given the current state and the selected action
                    reward = environment.get_reward(state, action, next_state)                   

                    # Task 1.1
                    # INSERT CODE HERE
                    # Update the value of the current state according to equation 4.5 in the book.
                    # List of acronyms and useful information
                    # pi(a|s) -> pi
                    # p(s', r|s, a) = 1 (once you take the action, you always land at a specific next state because it is deterministic)
                    # vk(s') -> V_next_state
                    # r -> reward
                    # We are already doing the sum over the all possible actions for you!
                    # End of list of acronyms and useful information

                    # value = value + (insert rest of expression here)
                    value = value + pi * (reward + (gamma * V_next_state))

                # Update our value table
                np.put(value_table[i], j, value)

                # Task 1.2
                # INSERT CODE HERE
                # Define the stop condition so that the algorithm stops updating the policy once the biggest
                # difference between the current value and the old value is smaller than the tolerance variable.
                #delta = (insert expression here)
                delta = value_old - value

    return value_table

def value_iteration(environment, tolerance, gamma):
    # We create a list of zeros of dimensions nxn, where n is the dimensionality of the grid
    # Note that this could be randomly instantiated, as per the algorithm states
    value_table = np.zeros((environment.grid_dim, environment.grid_dim)) 

    # Set the delta high so we enter the loop initially
    delta = float('inf')

    # We loop until the difference in values is smaller than the tolerance we define
    while delta > tolerance:
        delta = 0 
        # Since our state is a grid, we loop through every combination of x and y position
        # First is the x position
        for i in range(environment.grid_dim):
            # Second is the y position
            for j in range(environment.grid_dim):

                # Get the value to check the delta
                value_old = value_table[i, j]
                state = (i,j)

                # Loop through every action
                action_values = {}
                for action in ACTIONS:
                    # Get the next state according to the selected action and given the current state
                    next_state = environment.transition(state, action)

                    # Get the value of the next state
                    V_next_state = value_table[next_state[0], next_state[1]]

                    # Get the reward of arriving in next state given the current state and the selected action
                    reward = environment.get_reward(state, action, next_state)

                    # Task 2.1
                    # INSERT CODE HERE
                    # Update the value of the current state according to equation 4.10 in the book.
                    # Right now we are only calculating the values for each action, when we finish this we will get the maximum value!
                    # action_values[action] = (insert expression here)
                    action_values[action] = reward + (gamma * V_next_state)
                
                # Task 2.2
                # INSERT CODE HERE
                # Select the highest valued state/action. Tip: action_values.values() returns a list of all values for all actions.
                # best_val =  (insert expression here)
                best_val = max(action_values.values())

                # Update the table
                np.put(value_table[i], j, best_val)

        # Task 2.3
        # INSERT CODE HERE
        # Define the stop condition so that the algorithm stops updating the value once the biggest
        # difference between the current value and the old value is smaller than the tolerance variable.
        #delta = (insert expression here)
        delta = best_val - value_old

    # Now that we have all state values, we can define the optimal policy
    policy = {}
    # Loop through the states to assign actions
    # Start with x position
    for i in range(environment.grid_dim):
        # Second is the y position
        for j in range(environment.grid_dim):
            state = (i,j)
            # Get the action values
            action_values = get_prob_weighted_action_vals(environment, state, value_table, gamma)
            # Select the best action
            best_act_ind = np.argmax(list(action_values.values()))
            # Save the best action
            policy[state] = list(action_values)[best_act_ind]

    return value_table, policy

# A helper function to get a list of values received for all of the actions in a state
def get_prob_weighted_action_vals(environment, state, value_table, gamma):
    values = {}
    # Loop through every action
    for action in ACTIONS:
        value = 0
        # Get the probability of a transition occurring
        next_state = environment.transition(state, action)
        prob = 1
        # For every transition calculate the probability weighted value
        # First get the reward
        reward = environment.get_reward(state, action, next_state)
        # Add the discounted next state value and multiple by the occurence probability
        value += prob*(reward + gamma*value_table[next_state[0], next_state[1]])
        values[action] = value
    return values

if __name__=='__main__':
    # Set to true to run policy evaluation, false otherwise
    run_policy_eval = True
    # Set to true to run value iteration, false otherwise
    run_value_iter = True

    # Create the environment and policy
    env = gridworld()
    policy = assignmentUniformPolicy()

    if run_policy_eval:
        print('Running Policy Evaluation')
        value_table = policy_evaluation(env, 1e-3, policy, 0.9)
        value_table = np.squeeze(value_table)
        print_grid(value_table)
        print('-----------------------------------------------------')
    if run_value_iter:
        print('Running Value Iteration')
        value_table, policy = value_iteration(env, 1e-3, 0.9)
        value_table = np.squeeze(value_table)
        print_grid(value_table)
        print('\nOptimal policy after computing the optimal values')
        print_policy(policy)
        print('-----------------------------------------------------')
