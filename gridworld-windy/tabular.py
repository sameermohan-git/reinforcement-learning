import numpy as np

# Function to select an action based off of Q vals
# with epsilon greedy exploration
# Takes table entry for Q(s, *) and epsilon
# Return action index
def epsilon_greedy_action(q_table_entry, epsilon):
    num_acts = len(q_table_entry)
    probs = np.zeros((num_acts))
    # Choose the greedy action
    ind = np.argmax(q_table_entry)
    # Remove epsilon from the greedy action probability
    probs[ind] = 1.0 - epsilon
    # Distribute epsilon among all actions
    probs += epsilon/num_acts
    # Choose our action
    return np.random.choice(num_acts, p=probs)

# Tabular Sarsa implementation
def sarsa(env, step_size=0.5, epsilon=0.1, gamma=1.0, num_eps=100):

    # Initialize rewards and our q_table. 
    # The dimensions of this table are: number of columns x number of rows x number of actions  
    q_table = np.zeros((*env.observation_space.nvec, env.action_space.n))
    reward_list = []
    
    # Loop for each episode:
    for ep in range(num_eps):
        # Reset everything before starting a new episode. Initialize S
        state = env.reset()
        done = False
        reward_list.append(0)

        # Select an epsilon greedy action. (Choose A from S using policy derived from Q (e.g., epsilon-greedy))
        action = epsilon_greedy_action(q_table[state[0], state[1],:], epsilon)

        #Loop for each step of episode
        while not done:
            # Get our new state. (Take action A, observe R, S_prime)
            state_prime, rew, done, info = env.step(action)

            # Choose the action of our next state (Choose A_prime from S_prime using policy derived from Q
            # (e.g., epsilon-greedy))
            action_prime = epsilon_greedy_action(q_table[state_prime[0], state_prime[1],:], epsilon)

            # Get the relevant Q values for our action in this state and the next
            q = q_table[state[0], state[1], action]

            q_prime = q_table[state_prime[0], state_prime[1], action_prime]

            # List of acronyms and useful information
            # Q(S,A) -> q_table[state[0],state[1], action], which is also q
            # alpha -> step_size
            # R -> rew
            # Q(s',a') -> q_table[state_prime[0], state_prime[1], action_prime]
            # End of list of acronyms and useful information

            # Task 1.1 - Update the table according to SARSA
            # INSERT CODE HERE
            # Q(s,a) = Q(s_a) + alpha * (r(s,a) + gamma * Q(s',a') - Q(s,a))
            q_table[state[0], state[1], action] = q + step_size * (rew + gamma * q_prime - q)

            # Update our current state and action
            state = state_prime
            action = action_prime
            reward_list[ep] += rew

    return reward_list, q_table

# Tabular Q-learning implementation
def q_learning(env, step_size=0.5, epsilon=0.1, gamma=0.8, num_eps=100):
    # Initialize rewards and our q_table
    q_table = np.zeros((*env.observation_space.nvec, env.action_space.n))
    reward_list = []

    # Loop for each episode:
    for ep in range(num_eps):
        # Reset everything before starting a new episode
        state = env.reset()
        done = False
        reward_list.append(0)
        
        # Loop for each step of episode:
        while not done:
            # Select an epsilon greedy action (Choose A from S using policy derived from Q (e.g., epsilon-greedy))
            action = epsilon_greedy_action(q_table[state[0], state[1],:], epsilon)

            # Progress the environment (Take action A, observe R, S_prime)
            state_prime, rew, done, info = env.step(action)

            # Get the q_values we need
            q = q_table[state[0], state[1], action]


            # Task 2.1 - Select the best action, which is the one with the maximum Q values
            # (This is the max_a Q(S',a) in the equation) of q_all_actions
            q_all_actions = q_table[state_prime[0], state_prime[1], :]
            # INSERT CODE HERE
            q_prime = max(q_all_actions)

            # List of acronyms and useful information
            # Q(S,A) -> q_table[state[0],state[1], action], which is also q
            # alpha -> step_size
            # R -> rew
            # Q(s',a') -> q_table[state_prime[0], state_prime[1], action_prime]
            # End of list of acronyms and useful information

            # Task 2.2 - Update the table according to Q-learning
            # INSERT CODE HERE
            # Q(s,a) = Q(s_a) + alpha * (r(s,a) + gamma * max(Q(s',*)) - Q(s,a))
            q_table[state[0], state[1], action] = q + step_size * (rew + gamma * q_prime - q)

            # Update state and rewards
            state = state_prime
            reward_list[ep] += rew

    return reward_list, q_table
