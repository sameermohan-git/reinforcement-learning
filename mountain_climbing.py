import gym
import numpy as np

np.random.seed(4)

# Define our actions
CLIMB=0
PREPARE=1
ACTIONS=[CLIMB,PREPARE]

# Define our states
BASE_CAMP=0
BASE_CAMP_PREPARED=1
GLACIER=2
GLACIER_PREPARED=3
ICE_WALL=4
ICE_WALL_PREPARED=5
PEAK=6

# We define a mapping of actions and states for readability later
ACTION_TO_TEXT={CLIMB:'CLIMB', PREPARE:'PREPARE'}
STATE_TO_TEXT={BASE_CAMP:'BASE_CAMP', BASE_CAMP_PREPARED: 'BASE_CAMP_PREPARED', GLACIER: 'GLACIER', GLACIER_PREPARED: 'GLACIER_PREPARED', \
               ICE_WALL: 'ICE_WALL', ICE_WALL_PREPARED: 'ICE_WALL_PREPARED', PEAK: 'PEAK'}

# An OpenAI Gym code skeleton of the mountain climbing MDP for A2
# Developed for MMAI-845
class mountainClimbing(gym.Env):
    def __init__(self,
            ):

        # We set the number of states internally
        self.num_states = 7

        # We must set the size of our observations and actions so an agent
        # can be created for the environment
        self.observation_space = gym.spaces.Discrete(self.num_states)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        
        # Task A.1
        # INSERT CODE HERE
        # Fill in the correct start state to reset into
        self.init_state = None 

        # Create an empty table to hold our transition probabilities
        self.transition_table = {}
        
        # Task A.2
        # INSERT CODE HERE
        # Fill out the correct values for the probabilities for each state (Switch the None values to the right values)
        # Remember that each probability must add to 1
        # All of the entries for the table must be filled
        self.transition_table[BASE_CAMP] = {
                CLIMB: {GLACIER: None, BASE_CAMP: None},
                PREPARE: {BASE_CAMP_PREPARED: None}, 
                }

        self.transition_table[BASE_CAMP_PREPARED] = {
                CLIMB: {GLACIER: None},
                }

        self.transition_table[GLACIER] = {
                CLIMB: {ICE_WALL: None, GLACIER: None},
                PREPARE: {GLACIER_PREPARED: None}, 
                }

        self.transition_table[GLACIER_PREPARED] = {
                CLIMB: {ICE_WALL: None},
                }

        self.transition_table[ICE_WALL] = {
                CLIMB: {PEAK: None, ICE_WALL: None},
                PREPARE: {ICE_WALL_PREPARED: None}, 
                }

        self.transition_table[ICE_WALL_PREPARED] = {
                CLIMB: {PEAK: None},
                }

        # We define a fixed reward table based on the possible transitions
        # It is possible to calculate this purely on the transition without
        # predefining this table as well
        # Task A.3
        # INSERT CODE HERE
        # Fill out this table with the correct rewards from the MDP for every possible transition (Switch the None values to the right values)
        self.reward_table = {
                (BASE_CAMP, CLIMB, BASE_CAMP): None,
                (BASE_CAMP, CLIMB, GLACIER): None,
                (BASE_CAMP, PREPARE, BASE_CAMP_PREPARED): None,
                (BASE_CAMP_PREPARED, CLIMB, GLACIER): None,
                (GLACIER, CLIMB, GLACIER): None,
                (GLACIER, CLIMB, ICE_WALL): None,
                (GLACIER, PREPARE, GLACIER_PREPARED): None,
                (GLACIER_PREPARED, CLIMB, ICE_WALL): None,
                (ICE_WALL, CLIMB, ICE_WALL): None,
                (ICE_WALL, CLIMB, PEAK): None,
                (ICE_WALL, PREPARE, ICE_WALL_PREPARED): None,
                (ICE_WALL_PREPARED, CLIMB, PEAK): None,
                }

    # Place us in the initial state
    # This does not need to be deterministic
    # Returns:
    #   obs: an observation of our current state after the reset
    def reset(self):
        self.state = self.init_state
        return self._get_obs()

    # Get the observation based on our current state
    # This function is simple here, but may be more complex depending on
    # the task
    # Returns:
    #   obs: the observation of the current state
    def _get_obs(self):
        return self.state

    # This function progresses the environment one timestep given the current
    # state and the action. This is where the dynamics are applied.
    # Inputs:
    #   action: The desired action to apply to the environment
    # Returns:
    #   obs: observation of the new state
    #   reward: the reward received for the transition
    #   done: a variable indicating whether we have terminated or not
    #   info: a dictionary data structure containing additional information
    #         about the environment we may want to track
    def step(self, action):
        # Get our current state so we can calculate the reward later
        state = self._get_obs()

        # We have no additional information to pass back now
        info = {}

        # Get the entry for this state-action pair in our transition table
        transition_entry = self.transition_table[state][action]

        # Since we have a low number of fixed states, we can process the entry
        # into states and probabilities easily directly. With a more complex
        # table, we can iterate over the transitions
        possible_states = list(transition_entry.keys())
        state_probabilities = list(transition_entry.values())

        # We use the numpy library to select the next state according to our
        # probability distribution
        next_state = np.random.choice(possible_states, p=state_probabilities)
        
        # Task B.1
        # INSERT CODE HERE
        # Enter the correct state on which we terminate here
        done = True if next_state==None else False 

        # Task B.2
        # INSERT CODE HERE
        # Call the reward function for the transition correctly
        # Hint: use the self. prefix to call functions in the object we are in
        reward = None 

        # We make sure to update our current state
        self.state = next_state
        return self._get_obs(), reward, done, info 

    # This function calculates the reward for a given_transition
    # Inputs:
    #   state: The current state
    #   action: The action applied
    #   next_state: The next state we enter
    # Returns:
    #   reward: The given reward for the transitions
    def _get_reward(self, state, action, next_state):
        index = (state, action, next_state)
        return self.reward_table[index]

# This class will output an action for each state according to the specificatin given in the environment
class policy():
    # Nothing to do for initialization in this case
    def __init__(self):
        pass

    # This function take the state and returns the correct action
    # Inputs:
    #   state: an integer representing the state
    # Output:
    #   action: an integer representing the action specified in the assignmnet text
    def __call__(self, state):
        action = None
        # Q2
        # INSERT CODE HERE
        # Fill out the correct state in the empty lists in the if statements, and attach the correct actions
        # You may add more if/elif statements if you wish, but only two are necessary 
        if state in []:
            action = None
        elif state in []:
            action = None

        return action


if __name__=='__main__':
    # NOTE: Set the flag below to 0 if you want to debug only the environment code, and 1 if you want to check on the policy
    # Keep in mind that if the value is set to zero, only the climb action is taken, so there may still be an error in the 
    # prepare actions or states which this would not reveal
    debug_flag=0

    env = mountainClimbing()
    # We need to reset the environment to initialize it
    state = env.reset()
    if debug_flag==0: 
        done = False
        while not done:
            # Always select the climb action in this debug mode 
            action = CLIMB 
            # We apply our action and observe the outcome
            next_state, reward, done, info = env.step(action)
            # We print the transition and reward for visualization
            print("State: {}, Action: {}, State': {}, Reward: {}".format(\
                    STATE_TO_TEXT[state], ACTION_TO_TEXT[action], STATE_TO_TEXT[next_state], reward))
            state = next_state

    elif debug_flag==1:
        pol = policy()
        done = False
        while not done:
            # Call our policy to get the action for this state
            action =  pol(state) 
            # We apply our action and observe the outcome
            next_state, reward, done, info = env.step(action)
            # We print the transition and reward for visualization
            print("State: {}, Action: {}, State': {}, Reward: {}".format(\
                    STATE_TO_TEXT[state], ACTION_TO_TEXT[action], STATE_TO_TEXT[next_state], reward))
            state = next_state



