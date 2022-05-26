from pickletools import UP_TO_NEWLINE
from stat import UF_OPAQUE
import numpy as np
import gym
import matplotlib.pyplot as plt
import time
np.random.seed(4)

# Defining the actions
# For this environment, we will do math operations, so it is better to define the actions as numpy
UP = np.array([0,1])
DOWN = np.array([0,-1])
RIGHT = np.array([1,0])
LEFT = np.array([-1,0])


ACTIONS  = [ UP, DOWN, RIGHT, LEFT]
ACTION_TO_TEXT={tuple(UP):'UP', tuple(DOWN):'DOWN', tuple(LEFT):'LEFT', tuple(RIGHT):'RIGHT'}

class StochWindyGridWorldEnv(gym.Env):
    
    def __init__(self, GRID_HEIGHT=7, GRID_WIDTH=10,
                 WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0], 
                 START_CELL = np.array([0, 3]), 
                 GOAL_CELL = np.array([7, 4]),
                 REWARD = -1, 
                 RANGE_RANDOM_WIND=2):
        self.grid_height = GRID_HEIGHT
        self.grid_width = GRID_WIDTH
        self.grid_dimensions = (self.grid_height, self.grid_width)
        self.wind = np.array(WIND)
        self.start_cell = START_CELL
        self.goal_cell = GOAL_CELL        
        self.reward = REWARD
        self.range_random_wind = RANGE_RANDOM_WIND
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.MultiDiscrete((self.grid_width, self.grid_height))
        self.state = self.start_cell
        self.max_runs = 1000
        self.n_runs = 0

    def reset(self):
        self.state = self.start_cell
        self.n_runs = 0
        return self.state

    # Get the observation based on our current state
    # This function is simple here, but may be more complex depending on
    # the task
    # Returns:
    #   obs: the observation of the current state
    def _get_obs(self):
        return self.state

    # This function returns the next state ocurring in response to an action in the previous
    # state. 
    def transition(self, action, state): 
        # First we add the action and wind
        next_state = state + action
        next_state[1] += self.wind[state[0]]
        # Next we clip to make sure we are in the grid
        next_state[0] =  min(max(next_state[0], 0),self.grid_width-1)
        next_state[1] =  min(max(next_state[1], 0),self.grid_height-1)
        return next_state

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
        self.n_runs = self.n_runs + 1
        act = ACTIONS[action]
        # Get our current state so we can calculate the reward later
        state = self._get_obs()

        # We have no additional information to pass back now
        info = {}

        # Get the entry for this state-action pair in our transition table
        next_state = self.transition(act, state)

        done = True if np.array_equal(next_state,self.goal_cell) or (self.n_runs > self.max_runs) else False 

        reward = -1 

        # We make sure to update our current state
        self.state = next_state
        return self._get_obs(), reward, done, info 

    def render(self, draw_policy = False, draw_state = True):
        plt.ion()
        plt.clf()
        plt.xlim(0, self.grid_width)
        plt.ylim(0, self.grid_height)
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                #print("{:12.4f}".format(v[i, j]), end=' ')
                if(draw_policy):
                    a = ACTIONS[np.random.randint(0,3)]
                    plt.arrow(x+0.5,y+0.5,a[1]*0.3,-a[0]*0.3,head_width=0.05, head_length=0.1, fc='k', ec='k')
           #print('\n')
        if(draw_state):
            plt.plot(self.goal_cell[0]+0.5, self.goal_cell[1]+0.5, 'b*', markersize=12)
            plt.plot(self.state[0]+0.5, self.state[1]+0.5, 'r.', markersize=12)

        plt.gca().set_xticks(np.arange(0, self.grid_width, 1))
        plt.gca().set_yticks(np.arange(0, self.grid_height, 1))
        plt.grid(True,fillstyle = 'full')
        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left = False,
        right = False,
        labelbottom=False,
        labelleft = False) # labels along the bottom edge are off
        for i in range(0, len(self.wind)):
            plt.text(0.4 + i, -0.5, self.wind[i], fontsize=16)
        plt.draw()
        plt.pause(0.0001)



