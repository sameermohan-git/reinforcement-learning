import gym
from stable_baselines3 import DQN
import os
from maze_env import MazeEnvRandom5x5


env = MazeEnvRandom5x5()
# It will check your custom environment and output additional warnings if needed

models_dir = "../models/PPO"
gym.envs.register(
    id='maze-random-5x5-v0',
    entry_point='maze.envs:MazeEnvRandom5x5',
    #max_episode_steps=1,
    #nondeterministic=True,
)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env = gym.make('maze-random-5x5-v0')
env.reset()

model = DQN("MlpPolicy",
            env,
            verbose=1,
            train_freq=16,
            gradient_steps=8,
            gamma=0.9,
            exploration_fraction=0.1,
            exploration_final_eps=0.07,
            target_update_interval=600,
            learning_starts=1000,
            buffer_size=10000,
            batch_size=128,
            learning_rate=4e-3,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=2)

model.learn(total_timesteps=10000, log_interval=4)
model.save(f"{models_dir}/ppo_maze_random_5x5")

#del model # remove to demonstrate saving and loading

#model = model.load("ppo_maze_random_5x5")
obs = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    print('info=', info)
    #env.render(mode='rgb_array')
    if done:
     print("Goal reached!", "reward=", reward)
     break
