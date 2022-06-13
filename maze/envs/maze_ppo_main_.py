
import os
from time import time
import gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from maze_env import MazeEnvRandom5x5, MazeEnvRandom8x8, MazeEnvRandom10x10

env = MazeEnvRandom5x5()
#env = DummyVecEnv(lambda: env, n_envs=1)

models_dir = "../models/DQN"
'''
gym.envs.register(
    id='maze-random-5x5-v0',
    entry_point='maze.envs:MazeEnvRandom5x5',
    #max_episode_steps=1,
    #nondeterministic=True,
)
'''
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

#env = gym.make('maze-random-5x5-v0')
#env.reset()
'''
model = DQN("MlpPolicy",
            env,
            verbose=2,
            train_freq=16,
            gradient_steps=8,
            gamma=0.9,
            exploration_fraction=0.2,
            exploration_final_eps=0.07,
            target_update_interval=600,
            learning_starts=1000,
            buffer_size=10000,
            batch_size=128,
            learning_rate=2e-3,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=2,
            tensorboard_log="./logs")
'''
model = PPO("MlpPolicy",
            env,
            verbose=2,
            gamma=0.9,
            clip_range=0.2,
            batch_size=128,
            learning_rate=3e-3,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=2,
            tensorboard_log="./logs")
#Evaluate Agent before training
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), deterministic=True, n_eval_episodes=2)

#print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent
model.learn(total_timesteps=10000, log_interval=10)
model.save(f"{models_dir}/dqn_maze_random_5x5")

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), deterministic=True, n_eval_episodes=20)

#print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

#del model # remove to demonstrate saving and loading
#model = model.load("dqn_maze_random_5x5")

#Test trained agent
obs = env.reset()

n_steps = 20
for step in range(n_steps):
    action, _states = model.predict(obs, deterministic=True)
    print("Action: ", action)
    print("Step {}".format(step + 1))
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    print('info=', info)
    env.render(mode='rgb_array')
    if done:
     print("Goal reached!", "reward=", reward)
     break
