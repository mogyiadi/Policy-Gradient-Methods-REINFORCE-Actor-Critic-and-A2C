import numpy as np
import tensorflow as tf
import gymnasium as gym
import csv

from REINFORCE_model import ReinforceModel
from AC_model import ActorCritic
from A2C_model import AdvantageActorCritic
from train_REINFORCE import run_episodes as run_reinforce
from train_AC_model import run_episodes as run_ac
from train_A2C_model import run_episodes as run_a2c

seeds = [0, 1, 2, 3, 4]
n_episodes = 800
num_hidden_units = 128

configs = [
    ('REINFORCE', ReinforceModel,        run_reinforce, 0.0025),
]

env = gym.make('CartPole-v1')
num_actions = int(env.action_space.n)

with open('results_pg.csv', 'a', newline='') as f:
    writer = csv.writer(f)

    for config_name, ModelClass, run_fn, lr in configs:
        for seed in seeds:
            print(f"\nTraining {config_name}, seed {seed}...")
            tf.random.set_seed(seed)
            np.random.seed(seed)

            model = ModelClass(num_actions, num_hidden_units)
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
            scores = run_fn(env, model, opt, n_episodes)

            # CartPole reward = 1 per step, so episode score = steps taken
            # cumsum gives the total env steps at the end of each episode
            cumsteps = np.cumsum(scores).astype(int)
            for step, score in zip(cumsteps, scores):
                writer.writerow([config_name, seed, step, score])

print("\nSaved results_pg.csv")
