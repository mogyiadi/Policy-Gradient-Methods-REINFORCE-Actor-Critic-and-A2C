import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import the models
from REINFORCE_model import ReinforceModel
from AC_model import ActorCritic
from A2C_model import AdvantageActorCritic

# Import the training loops
from train_REINFORCE import run_episodes as run_reinforce
from train_AC_model import run_episodes as run_ac
from train_A2C_model import run_episodes as run_a2c


def smooth_curve(scalars, weight=0.85):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def main():
    env = gym.make('CartPole-v1')

    # Common hyperparameters
    seed = 69
    tf.random.set_seed(seed)
    np.random.seed(seed)
    num_actions = int(env.action_space.n)
    num_hidden_units = 128
    learning_rate = 0.0005
    n_episodes = 1000

    # REINFORCE
    print("Training REINFORCE")
    model_reinforce = ReinforceModel(num_actions, num_hidden_units)
    opt_reinforce = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    rewards_reinforce = run_reinforce(env, model_reinforce, opt_reinforce, n_episodes)
    steps_reinforce = np.cumsum(rewards_reinforce)

    # np.save('data_reinforce_rewards.npy', rewards_reinforce)
    # np.save('data_reinforce_steps.npy', steps_reinforce)

    # AC
    print("Training AC")
    model_ac = ActorCritic(num_actions, num_hidden_units)
    opt_ac = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    rewards_ac = run_ac(env, model_ac, opt_ac, n_episodes)
    steps_ac = np.cumsum(rewards_ac)

    # np.save('data_ac_rewards.npy', rewards_ac)
    # np.save('data_ac_steps.npy', steps_ac)

    # A2C
    print("Training A2C")
    model_a2c = AdvantageActorCritic(num_actions, num_hidden_units)
    opt_a2c = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    rewards_a2c = run_a2c(env, model_a2c, opt_a2c, n_episodes)
    steps_a2c = np.cumsum(rewards_a2c)

    # np.save('data_a2c_rewards.npy', rewards_a2c)
    # np.save('data_a2c_steps.npy', steps_a2c)

    print("Training complete")

    # PLOT
    plt.figure(figsize=(12, 7))

    plt.plot(steps_reinforce, smooth_curve(rewards_reinforce), label='REINFORCE', color='blue', alpha=0.8)
    plt.plot(steps_ac, smooth_curve(rewards_ac), label='Actor-Critic (AC)', color='orange', alpha=0.8)
    plt.plot(steps_a2c, smooth_curve(rewards_a2c), label='Advantage Actor-Critic (A2C)', color='green', alpha=0.8)

    plt.xlabel('Environment Steps')
    plt.ylabel('Episode Return')
    plt.title('Learning Curves: Policy Gradient Methods vs Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()