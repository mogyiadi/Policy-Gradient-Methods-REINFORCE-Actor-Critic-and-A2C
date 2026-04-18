from model import ActorCritic
import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm


def train_step(initial_state, env, model, optimizer, gamma=0.99):
    state_tensor = tf.convert_to_tensor([initial_state], dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Get action probs from current state
        action_probs, q_values = model(state_tensor)

        # Sample an action from the action probabilities
        action_tensor = tf.random.categorical(action_probs, 1)[0, 0]
        action = int(action_tensor.numpy())

        # Get the Q-value for the selected action
        current_q = q_values[0, action]

        # Take a step in the actual environment
        next_state, reward, done, truncated, _ = env.step(action)

        # Get action probs for the next state
        next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)
        next_action_probs, next_q_values = model(next_state_tensor)

        # Sample the potential next action the agent would take
        next_action_tensor = tf.random.categorical(next_action_probs, 1)[0, 0]
        next_action = int(next_action_tensor.numpy())

        # Get its q value
        next_q = next_q_values[0, next_action]

        # Calculate the temporal difference error
        target_q = reward + gamma * next_q * (1.0 - float(done or truncated))

        # MSE between current q value and target q value
        critic_loss = tf.math.square(target_q - current_q)

        # Calculating the actor loss
        negative_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[action], logits=action_probs)

        # Estimate the advantage
        # Need to stop the gradients flowing to the critic's weights
        actor_loss = negative_log_prob * tf.stop_gradient(current_q)

        total_loss = critic_loss + actor_loss

    # Update model weights
    gradients = tape.gradient(total_loss, model.trainable_variables)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

    return next_state, reward, done or truncated


def run_episodes(env, model, optimizer, n_episodes):
    rewards_list = []
    for _ in range(n_episodes):
        state, info = env.reset()

        done = False
        rewards = 0

        while not done:
            state, reward, done = train_step(state, env, model, optimizer)
            rewards += reward

        rewards_list.append(rewards)

    return rewards_list


env = gym.make('CartPole-v1')

seed = 69
tf.random.set_seed(seed)
np.random.seed(seed)

num_actions = env.action_space.n
num_hidden_units = 128

model = ActorCritic(int(num_actions), num_hidden_units)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

rewards_list = run_episodes(env, model, optimizer, 1000)

plt.plot(np.arange(len(rewards_list)), rewards_list)
plt.show()








