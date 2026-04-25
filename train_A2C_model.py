from A2C_model import AdvantageActorCritic
import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def train_step(initial_state, env, model, optimizer, gamma=0.99):
    state_tensor = tf.convert_to_tensor([initial_state], dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Get action logits and V(s) from current state
        action_probs, value = model(state_tensor)

        # V(s) is shape (1, 1) — squeeze to scalar
        current_value = value[0, 0]

        # Sample an action from the action probabilities
        action_tensor = tf.random.categorical(action_probs, 1)[0, 0]
        action = int(action_tensor.numpy())

        # Take a step in the actual environment
        next_state, reward, done, truncated, _ = env.step(action)
        terminal = done or truncated

        # Get V(s') for the next state
        next_state_tensor = tf.convert_to_tensor([next_state], dtype=tf.float32)
        _, next_value = model(next_state_tensor)
        next_value = next_value[0, 0]

        # TD target: r + γV(s'), zero if terminal
        td_target = reward + gamma * next_value * (1.0 - float(terminal))

        # Advantage: A(s,a) = td_target - V(s)
        # stop_gradient on td_target so the bootstrap target is treated as fixed
        advantage = tf.stop_gradient(td_target) - current_value

        # Critic loss: MSE between V(s) and TD target
        critic_loss = tf.math.square(advantage)

        # Actor loss: -log π(a|s) * A(s,a), stop gradients through advantage
        negative_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=[action], logits=action_probs
        )
        actor_loss = negative_log_prob * tf.stop_gradient(advantage)

        total_loss = critic_loss + actor_loss

    # Update model weights
    gradients = tape.gradient(total_loss, model.trainable_variables)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

    return next_state, reward, terminal


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

model = AdvantageActorCritic(int(num_actions), num_hidden_units)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

rewards_list = run_episodes(env, model, optimizer, 1000)

plt.plot(np.arange(len(rewards_list)), rewards_list)
plt.show()
