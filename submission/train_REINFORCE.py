from REINFORCE_model import ReinforceModel
import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_discounted_returns(rewards, gamma=0.99):
    returns = []
    discounted_sum = 0
    # Loop backwards through the rewards
    for r in rewards[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    return returns


def train_episode(env, model, optimizer, gamma=0.99):
    states = []
    actions = []
    rewards = []

    state, _ = env.reset()
    done = False

    while not done:
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_logits = model(state_tensor)
        action = int(tf.random.categorical(action_logits, 1)[0, 0].numpy())
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    # calculate monte carlo returns
    returns = get_discounted_returns(rewards, gamma)
    returns_tensor = tf.constant(returns, dtype=tf.float32)

    # single batch forward pass inside the tape for the loss
    with tf.GradientTape() as tape:
        all_logits = model(tf.constant(states, dtype=tf.float32))
        negative_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.constant(actions, dtype=tf.int32), logits=all_logits)
        loss = tf.reduce_sum(negative_log_probs * returns_tensor)

    # update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return sum(rewards)


def run_episodes(env, model, optimizer, n_episodes):
    rewards_list = []
    for ep in range(n_episodes):
        rewards_list.append(train_episode(env, model, optimizer))
        if (ep + 1) % 100 == 0:
            recent = np.mean(rewards_list[-100:])
            print(f"  episode {ep + 1}/{n_episodes}  avg return (last 100): {recent:.1f}")
    return rewards_list


if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    # Keep exactly the same seeds as your AC implementation for fair comparison
    seed = 69
    tf.random.set_seed(seed)
    np.random.seed(seed)

    num_actions = env.action_space.n
    num_hidden_units = 128

    model = ReinforceModel(int(num_actions), num_hidden_units)
    # Using the same optimizer and learning rate as your AC script
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    print("Starting REINFORCE training...")
    rewards_list = run_episodes(env, model, optimizer, 1000)

    # Plot the learning curve
    plt.plot(np.arange(len(rewards_list)), rewards_list, label="REINFORCE")
    plt.xlabel("Episodes")
    plt.ylabel("Total Return")
    plt.title("REINFORCE Learning Curve on CartPole-v1")
    plt.legend()
    plt.show()
