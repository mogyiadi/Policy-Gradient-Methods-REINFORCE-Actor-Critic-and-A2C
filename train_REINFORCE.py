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

    # Generate episode
    with tf.GradientTape() as tape:
        while not done:
            # Get action probabilities
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            action_logits = model(state_tensor)

            # Sample an action
            action_tensor = tf.random.categorical(action_logits, 1)[0, 0]
            action = int(action_tensor.numpy())

            # Take step
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # Calculate mote carlo returns
        returns = get_discounted_returns(rewards, gamma)

        # Convert lists to tensors for loss calculation
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)

        # Calculate Loss
        # Get logits for all states visited in the episode
        all_logits = model(states_tensor)

        # Calculate the negative log probability: -log(pi(a|s))
        negative_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions_tensor, logits=all_logits)

        # Multiply by the Monte Carlo estimate (the return G_t)
        # This matches the assignment formula: expected value of [-Q * log_prob]
        loss = tf.reduce_sum(negative_log_probs * returns_tensor)

    # Update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return sum(rewards)


def run_episodes(env, model, optimizer, n_episodes):
    rewards_list = []
    for episode in range(n_episodes):
        total_reward = train_episode(env, model, optimizer)
        rewards_list.append(total_reward)

        if episode % 50 == 0:
            print(f"Episode {episode}: Total Reward = {total_reward}")

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