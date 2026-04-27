from A2C_model import AdvantageActorCritic
import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def train_episode(env, model, optimizer, gamma=0.99):
    states, actions, rewards = [], [], []
    state, _ = env.reset()
    done = False

    # collect full episode without tape — same collection strategy as REINFORCE
    while not done:
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_logits, _ = model(state_tensor)
        action = int(tf.random.categorical(action_logits, 1)[0, 0].numpy())
        next_state, reward, terminated, truncated, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        done = terminated or truncated

    # MC returns G_t: same estimate REINFORCE uses for Q^π
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    returns_tensor = tf.constant(returns, dtype=tf.float32)

    # single batch update over the full episode
    with tf.GradientTape() as tape:
        states_tensor = tf.constant(states, dtype=tf.float32)
        action_logits, values = model(states_tensor)
        # squeeze critic output from (T, 1) to (T,)
        values = tf.squeeze(values, axis=1)

        # advantage A(s,a) = G_t - V(s_t): how much better was this return than expected
        advantages = tf.stop_gradient(returns_tensor) - values

        # critic loss: push V(s) toward the MC return
        critic_loss = tf.reduce_mean(tf.math.square(advantages))

        # actor loss: -log π(a|s) * A(s,a)
        neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.constant(actions, dtype=tf.int32),
            logits=action_logits
        )
        actor_loss = tf.reduce_mean(neg_log_probs * tf.stop_gradient(advantages))

        total_loss = critic_loss + actor_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    clipped, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(clipped, model.trainable_variables))

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

    seed = 69
    tf.random.set_seed(seed)
    np.random.seed(seed)

    num_actions = int(env.action_space.n)
    num_hidden_units = 128

    model = AdvantageActorCritic(num_actions, num_hidden_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    rewards_list = run_episodes(env, model, optimizer, 1000)

    plt.plot(np.arange(len(rewards_list)), rewards_list)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('A2C on CartPole-v1')
    plt.grid(True, alpha=0.3)
    plt.show()
