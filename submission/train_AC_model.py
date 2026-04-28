from AC_model import ActorCritic
import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm


def train_episode(env, model, optimizer, gamma=0.99):
    states, actions, rewards, next_states, terminals = [], [], [], [], []
    state, _ = env.reset()
    done = False

    while not done:
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        action_logits, _ = model(state_tensor)
        action = int(tf.random.categorical(action_logits, 1)[0, 0].numpy())
        next_state, reward, terminated, truncated, _ = env.step(action)
        terminal = terminated or truncated
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        terminals.append(float(terminal))
        state = next_state
        done = terminal

    T = len(states)
    states_t = tf.constant(states, dtype=tf.float32)
    next_states_t = tf.constant(next_states, dtype=tf.float32)
    actions_t = tf.constant(actions, dtype=tf.int32)
    rewards_t = tf.constant(rewards, dtype=tf.float32)
    terminals_t = tf.constant(terminals, dtype=tf.float32)

    # sample next actions from current policy
    next_logits, next_q_vals = model(next_states_t)
    next_actions = tf.cast(tf.squeeze(tf.random.categorical(next_logits, 1), axis=1), tf.int32)
    next_q = tf.gather_nd(next_q_vals, tf.stack([tf.range(T), next_actions], axis=1))
    sarsa_targets = tf.stop_gradient(rewards_t + gamma * next_q * (1.0 - terminals_t))

    # single batch update inside one tape
    with tf.GradientTape() as tape:
        action_logits, q_vals = model(states_t)
        current_q = tf.gather_nd(q_vals, tf.stack([tf.range(T), actions_t], axis=1))

        # critic loss
        critic_loss = tf.reduce_mean(tf.math.square(sarsa_targets - current_q))

        # actor loss
        neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions_t, logits=action_logits)
        actor_loss = tf.reduce_mean(neg_log_probs * tf.stop_gradient(current_q))

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

    num_actions = env.action_space.n
    num_hidden_units = 128

    model = ActorCritic(int(num_actions), num_hidden_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    rewards_list = run_episodes(env, model, optimizer, 1000)

    plt.plot(np.arange(len(rewards_list)), rewards_list)
    plt.show()








