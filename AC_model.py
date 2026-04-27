import tensorflow as tf
from tensorflow.keras import layers


class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions, num_hidden_units):
        super().__init__()

        # self.shared = layers.Dense(num_hidden_units, activation='relu')
        self.actor_dense = layers.Dense(num_hidden_units, activation='relu')
        self.actor = layers.Dense(num_actions)

        self.critic_dense = layers.Dense(num_hidden_units, activation='relu')
        # Q(s,a): one Q-value per action, approximating Q^π
        self.critic = layers.Dense(num_actions)

    def call(self, inputs):
        # x = self.shared(inputs)
        actor = self.actor_dense(inputs)
        critic = self.critic_dense(inputs)
        return self.actor(actor), self.critic(critic)