import tensorflow as tf
from tensorflow.keras import layers

class ReinforceModel(tf.keras.Model):
    def __init__(self, num_actions, num_hidden_units):
        super().__init__()
        # Only need an actor (policy) network, no critic.
        self.dense = layers.Dense(num_hidden_units, activation='relu')
        self.policy_logits = layers.Dense(num_actions)

    def call(self, inputs):
        x = self.dense(inputs)
        # Output the raw logits.
        # Softmax is in the loss function btw.
        return self.policy_logits(x)