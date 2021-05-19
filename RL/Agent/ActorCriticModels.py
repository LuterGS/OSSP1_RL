import tensorflow as tf
import tensorflow.keras.layers as layers

model_size = (480, 640, )


def Actor_Continuos_Default(state_dimension, action_dimension, action_bound):

    inputs = layers.Input((state_dimension, ))
    dense_1 = layers.Dense(32, activation='relu')(inputs)
    dense_2 = layers.Dense(32, activation='relu')(dense_1)
    out_mu = layers.Dense(action_dimension, activation='tanh')(dense_2)
    mu_output = layers.Lambda(lambda x: x * action_bound)(out_mu)
    std_output = layers.Dense(action_dimension, activation='softplus')(dense_2)
    return tf.keras.models.Model(inputs, [mu_output, std_output])


def Actor_Continuos_IMG(state_dimension, action_dimension, action_bound):

    inputs = layers.Input((model_size, ))
    cnn_1 = layers.Conv2D(64, 10, activation='relu')(inputs)
    batch_norm = layers.BatchNormalization()(cnn_1)
    dropout = layers.Dropout(0.2)(batch_norm)
    cnn_2 = layers.Conv2D(128, 8, activation='relu')(dropout)
    batch_norm = layers.BatchNormalization()(cnn_2)
    dropout = layers.Dropout(0.2)(batch_norm)
    flatten = layers.Flatten()(dropout)
    out_mu = layers.Dense(action_dimension, activation='relu')(flatten)
    mu_output = layers.Lambda(lambda x: x * action_bound)(out_mu)
    std_output = layers.Dense(action_dimension, activation='relu')(flatten)
    return tf.keras.models.Model(inputs, [mu_output, std_output])


def Critic_Default(state_dimension):
    return tf.keras.Sequential([
        layers.Input((state_dimension, )),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='linear')
    ])


def Critic_CNN():
    return tf.keras.Sequential([
        layers.Input((model_size,)),
        layers.Conv2D(64, 10, activation='relu'),
        layers.Dropout(0.2),
        layers.Conv2D(128, 8, activation='relu'),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dense(1, activations='linear')
    ])