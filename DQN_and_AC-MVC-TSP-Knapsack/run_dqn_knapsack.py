import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *
from atari_wrappers import *
from knapsack_env import *
import tsp_env

def knapsack_model(input_placeholder, num_actions, scope, reuse=False,
                   n_layers=2, size=64, activation=tf.nn.relu,
                   output_activation=tf.nn.log_softmax):
    with tf.variable_scope(scope, reuse=reuse):
        curr_activations = activation(input_placeholder)
        for _ in range(n_layers):
            curr_activations = activation(tf.layers.dense(inputs=curr_activations,
                                                          units=size))
        if output_activation == None:
            return tf.layers.dense(inputs=curr_activations,
                                   units=num_actions, name='Q_output')
        else:
            return output_activation(tf.layers.dense(inputs=curr_activations,
                                                     units=num_actions))

def knapsack_learn(env, num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return False

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=knapsack_model,
        nn_size=3,
        n_hidden_units=128,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        double_DQN=True,
        n_steps_ahead=3
    )
    env.close()


def main():
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[2]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    # env = get_env(task, seed)
    env = Knapsack(10, 3)
    #env = tsp_env.TSP_env(5, no_move_penalty=0,
    #                      use_alternative_state=True)
    knapsack_learn(env, num_timesteps=task.max_timesteps)


if __name__ == "__main__":
    main()
