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
# from atari_wrappers import *
from knapsack_env import *

def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

    return out

def knapsack_model(input_placeholder, num_actions, scope, reuse=False,
                   n_layers=2, size=64, activation=tf.nn.relu,
                   output_activation=tf.nn.log_softmax):
    # import pdb; pdb.set_trace()
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

def knapsack_learn(env,
                   session,
                   num_timesteps, lr_multiplier=1.0, target_update_freq=10000,
                   exp_name='Knapsack_DQN', boltzmann_exploration=False):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

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
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

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
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=512,
        gamma=1,
        learning_starts=5000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=target_update_freq,
        grad_norm_clipping=10,
        exp_name=exp_name,
        boltzmann_exploration=boltzmann_exploration
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_multiplier', type=float, default=1.0)
    parser.add_argument('--target_update_freq', type=float, default=10000)
    parser.add_argument('--exp_name', type=str, default='Knapsack')
    parser.add_argument('--boltzmann_exploration', action='store_true')
    args = parser.parse_args()

    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    # env = get_env(task, seed)
    env = Knapsack(10, 3)
    # test_env = Knapsack(5, 1)
    # session = get_session()
    knapsack_learn(env, None, num_timesteps=task.max_timesteps,
                lr_multiplier=args.lr_multiplier,
                target_update_freq=args.target_update_freq,
                exp_name=args.exp_name,
                boltzmann_exploration=args.boltzmann_exploration)

if __name__ == "__main__":
    main()
