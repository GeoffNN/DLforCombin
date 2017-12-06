import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process
import knapsack_env
import mvc_env
# import tsp_env
import Q_function_graph_model2 as Q_function_graph_model

# ============================================================================================#
# Utilities
# ============================================================================================#

def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        output_activation=None
):
    with tf.variable_scope(scope):
        curr_activations = activation(input_placeholder)
        for _ in range(n_layers):
            curr_activations = activation(tf.layers.dense(inputs=curr_activations,
                                                          units=size))
        if output_activation == None:
            return tf.layers.dense(inputs=curr_activations,
                                   units=output_size)
        else:
            return output_activation(tf.layers.dense(inputs=curr_activations,
                                                     units=output_size))


def pathlength(path):
    return len(path["reward"])


# ============================================================================================#
# Policy Gradient
# ============================================================================================#

def train_PG(exp_name,
             env_name,
             n_iter=100,
             gamma=1.0,
             min_timesteps_per_batch=1000,
             max_path_length=None,
             learning_rate=5e-3,
             reward_to_go=True,
             animate=True,
             logdir=None,
             normalize_advantages=True,
             nn_baseline=False,
             seed=0,
             # network arguments
             n_layers=1,
             size=32,
             T=4,
             ):

    start = time.time()

    #env = knapsack_env.Knapsack(10, 3)
    env = mvc_env.MVC_env(7, replay_penalty=2)
    # env = tsp_env.TSP_env(10, 2)

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Maximum length for episodes
    max_path_length = 15


    # Observation and action sizes
    ob_dim = env.state_shape[0]
    ac_dim = env.num_actions
    # ========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    #
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    # ========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
    # Define a placeholder for advantages```````````````````````````````````````````````````````````````````````````````
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
    # Graph placeholders
    if env.env_name != 'Knapsack':
        adj_ph = tf.placeholder(tf.float32, [None, env.number_nodes, env.number_nodes],
                                name='adj_ph')
        graph_weights_ph = tf.placeholder(tf.float32, \
                                          [None, env.number_nodes, env.number_nodes],
                                          name='graph_weights_ph')
    # ========================================================================================#
    # Networks
    # ========================================================================================#

    if env.env_name == 'Knapsack':
        sy_logits_na = build_mlp(
            sy_ob_no,
            output_size=ac_dim,
            scope="nn_actions",
            n_layers=n_layers,
            size=size,
            activation=tf.tanh,
            output_activation=None
        )
    else:
        sy_logits_na = Q_function_graph_model.Q_func(sy_ob_no, adj_ph, graph_weights_ph,
                                                     p=size, T=T, scope='nn_actions',
                                                     initialization_stddev=1e-3,
                                                     reuse=False,
                                                     pre_pooling_mlp_layers=2, post_pooling_mlp_layers=2)

    sy_sampled_ac = tf.multinomial(sy_logits_na, 1, seed)[0]
    sy_logprob_n = -tf.reduce_sum(tf.multiply(tf.nn.log_softmax(sy_logits_na),
                                              tf.one_hot(sy_ac_na, depth=ac_dim)), axis=1)



    # ========================================================================================#
    # Loss Function and Training Operation
    # ========================================================================================#

    loss = tf.reduce_mean(
        tf.multiply(sy_logprob_n, sy_adv_n))  # Loss function that we'll differentiate to get the policy gradient.
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # ========================================================================================#
    # Baseline
    # ========================================================================================#

    if nn_baseline:
        if env.env_name == 'Knapsack':
            baseline_prediction = tf.squeeze(build_mlp(
                sy_ob_no,
                1,
                "nn_baseline",
                n_layers=n_layers,
                size=size))
        else:
            baseline_prediction = tf.sigmoid(tf.squeeze(tf.layers.dense(inputs=Q_function_graph_model.Q_func(
                sy_ob_no, adj_ph, graph_weights_ph,
                p=size, T=T, scope='pi',
                initialization_stddev=1e-3,
                reuse=False, pre_pooling_mlp_layers=2, post_pooling_mlp_layers=2),
            units=1)))
        # Define placeholders for targets, a loss function and an update op for fitting a
        # neural network baseline. These will be used to fit the neural network baseline.
        sy_q_n = tf.placeholder(shape=[None], dtype=tf.float32)
        baseline_loss = tf.losses.mean_squared_error(labels=sy_q_n,
                                                     predictions=baseline_prediction)
        baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize(baseline_loss)

    # ========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    # ========================================================================================#
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    log_files_name = 'PG_' + str(env.env_name) + \
                     '-sz=' + str(size) + \
                     '-b=' + str(min_timesteps_per_batch) + '-' + \
                     time.strftime('%m-%d-%Y-%H:%M:%S')
    saver.save(sess, '/tmp/saved_models/' + log_files_name)

    # ========================================================================================#
    # Training Loop
    # ========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                if env.env_name == 'Knapsack':
                    feed_dict = {sy_ob_no: ob[None]}
                else:
                    feed_dict = {sy_ob_no: ob[None],
                                 adj_ph: env.adjacency_matrix[None],
                                 graph_weights_ph: env.weight_matrix[None]}
                ac = sess.run(sy_sampled_ac, feed_dict=feed_dict)
                ac = ac[0]
                acs.append(ac)
                ob, rew, done = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break

            if env.env_name == 'Knapsack':
                optimal_solution = env.optimal_solution()
                accuracy = np.mean(1 - np.abs(env.xs - optimal_solution[1]))
                approx_ratio = np.sum(rewards) / optimal_solution[0]
                path = {"observation": np.array(obs),
                        "reward": np.array(rewards),
                        "action": np.array(acs),
                        "accuracy": accuracy,
                        "approx_ratio": approx_ratio}
            else:
                accuracy = 0
                path = {"observation": np.array(obs),
                        "reward": np.array(rewards),
                        "action": np.array(acs),
                        "adj": [env.adjacency_matrix] * len(obs),
                        "weight_matrix": [env.weight_matrix] * len(obs),
                        "accuracy": accuracy}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        accuracies = [path["accuracy"] for path in paths]
        if env.env_name != 'Knapsack':
            adj_batch = np.concatenate([path["adj"] for path in paths])
            graph_weights_batch = np.concatenate([path["weight_matrix"] for path in paths])
        else:
            approx_ratios = [path["approx_ratio"] for path in paths]

        # ====================================================================================#
        # Computing Q-values
        # ====================================================================================#
        if reward_to_go:
            paths_rtgs = []
            for path in paths:
                rtg_current_path = 1 / np.power(gamma, np.arange(0, len(path["reward"]))) * \
                                   np.cumsum((path["reward"] *
                                              np.power(gamma, np.arange(0, len(path["reward"]))))[::-1])[::-1]
                paths_rtgs.append(rtg_current_path)
            q_n = np.concatenate(paths_rtgs)
        else:
            paths_Rets = []
            for path in paths:
                Ret_current_path = np.sum(path["reward"] * \
                                          np.power(gamma, -np.arange(0, len(path["reward"]))))
                paths_Rets.append(Ret_current_path * np.ones((len(path["reward"]) ) ) )
            q_n = np.concatenate(paths_Rets)

        # ====================================================================================#
        # Computing Baselines
        # ====================================================================================#
        if nn_baseline:
            if env.env_name == 'Knapsack':
                feed_dict = {sy_ob_no: ob_no}
            else:
                feed_dict = {sy_ob_no: ob_no,
                             adj_ph: adj_batch,
                             graph_weights_ph: graph_weights_batch}
            raw_b_n = sess.run(baseline_prediction, feed_dict=feed_dict)

            centered_b_n = raw_b_n - np.mean(raw_b_n)
            scaled_b_n = np.std(q_n) / np.std(centered_b_n) * centered_b_n
            b_n = scaled_b_n + np.mean(q_n)
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        # ====================================================================================#
        # Advantage Normalization
        # ====================================================================================#
        if normalize_advantages:
            adv_n = (adv_n - np.mean(adv_n)) / np.std(adv_n)

        # ====================================================================================#
        # Optimizing Neural Network Baseline
        # ====================================================================================#
        if nn_baseline:
            rescaled_q_n = (q_n - np.mean(q_n)) / np.std(q_n)
            for i in range(10):
                if env.env_name == 'Knapsack':
                    feed_dict = {sy_ob_no: ob_no,
                                 sy_q_n: rescaled_q_n}
                else:
                    feed_dict = {sy_ob_no: ob_no,
                                 sy_q_n: rescaled_q_n,
                                 adj_ph: adj_batch,
                                 graph_weights_ph: graph_weights_batch}
                sess.run(baseline_update_op, feed_dict=feed_dict)

        # ====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        # ====================================================================================#
        if env.env_name == 'Knapsack':
            feed_dict = {sy_ob_no: ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n}
        else:
            feed_dict = {sy_ob_no: ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n,
                         adj_ph: adj_batch, graph_weights_ph: graph_weights_batch}
        sess.run(update_op, feed_dict=feed_dict)

        # Save model
        saver.save(sess, '/tmp/saved_models/' + log_files_name, global_step=itr)
        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        if env.env_name == 'Knapsack':
            logz.log_tabular("AverageAccuracy", np.mean(accuracies))
            logz.log_tabular("ApproxRatio", np.mean(approx_ratios))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    if not (os.path.exists('data')):
        os.makedirs('data')
    logdir = 'data/MVC-PG-' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    # n_experiments = 3
    #
    # for e in range(n_experiments):
    #     seed = 10 * e
    #     print('Running experiment with seed %d' % seed)

    # def train_func():
    train_PG(
        exp_name='MVC-PG',
        env_name='MVC',
        n_iter=int(1e5),
        gamma=0.98,
        min_timesteps_per_batch=5000,
        max_path_length=15,
        learning_rate=1e-2,
        reward_to_go=True,
        animate=False,
        logdir=os.path.join(logdir, '%d' % 1),
        normalize_advantages=True,
        nn_baseline=False,
        seed=1,
        n_layers=1,
        size=64, T=4)

    # # Awkward hacky process runs, because Tensorflow does not like
    # # repeatedly calling train_PG in the same thread.
    # p = Process(target=train_func, args=tuple())
    # p.start()
    # p.join()



if __name__ == "__main__":
    main()
