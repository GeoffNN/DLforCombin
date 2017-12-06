import tensorflow as tf
import tsp_env
import numpy as np
import itertools
import Q_function_graph_model2 as Q_function_graph_model

# Define
n_cities = 5
T = 4
n_mlp_layers = 1
p = 64
n_dagger_steps = 100; max_steps_per_rollout = 10; n_rollouts = 10; n_gradient_steps = 20
learning_rate = 1e-3

# Define ph
obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, n_cities])
expert_act_ph = tf.placeholder(dtype=tf.int32, shape=[None])
adj_ph = tf.placeholder(tf.float32, [None, n_cities, n_cities],
                            name='adj_ph')
graph_weights_ph = tf.placeholder(tf.float32,
                                  [None, n_cities, n_cities],
                                  name='graph_weights_ph')
action_logits_ph = Q_function_graph_model.Q_func(
    obs_ph, adj_ph, graph_weights_ph,
    p=p, T=T, scope='nn_actions',
    initialization_stddev=1e-3,
    reuse=False,
    pre_pooling_mlp_layers=2, post_pooling_mlp_layers=1
)

sampled_act_ph = tf.multinomial(action_logits_ph, 1)[0]

loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.nn.log_softmax(action_logits_ph),
                     tf.one_hot(expert_act_ph, depth=n_cities)), axis = 1))
update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Define env and session
env = tsp_env.TSP_env(5, 2)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Rollouts
data = {'observations': [], 'adj_matrices': [],
        'weight_matrices': [], 'expert_actions': []}
for i in range(n_dagger_steps):
    # Do n_rollouts rollouts under fitted policy
    rewards = []
    for j in range(n_rollouts):
        n_env_steps = 0
        state = env.reset()
        total_reward = 0
        while True:
            # Get actions suggested by fitted policy and by expert policy
            action = sess.run(sampled_act_ph,
                              feed_dict={obs_ph: state[None],
                                         adj_ph: env.adjacency_matrix[None],
                                         graph_weights_ph: env.weight_matrix[None]})[0]
            # import pdb; pdb.set_trace()
            expert_action = env.best_solution_from_now()[1][0]

            # Store data from current step
            data['observations'].append(state)
            data['adj_matrices'].append(env.adjacency_matrix)
            data['weight_matrices'].append(env.weight_matrix)
            data['expert_actions'].append(expert_action)

            # Step the according to fitted policy
            state, reward, done = env.step(action)
            total_reward += reward

            n_env_steps += 1
            if done or n_env_steps > max_steps_per_rollout:
                break
        rewards.append(total_reward)

    print('Mean rollout reward for DAGGER step ', i, ': ', np.mean(rewards))

    # Perform training step
    for j in range(n_gradient_steps):
        loss_val, _ = sess.run([loss, update_op],
                               feed_dict={obs_ph: np.array(data['observations']),
                                          adj_ph: np.array(data['adj_matrices']),
                                          graph_weights_ph: np.array(data['weight_matrices']),
                                          expert_act_ph: np.array(data['expert_actions'])})
    print('Loss val at the end of DAGGER step ', i, ': ', loss_val)
