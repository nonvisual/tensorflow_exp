from gym import spaces
import gym
import inspect
from multiprocessing import Process
from operator import itemgetter
import os
import time

from numpy import shape, math
import scipy.signal

import numpy as np
from policy_gradient import logz
import tensorflow as tf


#============================================================================================#
# Utilities
#============================================================================================#
def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh,
        output_activation=None
        ):
    #========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units. 
    # 
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    #========================================================================================#


    
    with tf.variable_scope(scope):
        previous_layer = input_placeholder
        
        for i in range(1, n_layers):    
            with tf.name_scope(str(scope) + str(i)):
                previous_layer = tf.layers.dense(inputs=previous_layer, units=size,
                                                 activation=activation, kernel_initializer=tf.glorot_uniform_initializer())
                #===============================================================
                # weights = tf.Variable(
                #     tf.truncated_normal([previous_size, size],
                #             stddev=1.0/math.sqrt(previous_size)),
                #                       name='weights')
                # biases = tf.Variable(tf.zeros([size]),
                #          name='biases')
                # previous_layer= activation(tf.matmul(previous_layer, weights) + biases)
                #===============================================================
                previous_layer = tf.layers.dropout(inputs=previous_layer, rate=0.1)
            
    # add output layer
        with tf.name_scope('output'):
            #-------------------------------------------- weights = tf.Variable(
                #------------- tf.truncated_normal([previous_size, output_size],
                        #---------------- stddev=1.0/math.sqrt(previous_size)) ,
                                  #----------------------------- name='weights')
            #--------------------- biases = tf.Variable(tf.zeros([output_size]),
                     #------------------------------------------- name='biases')
            # output= output_activation(tf.matmul(previous_layer, weights) + biases)
            output = tf.layers.dense(inputs=previous_layer, units=output_size, activation=output_activation)
    return output

def pathlength(path):
    return len(path["reward"])



#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100,
             gamma=1.0,
             min_timesteps_per_batch=1000,
             max_path_length=None,
             learning_rate=5e-3,
             reward_to_go=True,
             animate=True,
             logdir=None,
             normalize_advantages=True,
             nn_baseline=True,
             seed=0,
             # network arguments
             n_layers=1,
             size=32,
             store_rewards = True

             ):

    start = time.time()

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

    # Make the gym environment
    env = gym.make(env_name)
    
    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    #========================================================================================#
    # Notes on notation:
    # 
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    # 
    # Prefixes and suffixes:
    # ob - observation 
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    # 
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    print("Env observ: ", ob_dim, " actions ", ac_dim)
    #========================================================================================#
    #                           ----------SECTION 4----------
    # Placeholders
    # 
    # Need these for batch observations / actions / advantages in policy gradient loss function.
    #========================================================================================#

    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
    else:
        if ac_dim == 1:  # in case of single action, still shape is None, but type is different
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.float32) 
        else:
            sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) 

    # Define a placeholder for advantages = how much better our policy is than baseline
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) 


   
    output_ops = build_mlp(sy_ob_no, ac_dim, "neuralnetwork", n_layers, size, activation=tf.tanh)
    
    if discrete:
        # policy network output
        sy_logits_na = output_ops
        
        sy_sampled_ac = tf.reshape(tf.multinomial(sy_logits_na, num_samples=1), [-1])  # sample from distribution, serves as action output
        
        # Computing the log probability of a set of actions that were actually taken, 
        #      according to the policy. Use it then in loss function approximation
        sy_logprob_n = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sy_logits_na, labels=sy_ac_na)
    else:
        # policy network output mean and variance
        sy_mean_a = tf.reduce_mean(output_ops, 0)
        
        # trainable variable for std deviation of output
        sy_logstd_a = tf.get_variable(initializer=tf.constant([0.5]), dtype=tf.float32,
                                  name='logstd') 
        # sampling from gaussian distribution (can use also sample from MultivariateNormalDiag)
        sy_sampled_ac = sy_mean_a + tf.multiply(sy_logstd_a, tf.random_normal(shape=[ac_dim]))
        
        dist = tf.contrib.distributions.MultivariateNormalDiag(loc=output_ops, scale_diag=sy_logstd_a)
        sy_logprob_n = -dist.log_prob(sy_ac_na[:, None])  # probabilities of taken actions



    #========================================================================================#
    #                           ----------SECTION 4----------
    # Loss Function and Training Operation
    #========================================================================================#
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    
    weighted_negative_lkhs = tf.multiply(sy_logprob_n, sy_adv_n)
    loss = tf.reduce_mean(weighted_negative_lkhs)  # Loss function that we'll differentiate to get the policy gradient.
    update_op = optimizer.minimize(loss)


    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
                                sy_ob_no,
                                1,
                                "nn_baseline",
                                n_layers=n_layers,
                                size=size))
    # Define placeholders for targets, a loss function and an update op for fitting a 
    # neural network baseline. These will be used to fit the neural network baseline. 
        sy_bvalue_n = tf.placeholder(shape=[None], name="b_value", dtype=tf.float32)  # placeholder for value function values

        baseline_loss = tf.losses.mean_squared_error(labels=sy_bvalue_n, predictions=baseline_prediction)
        baseline_update_op = optimizer.minimize(baseline_loss)


    #
    # Tensorflow Engineering: Config, Session, Variable initialization
    #

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=4) 

    sess = tf.Session(config=tf_config)
    sess.__enter__()  # equivalent to `with sess:`
    tf.global_variables_initializer().run()  # pylint: disable=E1101



    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0
    prev_loss = 0
    stored_paths = []
    stored_size = 40
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
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                ac = ac[0]
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation" : np.array(obs),
                    "reward" : np.array(rewards),
                    "action" : np.array(acs),
                    "final_reward": np.sum(rewards)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
       
        # TODO: store more rewarding paths and add them to batches
                          
        q_n = []
        extended_paths=paths
        if(store_rewards):
            stored_paths=np.concatenate([stored_paths,sorted(paths, key=itemgetter('final_reward'))])
            stored_paths = sorted(stored_paths, key=itemgetter('final_reward'),reverse=True)[0:stored_size]
            extended_paths=np.concatenate([paths,stored_paths])
            print("max reward", stored_paths[0]["final_reward"])   
        ob_no = np.concatenate([path["observation"] for path in extended_paths])
        ac_na = np.concatenate([path["action"] for path in extended_paths])
 
        if reward_to_go == True:
            for path in extended_paths:
                r = []  # compute full reward for the trajectory
                t = 0
                for i in reversed(path['reward']):
                    t = gamma * t + i
                    r.append(t)
                q_n += r[::-1]
        else:
            for path in extended_paths:
                r = 0  # compute full reward for the trajectory
                for i in reversed(path['reward']):
                    r = gamma * r + i
                for i in path['reward']:
                    q_n.append(r)
        
        
          
                            
        #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#

        if nn_baseline:
            # predicting baseline
            prediction = sess.run([
            baseline_prediction
              ], {
                sy_ob_no: ob_no,
                sy_ac_na: ac_na,
                sy_adv_n: q_n             
              })
            b_n = prediction[0] * np.std(q_n) + np.mean(q_n)  # scale back to the std and mean of values (trained on normalized)
            adv_n = q_n - b_n
        else:
            adv_n = q_n.copy()

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            adv_n = (adv_n - np.mean(q_n)) / (np.std(q_n) + 0.000001)



        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)
            values = []
            for path in extended_paths:
                r = 0  # compute full reward for the trajectory
                for i in reversed(path['reward']):
                    r = gamma * r + i
                for i in path['reward']:
                    values.append(r)
            values = (values - np.mean(values)) / (np.std(values) + 0.00001)  # normalize
            # updating nn for baseline prediction
            sess.run([
            baseline_update_op
              ], {
                sy_ob_no: ob_no,
                sy_ac_na: ac_na,
                sy_adv_n: adv_n,
                sy_bvalue_n: values              
              })
            

        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#


        _, exp_loss = sess.run([
            update_op,
            loss
          ], {
            sy_ob_no: ob_no,
            sy_ac_na: ac_na,
            sy_adv_n: adv_n
          })
        
        print("loss before update", exp_loss)

        exp_loss = sess.run([           
            loss
          ], {
            sy_ob_no: ob_no,
            sy_ac_na: ac_na,
            sy_adv_n: adv_n
          })
        
        print("loss after update", exp_loss)
        
        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("Loss", exp_loss)
        logz.log_tabular("AverageReturn", np.mean(returns))
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name  # + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10 * e
        print('Running experiment with seed %d' % seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir, '%d' % seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline,
                seed=seed,
                n_layers=args.n_layers,
                size=args.size
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()
        

if __name__ == "__main__":
    main()
