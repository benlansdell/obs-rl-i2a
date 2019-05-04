# To train the environment model. See paper appendix for implementation
# details.

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow as tf
import gym
import gym_minigrid
from a2c import get_actor_critic, CnnPolicy
from common.multiprocessing_env import SubprocVecEnv
import numpy as np

from tqdm import tqdm

from common.minigrid_util import num_pixels, mode_rewards, pix_to_target, rewards_to_target

# How many iterations we are training the environment model for.
NUM_UPDATES = 10000

LOG_INTERVAL = 500

N_ENVS = 16
N_STEPS = 5

# This can be anything from "regular" "avoid" "hunt" "ambush" "rush" each
# resulting in a different reward function giving the agent different behavior.
REWARD_MODE = 'regular'

# Replace this with the location of your own weights. This is a partially trained model... 
A2C_WEIGHTS = 'weights/a2c_100000.ckpt'

def initialize_uninitialized_vars(sess):
    from itertools import compress
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def pool_inject(X, batch_size, depth, width, height):
    m = tf.layers.max_pooling2d(X, pool_size=(width, height), strides=(width, height))
    tiled = tf.tile(m, (1, width, height, 1))
    return tf.concat([tiled, X], axis=-1)

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

def basic_block(X, batch_size, depth, width, height, n1, n2, n3):
    with tf.variable_scope('pool_inject'):
        p = pool_inject(X, batch_size, depth, width, height)

    #with tf.variable_scope('part_1_block'):
    #    # Padding was 6 here
    #    p_padded = tf.pad(p, [[0, 0], [6, 6], [6, 6], [0, 0]])
    #    p_1_c1 = tf.layers.conv2d(p_padded, n1, kernel_size=1,
    #            strides=2, padding='valid', activation=tf.nn.relu)

    #    # Padding was 5, 6
    #    p_1_c1 = tf.pad(p_1_c1, [[0,0], [5, 5], [6, 6], [0, 0]])
    #    p_1_c2 = tf.layers.conv2d(p_1_c1, n1, kernel_size=8, strides=1,
    #            padding='valid', activation=tf.nn.relu)

    with tf.variable_scope('part_2_block'):
        p_2_c1 = tf.layers.conv2d(p, n2, kernel_size=1,
                activation=tf.nn.relu)

        p_2_c1 = tf.pad(p_2_c1, [[0,0],[1,1],[1,1],[0,0]])
        p_2_c2 = tf.layers.conv2d(p_2_c1, n2, kernel_size=3, strides=1,
                padding='valid', activation=tf.nn.relu)

    with tf.variable_scope('combine_parts'):
        #combined = tf.concat([p_1_c2, p_2_c2], axis=-1)
        combined = p_2_c2
        c = tf.layers.conv2d(combined, n3, kernel_size=1,
                activation=tf.nn.relu)

    return tf.concat([c, X], axis=-1)

""" ###########################################################################"""
""" ###########################################################################"""
""" ###########################################################################"""
""" ###########################################################################"""
""" ###########################################################################"""
""" ###########################################################################"""

def create_env_model(obs_shape, num_actions, num_pixels, num_rewards,
        should_summary=True, reward_coeff=1.0):
    width = obs_shape[0]
    height = obs_shape[1]
    depth = obs_shape[2]

    states = tf.placeholder(tf.float32, [None, width, height, depth])

    onehot_actions = tf.placeholder(tf.float32, [None, width,
        height, num_actions])

    batch_size = tf.shape(states)[0]
    target_states = tf.placeholder(tf.uint8, [None])
    target_rewards = tf.placeholder(tf.uint8, [None])

    inputs = tf.concat([states, onehot_actions], axis=-1)

    with tf.variable_scope('pre_conv'):
        c = tf.layers.conv2d(inputs, 64, kernel_size=1, activation=tf.nn.relu)

    with tf.variable_scope('basic_block_1'):
        bb1 = basic_block(c, batch_size, 64, width, height, 16, 32, 64)

    with tf.variable_scope('basic_block_2'):
        bb2 = basic_block(bb1, batch_size, 128, width, height, 16, 32, 64)

    with tf.variable_scope('image_conver'):
        image = tf.layers.conv2d(bb2, 256, kernel_size=1, activation=tf.nn.relu)
        image = tf.reshape(image, [batch_size * width * height, 256])
        image = tf.layers.dense(image, num_pixels)

    with tf.variable_scope('reward'):
        reward = tf.layers.conv2d(bb2, 64, kernel_size=1,
                activation=tf.nn.relu)

        #reward = tf.layers.conv2d(reward, 64, kernel_size=1,
        #        activation=tf.nn.relu)

        reward = tf.reshape(reward, [batch_size, width * height * 64])

        reward = tf.layers.dense(reward, num_rewards)

    target_states_one_hot = tf.one_hot(target_states, depth=num_pixels)
    image_loss = tf.losses.softmax_cross_entropy(target_states_one_hot, image)

    target_reward_one_hot = tf.one_hot(target_rewards, depth=num_rewards)
    reward_loss = tf.losses.softmax_cross_entropy(target_reward_one_hot, reward)

    loss = image_loss + (reward_coeff * reward_loss)
    #Just learn image_loss
    #loss = image_loss

    opt = tf.train.AdamOptimizer(beta1 = 0.9, beta2 = 0.999).minimize(loss)

    # Tensorboard
    if should_summary:
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Reward Loss', reward_loss)
        tf.summary.scalar('Image Loss', image_loss)

    return EnvModelData(image, reward, states, onehot_actions, loss,
            reward_loss, image_loss, target_states, target_rewards, opt)

""" ###########################################################################"""
""" ###########################################################################"""
""" ###########################################################################"""
""" ###########################################################################"""
""" ###########################################################################"""
""" ###########################################################################"""

def blocks_head(x):
    #Common encoding
    ## 2 CNNs. 3x3 kernel, stride 2, 32 filters. ReLU non-linearity
    x = tf.layers.conv2d(x, 32, kernel_size=(3,3), strides = (2,2), activation=tf.nn.relu, name = 'L1')
    #x = tf.layers.conv2d(x, 32, kernel_size=(3,3), strides = (2,2), activation=tf.nn.relu)
    x = flatten(x)
    return x

def create_latentinverse_env_model(obs_shape, num_actions, num_pixels, num_rewards,
        should_summary=True, reward_coeff=1.0, beta = 1.0):
    width = obs_shape[0]
    height = obs_shape[1]
    depth = obs_shape[2]

    size = 256
    
    states = tf.placeholder(tf.float32, [None, width, height, depth])
    next_states = tf.placeholder(tf.float32, [None, width, height, depth])

    #onehot_actions = tf.placeholder(tf.float32, [None, width,
    #    height, num_actions])
    onehot_actions = tf.placeholder(tf.float32, [None, num_actions])

    batch_size = tf.shape(states)[0]
    #target_states = tf.placeholder(tf.uint8, [None])
    target_rewards = tf.placeholder(tf.uint8, [None])
    #inputs = tf.concat([states, onehot_actions], axis=-1)

    #pred rewards
    pred_rewards = target_rewards

    phi1 = blocks_head(states)
    with tf.variable_scope(tf.get_variable_scope(), reuse = True):
        phi2 = blocks_head(next_states)

    # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
    g = tf.concat([phi1, phi2],1)
    g = tf.nn.relu(linear(g, size, "g1", normalized_columns_initializer(0.01)))
    aindex = tf.argmax(onehot_actions, axis=1)  # aindex: [batch_size,]
    logits = linear(g, num_actions, "glast", normalized_columns_initializer(0.01))
    invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits = logits, labels = aindex), name="invloss")
    ainvprobs = tf.nn.softmax(logits, dim=-1)

    # forward model: f(phi1,asample) -> phi2
    # Note: no backprop to asample of policy: it is treated as fixed for predictor training
    f = tf.concat([phi1, onehot_actions],1)
    f = tf.nn.relu(linear(f, size, "f1", normalized_columns_initializer(0.01)))
    f = linear(f, phi1.get_shape()[1].value, "flast", normalized_columns_initializer(0.01))
    forwardloss = 0.5*tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
    #forwardloss = 0.5 * tf.reduce_mean(tf.sqrt(tf.abs(tf.subtract(f, phi2))), name='forwardloss')
    #forwardloss = cosineLoss(f, phi2, name='forwardloss')
    forwardloss = forwardloss*height*width

    #Loss is joint between forward and inverse model
    loss = forwardloss + beta*invloss

    opt = tf.train.AdamOptimizer().minimize(loss)

    # Tensorboard
    if should_summary:
        tf.summary.scalar('Loss', loss)
        tf.summary.scalar('Forward Loss', forwardloss)
        tf.summary.scalar('Inverse Loss', invloss)

    #Arguments here:
    #image is the predicted image, 
    #reward is the predicted reward
    #states

    return EnvModelDataInverse(f, pred_rewards, states, onehot_actions, loss,
            forwardloss, invloss, next_states, target_rewards, ainvprobs, opt)
    #Become:
    #return EnvModelDataInverse(imag_state, imag_reward, input_states, input_actions, loss, 
    #        reward_loss, image_loss, target_states, target_rewards, opt)

#def pred_act(env_model, s1, s2):
#    '''
#    returns action probability distribution predicted by inverse model
#        input: s1,s2: [h, w, ch]
#        output: ainvprobs: [ac_space]
#    '''
#    sess = tf.get_default_session()
#    return sess.run(env_model.ainvprobs, {self.s1: [s1], self.s2: [s2]})[0, :]

""" ###########################################################################"""
""" ###########################################################################"""
""" ###########################################################################"""
""" ###########################################################################"""
""" ###########################################################################"""
""" ###########################################################################"""

#Minigrid env
env_name = "MiniGrid-BlockMaze-v0"
def make_env(env_name):
    return lambda: gym_minigrid.wrappers.ImgObsWrapper(gym.make(env_name))

def play_games(actor_critic, envs, frames):
    states = envs.reset()
    for frame_idx in range(frames):
        #Reset when done...
        #if frame_idx % 200 == 0:
        #    print("Resetting")
        #    states = envs.reset()
        actions, _, _ = actor_critic.act(states)
        next_states, rewards, dones, _ = envs.step(actions)
        #print(rewards)
        #print(dones)
        #if np.any(rewards > 0):
        #    print("We have a winner")
        #if np.any(dones == True):
        #    print("We are done")
        yield frame_idx, states, actions, rewards, next_states, dones
        states = next_states

class EnvModelData(object):
    def __init__(self, imag_state, imag_reward, input_states, input_actions,
            loss, reward_loss, image_loss, target_states, target_rewards, opt):
        self.imag_state       = imag_state
        self.imag_reward      = imag_reward
        self.input_states     = input_states
        self.input_actions    = input_actions

        self.loss             = loss
        self.reward_loss      = reward_loss
        self.image_loss       = image_loss
        self.target_states    = target_states
        self.target_rewards   = target_rewards
        self.opt              = opt

class EnvModelDataInverse(object):
    def __init__(self, imag_phi, imag_reward, input_states, input_actions,
            loss, forwardloss, invloss, target_states, target_rewards, ainvprobs, opt):
        self.imag_phi         = imag_phi
        self.imag_reward      = imag_reward
        self.input_states     = input_states
        self.input_actions    = input_actions
        self.loss             = loss
        self.forwardloss      = forwardloss
        self.invloss          = invloss
        self.target_states    = target_states
        self.target_rewards   = target_rewards
        self.ainvprobs        = ainvprobs
        self.opt              = opt

if __name__ == '__main__':
    envs = [make_env(env_name) for i in range(N_ENVS)]
    envs = SubprocVecEnv(envs)

    ob_space = envs.observation_space.shape
    ac_space = envs.action_space
    num_actions = envs.action_space.n
    obs_shape = ob_space

    with tf.Session() as sess:

        #Interactive mode
        #sess = tf.Session()

        with tf.variable_scope('actor'):
            actor_critic = get_actor_critic(sess, N_ENVS, N_STEPS, ob_space, ac_space, CnnPolicy, should_summary=False)    
            actor_critic.load(A2C_WEIGHTS)

        obs_shape = ob_space
        with tf.variable_scope('env_model'):
            #env_model = create_env_model(ob_space, num_actions, num_pixels,
            #        len(mode_rewards[REWARD_MODE]))
            env_model = create_latentinverse_env_model(ob_space, num_actions, num_pixels,
                    len(mode_rewards[REWARD_MODE]))

        summary_op = tf.summary.merge_all()
        #This seems to overwrite parameters from actor model...
        #sess.run(tf.global_variables_initializer())
        initialize_uninitialized_vars(sess)

        losses = []
        all_rewards = []

        width = ob_space[0]
        height = ob_space[1]
        depth = ob_space[2]

        save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='env_model')
        saver = tf.train.Saver(var_list=save_vars)

        writer = tf.summary.FileWriter('./env_logs', graph=sess.graph)

        for frame_idx, states, actions, rewards, next_states, dones in tqdm(play_games(actor_critic, envs, NUM_UPDATES), total=NUM_UPDATES):
            #Interactive...
            #frame_idx, states, actions, rewards, next_states, dones = next(play_games(actor_critic, envs, NUM_UPDATES))
            
            #target_state = pix_to_target(next_states)
            target_reward = rewards_to_target(REWARD_MODE, rewards)
            onehot_actions = np.zeros((N_ENVS, num_actions))
            onehot_actions[range(N_ENVS), actions] = 1

            s, r, l, forw_loss, inv_loss, summary, _ = sess.run([
                env_model.imag_phi,
                env_model.imag_reward,
                env_model.loss,
                env_model.forwardloss,
                env_model.invloss,
                summary_op,
                env_model.opt], feed_dict={
                    env_model.input_states: states,
                    env_model.input_actions: onehot_actions,
                    env_model.target_states: next_states,
                    env_model.target_rewards: target_reward
                })

            if frame_idx % LOG_INTERVAL == 0:
                print('%i) %.5f, %.5f, %.5f' % (frame_idx, l, forw_loss, inv_loss))
            writer.add_summary(summary, frame_idx)

        saver.save(sess, 'weights/env_model_inverse.ckpt')
        print('Environment model saved!')

