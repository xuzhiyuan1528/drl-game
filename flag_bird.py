import tensorflow as tf
import os
from os.path import join as pjoin

def home_out(path):
    full_path = pjoin('Res', 'Flappy-Bird', path)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    return full_path

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('display', False, 'display screen')

flags.DEFINE_integer('random_seed', 8, "seed for random generation")

flags.DEFINE_integer('frames', 4, 'the frames for each state')
flags.DEFINE_integer('screen_width', 288, 'the width for screen')
flags.DEFINE_integer('screen_height', 512, 'the height for screen')
flags.DEFINE_integer('dim_action', 2, 'the dimension of action')

flags.DEFINE_integer('size_buffer', 50000, 'the size of replay buffer')
flags.DEFINE_integer('mini_batch', 32, "size of mini batch")

flags.DEFINE_float('epsilon_begin', 0.1, 'epsilon greedy in the beginning')
flags.DEFINE_float('epsilon_end', 0.00001, 'epsilon greedy in the end')
flags.DEFINE_integer('observe_steps', 10000, 'steps for observation')
flags.DEFINE_integer('epsilon_steps', 500000, 'steps for epsilon greedy explore')

flags.DEFINE_integer('check_point_steps', 10000, 'steps for save model')

flags.DEFINE_integer('episodes', 20000, "training episode")
flags.DEFINE_integer('epochs', 2000, 'training epochs for each episode')

flags.DEFINE_float('gamma', 0.99, "discount value for reward")

flags.DEFINE_float('learning_rate', 0.000001, "learning rate for DQN")
flags.DEFINE_float('tau', 0.001, "tau for target network update")

flags.DEFINE_string('dir_sum', home_out('sum') + '/{0}', "directory to store the tf summary")
flags.DEFINE_string('dir_mod', home_out('mod') + '/{0}', 'the path of saved models')
