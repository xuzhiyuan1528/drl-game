import datetime
from ple import PLE
from ple.games.flappybird import FlappyBird
import tensorflow as tf
import numpy as np
# The doc of Flappy Bird is at
# http://pygame-learning-environment.readthedocs.io/en/latest/user/games/flappybird.html.
from Agent.agent_dqn import DQNAgent
from Tools.explorer import Explorer
from Tools.replaybuffer import ReplayBuffer
from flag_bird import FLAGS

time_stamp = str(datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'))

STATE_FRAMES = getattr(FLAGS, 'frames')
SCREEN_WIDTH = getattr(FLAGS, 'screen_width')
SCREEN_HEIGHT = getattr(FLAGS, 'screen_height')
OBV_STEPS = getattr(FLAGS, 'observe_steps')
CKP_STEP = getattr(FLAGS, 'check_point_steps')

MAX_EP = getattr(FLAGS, 'episodes')
EP_STEPS = getattr(FLAGS, 'epochs')

DIM_STATE = [SCREEN_WIDTH, SCREEN_HEIGHT, STATE_FRAMES]
DIM_ACTION = getattr(FLAGS, 'dim_action')

LR = getattr(FLAGS, 'learning_rate')
TAU = getattr(FLAGS, 'tau')
GAMMA = getattr(FLAGS, 'gamma')

dir_sum = getattr(FLAGS, 'dir_sum').format(time_stamp)
DIR_MOD = getattr(FLAGS, 'dir_mod').format(time_stamp)

BUFFER_SIZE = getattr(FLAGS, 'size_buffer')
MINI_BATCH = getattr(FLAGS, 'mini_batch')

EPS_BEGIN = getattr(FLAGS, 'epsilon_begin')
EPS_END = getattr(FLAGS, 'epsilon_end')
EPS_STEPS = getattr(FLAGS, 'epsilon_steps')

playback_mode = False

class DqnBirdSyr():

    def __init__(self):
        env = FlappyBird()
        self._ple = PLE(env, fps=30, display_screen=False)
        self._ple.init()

        self._sess = tf.Session()
        self._agent = DQNAgent(self._sess, DIM_STATE, DIM_ACTION, LR, TAU)
        self._sess.run(tf.global_variables_initializer())
        self._agent.update_target_paras()

        self._saver = tf.train.Saver()
        self._replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self._explorer = Explorer(EPS_BEGIN, EPS_END, EPS_STEPS, playback_mode)

        self._steps = 0

    def start(self):
        for ep in range(MAX_EP):
            sum_reward = 0
            last_state = []

            for _ in range(STATE_FRAMES):
                last_state.append(self._ple.getScreenGrayscale())
            last_state = np.dstack(last_state)

            for step in range(EP_STEPS):

                q_value = self._agent.predict([last_state])[0]

                act_1_hot = self._explorer.get_action(q_value)
                act_index = np.argmax(act_1_hot)

                reward = self._ple.act(self._ple.getActionSet()[act_index])

                state = []
                for _ in range(STATE_FRAMES):
                    state.append(self._ple.getScreenGrayscale())
                state = np.dstack(state)

                done = False
                if self._ple.game_over() or step == EP_STEPS - 1:
                    done = True

                self._replay_buffer.add(last_state, act_1_hot, reward, state, done)

                if len(self._replay_buffer) > OBV_STEPS:
                    self._train()

                last_state = state
                sum_reward += reward
                if done:
                    print('| Step: %i' % self._steps,
                          '| Episode: %i' % ep,
                          '| Epoch: %i' % step,
                          '| Sum_Reward: %i' % sum_reward)
                    self._ple.reset_game()
                    break


    def _train(self):
        self._steps += 1
        batch_state, batch_action, batch_reward, batch_state_next, batch_done = \
            self._replay_buffer.sample_batch(MINI_BATCH)

        q_value = self._agent.predict(batch_state_next)
        max_q_value_index = np.argmax(q_value, axis=1)
        target_q_value = self._agent.predict_target(batch_state_next)
        double_q = target_q_value[range(len(target_q_value)), max_q_value_index]

        batch_y = []
        for r, q, d in zip(batch_reward, double_q, batch_done):
            if d:
                batch_y.append(r)
            else:
                batch_y.append(r + GAMMA * q)

        opt, loss = self._agent.train(batch_state, batch_action, batch_y)
        self._agent.update_target_paras()

        if not self._steps % CKP_STEP:
            self._saver.save(self._sess, DIR_MOD + '/net', global_step=self._steps)
            print('Mod saved!')


if __name__ == '__main__':
    dqn_bird = DqnBirdSyr()
    dqn_bird.start()
