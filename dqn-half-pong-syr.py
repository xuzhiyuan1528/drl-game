import datetime
from collections import deque

import cv2
import tensorflow as tf
import numpy as np
import os

from Agent.agent_dqn import DQNAgent
from Env.pygame_player import PyGamePlayer
from Tools.explorer import Explorer
from Tools.replaybuffer import ReplayBuffer
from Tools.summary import Summary
from flag_pong import FLAGS
from pygame.constants import K_DOWN, K_UP

time_stamp = str(datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'))

STATE_FRAMES = getattr(FLAGS, 'frames')
SCREEN_WIDTH = getattr(FLAGS, 'screen_width')
SCREEN_HEIGHT = getattr(FLAGS, 'screen_height')
OBV_STEPS = getattr(FLAGS, 'observe_steps')
CKP_STEP = getattr(FLAGS, 'check_point_steps')
EP_STEPS = getattr(FLAGS, 'epochs')

DIM_STATE = [SCREEN_WIDTH, SCREEN_HEIGHT, STATE_FRAMES]
DIM_ACTION = getattr(FLAGS, 'dim_action')

LR = getattr(FLAGS, 'learning_rate')
TAU = getattr(FLAGS, 'tau')
GAMMA = getattr(FLAGS, 'gamma')

DIR_SUM = getattr(FLAGS, 'dir_sum').format(time_stamp)
DIR_MOD = getattr(FLAGS, 'dir_mod').format(time_stamp)

BUFFER_SIZE = getattr(FLAGS, 'size_buffer')
MINI_BATCH = getattr(FLAGS, 'mini_batch')

EPS_BEGIN = getattr(FLAGS, 'epsilon_begin')
EPS_END = getattr(FLAGS, 'epsilon_end')
EPS_STEPS = getattr(FLAGS, 'epsilon_steps')

DISPLAY = getattr(FLAGS, 'display')

class DqnHalfPongSyr(PyGamePlayer):
    def __init__(self, playback_mode, mod=None):
        self._playback_mode = playback_mode
        self._last_reward = 0
        super(DqnHalfPongSyr, self).__init__(force_game_fps=8, run_real_time=playback_mode)

        self._last_state = None
        self._last_action = np.zeros(DIM_ACTION)
        self._last_action[1] = 1

        self.sess = tf.Session()
        self.agent = DQNAgent(self.sess, DIM_STATE, DIM_ACTION, LR, TAU, net_name='cnn_pong')
        self.sess.run(tf.global_variables_initializer())
        self.agent.update_target_paras()
        self.saver = tf.train.Saver()

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.explorer = Explorer(EPS_BEGIN, EPS_END, EPS_STEPS, playback_mode)
        self.summary = Summary(self.sess, DIR_SUM)

        self.summary.add_variable(tf.Variable(0.), 'reward')
        self.summary.add_variable(tf.Variable(0.), 'loss')
        self.summary.build()
        self.summary.write_variables(FLAGS)

        self._steps = 0
        self._sum_reward = 0
        self._dif_reward = deque(maxlen=EP_STEPS)

        if mod and os.path.exists(FLAGS.dir_mod.format(mod)):
            checkpoint = tf.train.get_checkpoint_state(FLAGS.dir_mod.format(mod))
            self.saver.restore(self.sess, save_path=checkpoint.model_checkpoint_path)
            print("Loaded checkpoints {0}".format(checkpoint.model_checkpoint_path))

    def get_keys_pressed(self, screen_array, feedback, terminal):
        _, screen_binary = cv2.threshold(cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY), 1, 255,
                                         cv2.THRESH_BINARY)

        if self._last_state is None:
            self._last_state = np.stack(tuple(screen_binary for _ in range(STATE_FRAMES)), axis=2)
            return DqnHalfPongSyr._key_presses_from_action(self._last_action)

        screen_binary = np.reshape(screen_binary, (SCREEN_WIDTH, SCREEN_HEIGHT, 1))
        current_state = np.append(self._last_state[:, :, 1:], screen_binary, axis=2)

        if not self._playback_mode:
            self.replay_buffer.add(self._last_state, self._last_action, feedback, current_state, terminal)
            if len(self.replay_buffer) > OBV_STEPS:
                loss = self._train()
                self._sum_reward += feedback
                if feedback != 0.0:
                    self._dif_reward.append(feedback)
                if not self._steps % EP_STEPS:
                    print('| Step: %i' % self._steps,
                          '| Epoch: %i' % (self._steps / EP_STEPS),
                          '| Sum_Reward: %i' % self._sum_reward,
                          '| Dif_Reward: %.4f' % (sum(self._dif_reward) / len(self._dif_reward)))
                    if not self._steps % (EP_STEPS * 10):
                        self.summary.run(feed_dict={
                            'loss': loss,
                            'reward': self._sum_reward})
                    self._sum_reward = 0

        self._last_state = current_state
        self._last_action = self._get_action()

        return DqnHalfPongSyr._key_presses_from_action(self._last_action)

    def _get_action(self):
        target_q = self.agent.predict([self._last_state])[0]
        return self.explorer.get_action(target_q)

    def _train(self):
        self._steps += 1
        batch_state, batch_action, batch_reward, batch_state_next, batch_done = \
            self.replay_buffer.sample_batch(MINI_BATCH)

        q_value = self.agent.predict(batch_state_next)
        max_q_value_index = np.argmax(q_value, axis=1)
        target_q_value = self.agent.predict_target(batch_state_next)
        double_q = target_q_value[range(len(target_q_value)), max_q_value_index]

        batch_y = []
        for r, q, d in zip(batch_reward, double_q, batch_done):
            if d:
                batch_y.append(r)
            else:
                batch_y.append(r + GAMMA * q)

        opt, loss = self.agent.train(batch_state, batch_action, batch_y)
        self.agent.update_target_paras()

        if not self._steps % CKP_STEP:
            self.saver.save(self.sess, DIR_MOD + '/net', global_step=self._steps)
            print('Mod saved!')

        return loss


    def get_feedback(self):
        from Env.games.half_pong import score

        # get the difference in score between this and the last run
        score_change = (score - self._last_reward)
        self._last_reward = score

        return float(score_change), score_change == -1

    @staticmethod
    def _key_presses_from_action(action_set):
        if action_set[0] == 1:
            return [K_DOWN]
        elif action_set[1] == 1:
            return []
        elif action_set[2] == 1:
            return [K_UP]
        raise Exception("Unexpected action")

    def start(self):
        super(DqnHalfPongSyr, self).start()

        from Env.games.half_pong import run
        run(screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT)


if __name__ == '__main__':
    dqn = DqnHalfPongSyr(playback_mode=False, mod=None)
    dqn.start()
