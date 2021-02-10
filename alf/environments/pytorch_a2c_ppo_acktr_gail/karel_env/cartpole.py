import numpy as np


class CartPole_World(object):

    def __init__(self, env):
        self.env = env

        self.s = self.env.reset()
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.r_h = []
        self.d_h = []
        self.program_reward = 0.0
        self.done = False
        self.info = {}

    def clear_history(self):
        self.s = self.env.reset()
        self.s_h = [self.s.copy()]
        self.a_h = []
        self.r_h = []
        self.d_h = []
        self.program_reward = 0.0
        self.done = False
        self.info = {}

    def add_to_history(self, state, a_idx, reward, done):
        self.s_h.append(state)
        self.a_h.append(a_idx)
        self.r_h.append(reward)
        self.d_h.append(done)
        p_v = self.get_perception_vector()
        self.p_v_h.append(p_v.copy())
        self.program_reward += reward

    ###################################
    ###    Perception Primitives    ###
    ###################################

    def pole_pos_left(self):
        return True

    def pole_pos_center(self):
        return False

    def pole_pos_right(self):
        return False

    def cart_pos_left(self):
        return True

    def cart_pos_center(self):
        return False

    def cart_pos_right(self):
        return False

    def pole_vel_fast(self):
        return False

    def pole_vel_med(self):
        return False

    def pole_vel_slow(self):
        return True

    def cart_vel_fast(self):
        return False

    def cart_vel_med(self):
        return False

    def cart_vel_slow(self):
        return True

    def get_perception_vector(self):
        vec = [self.pole_pos_left(), self.pole_pos_center(), self.pole_pos_right(), self.cart_pos_left(),
               self.cart_pos_center(), self.cart_pos_right(), self.pole_vel_fast(), self.pole_vel_med(),
               self.pole_vel_slow(), self.cart_vel_fast(), self.cart_vel_med(), self.cart_vel_slow()]
        return np.array(vec)

    ###################################
    ###       State Transition      ###
    ###################################
    # given a state and a action, return the next state
    def state_transition(self, a):
        a_idx = np.argmax(a)
        state, reward, done, info = self.env.step(a_idx)
        self.add_to_history(state, a_idx, reward, done, info)
        self.s = state
        self.done = done
        self.info = info
        return
