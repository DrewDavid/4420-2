# Drew David

import gymnasium as gym # for environment
import numpy as np # for random actions
import matplotlib.pyplot as plt # for plotting eventual results

class QLearningAgent: # agent class
    def __init__(self, states = 48, actions = 4, alpha = 0.1, gamma = 0.99,
                 epsilon = 1.0, epsilon_decay = 0.995, min_epsilon = 0.01):
        # initializing hyperparameters
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = epsilon_decay

        # initializing q-table
        self.q_table = np.zeros((states, actions))

    def get_action(self, state, explore = True):
        # epsilon-greedy action-selection policy
        # explore = True is default
        # will be set to false for video of "trained agent"
        if explore and (np.random.random() < self.epsilon):
            # explore
            return np.random.randint(self.actions)
        else:
            # exploit
            return np.argmax(self.q_table[state])

    def q_learn(self, state, action, reward, next_state, done):
        # update Bellman equation (Q-learning)
        current_q = self.q_table[state, action] # Q(s,a)

        if done:
            target_q = reward # episode over, either by goal or cliff
        else:
            best_future_q = np.max(self.q_table[next_state]) # max_a'(Q(s',a'))
            target_q = reward + (self.gamma * best_future_q) # gamma * max_a...

        # actual value finally updates
        self.q_table[state, action] = current_q + self.alpha * (target_q - current_q)

    def decay_epsilon(self):
        # decay epsilon to lead to more early exploration and more late exploitation
        if self.epsilon > self.min_epsilon: # if epsilon hasn't already decayed to minimum
            self.epsilon *= self.decay_rate