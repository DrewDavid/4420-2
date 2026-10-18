# Drew David

import gymnasium as gym # for environment
import numpy as np # for random actions and help plotting
import matplotlib.pyplot as plt # for plotting eventual results

class QLearningAgent: # agent class
    def __init__(self, states, actions, alpha = 0.1, gamma = 0.99,
                 epsilon = 1.0, epsilon_decay = 0.995, min_epsilon = 0.01):
        # initializing hyperparameters
        self.states = states
        self.actions = actions
        self.alpha = alpha # (0.1, 0.5, 0.9)
        self.gamma = gamma # (0.99, 0.90, 0.50)
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

def train(env, agent, episodes, learn = True):
    # training loop
    all_rewards = []

    for ep in range(episodes): # loops for all episodes
        state, _ = env.reset() # reset episode for new run
        # info isn't needed, replaced with _
        done = False
        episode_reward = 0 # reset episode reward

        while not done: # loops for entire episode
            action = agent.get_action(state, learn) # choose action
            next_state, reward, terminated, truncated, _ = env.step(action) # perform action
            done = terminated or truncated

            # only performs q-learning if not doing the 30-episode trained evaluation
            if learn:
                agent.q_learn(state, action, reward, next_state, done) # learn

            state = next_state # increment state
            episode_reward += reward # add to episode reward

        agent.decay_epsilon()
        all_rewards.append(episode_reward) # compile list of rewards for later graphing

    print("Training done.")
    return all_rewards

def graph(rewards, moving_avg = True):
    # plot baseline
    plt.plot(rewards, alpha = 0.3, label='Reward per Episode')

    # calculate moving average to smooth graph
    if moving_avg:
        window = 10
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode = 'valid')
        plt.plot(range(window - 1, len(rewards)), moving_avg, label = f'{window}-Episode Moving Average')

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Training on Cliff Walking")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    environment = gym.make("CliffWalking-v1")
    total_states = environment.observation_space.n
    total_actions = environment.action_space.n
    cliff_agent = QLearningAgent(states = total_states, actions = total_actions)
    training_rewards = train(env = environment, agent = cliff_agent, episodes=300)
    graph(training_rewards)
    evaluation = input('Would you like to evaluate the final trained policy over 30 episodes? (y/n)')

    if evaluation == 'y':
        training_rewards = train(env = environment, agent = cliff_agent, episodes = 30, learn = False)
        graph(training_rewards, moving_avg = False)

    environment.close()