import numpy as np


class QTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = {}

    def check_state_exist(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.] * len(self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = np.argmax(self.q_table[observation])
            action = self.actions[state_action]
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s1, done):
        self.check_state_exist(s1)
        reward_decay = self.gamma if not done else 0
        q_target = r + reward_decay * self.q_table[s1][self.actions.index(a)]
        q_predict = self.q_table[s][self.actions.index(a)]
        self.q_table[s][self.actions.index(a)] = (1 - self.lr) * q_predict + self.lr * q_target
