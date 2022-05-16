import numpy as np

class Agent():
  def __init__(self,epsilon = 0.1,alpha = 0.1,gamma = 0.9):
    #self.state_memory = np.zeros(mem_length)
    #self.observation_memory = np.zeros(mem_length)
    #self.reward_memory = np.zeros(mem_length)
    self.Q = {}
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma

  def take_action(self,state):
    if np.random.random() < self.epsilon:
        return np.random.randint(0,7)
    else:
        self.in_Q(state)
        return np.argmax(self.Q[state])

  def update_Q(self,state,action,reward,next_state):
    self.in_Q(state)
    self.in_Q(next_state)
    self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state][:]) - self.Q[state][action])

  def in_Q(self,state):
      try:
          self.Q[state][0]
      except KeyError as e:
          self.Q[state] = np.zeros(7)
          

