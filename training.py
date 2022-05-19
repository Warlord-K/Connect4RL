from agent import DQNAgent
from collections import deque
from env import Connect4
import torch
import pandas as pd
Agent = DQNAgent(state_size=42,action_size=7,seed=0)
env = Connect4()

try:
    Agent.qnetwork_local.load_state_dict(torch.load('q_dict.pt'))
    print("Loaded Agent")
except:
    print("Agent not found")


def train(n_episodes= 100, eps_start=1.0, eps_end = 0.01,eps_decay=0.996):
    winners = deque(maxlen=10000)
    epsilon = eps_start
    for episode in range(1,n_episodes+1):
        
        states = deque(maxlen=10)
        env.reset()
        state = env.board.flatten()
        while not env.done:
            action = Agent.take_action(state,epsilon)
            while not env.is_valid_location(action):
                action = Agent.take_action(state,epsilon)

            next_state, reward, done, winner = env.step(action)
            states.append(state)
            state = next_state
            try:
                Agent.step(states[-2],action,reward,state,done)
            except IndexError as e:
                pass
        winners.append(winner)
        Agent.step(states[-2],action,reward,state,done)
        if episode %100 == 0:
            epsilon = max(epsilon*eps_decay,eps_end)
            print(f"Episode: {episode}")
            env.render()
            print(env.winner)
    #pd.DataFrame(winners,columns = "Winner").to_csv("winners.csv",index = False)
    torch.save(Agent.qnetwork_local.state_dict(),'q_dict.pt')

for i in range(10):
    train(n_episodes=10000)