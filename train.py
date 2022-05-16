import joblib
import pandas as pd
from env import Connect4
from agent import Agent

#Create Envirionment and Agents
env = Connect4()

#Initially the agents are totally random

Agent1 = Agent(epsilon=1)
Agent2 = Agent(epsilon=1)

#For keeping track of progress
all_epochs = []
all_penalties = []
winners = []

#Try to load Q tables if they exist
try:
    Q1 = joblib.load('q_table1.pkl')
    Agent1.Q = Q1
    
    print("Loaded Q1 table")
except:
    print("Q1 table not found")

try:
    Q2 = joblib.load('q_table2.pkl')
    Agent2.Q = Q2
    print("Loaded Q2 table")

except:
    print("Q2 table not found")
#Begin Training
n_episodes = 100000
for i in range(1, n_episodes + 1):
    state = 0
    agent1_states = []
    agent2_states = []
    agent1_rewards = []
    agent2_rewards = []
    agent1_actions = []
    agent2_actions = []
    env.reset()
    penalties, reward, = 0, 0
    
    #Begin Game
    while not env.done:
        
        #Make the Agent try until it gets a valid move
        while True:
          action = Agent1.take_action(state)
          next_state, reward, done, info = env.step(action) 

          agent1_states.append(state)
          agent1_actions.append(action)
          agent1_rewards.append(reward)
          state = next_state
          
          if reward != -10:
            break
          penalties += 1

        
        #Break if the game is over
        if done:
          break

        #Make the Agent try until it gets a valid move
        while True:
          action = Agent2.take_action(state)
          next_state, reward, done, info = env.step(action) 

          agent2_states.append(state)
          agent2_actions.append(action)
          agent2_rewards.append(reward)
          state = next_state
          
          if reward != -10:
            break
          penalties += 1
        
        
    #Update the Q tables
    for j in range(0,len(agent1_states)-1):
      Agent1.update_Q(agent1_states[j],agent1_actions[j],agent1_rewards[j],agent1_states[j+1])
    for k in range(0,len(agent2_states)-1):
      Agent2.update_Q(agent2_states[k],agent2_actions[k],agent2_rewards[k],agent2_states[k+1])
    
    all_epochs.append(len(agent1_states)+len(agent2_states) -1)
    all_penalties.append(penalties)
    winners.append(info)
    if i % (n_episodes/100) == 0:
        Agent1.epsilon *= 0.99
        Agent2.epsilon *= 0.99
        if Agent1.epsilon < 0.1:
            Agent1.epsilon = 0.1
        if Agent2.epsilon < 0.1:
            Agent2.epsilon = 0.1
        print(f"Episode: {i}")
        env.render()
        print(env.winner)


print("Training finished.\n")

df = pd.DataFrame({'epochs': all_epochs,'penalties': all_penalties, 'winners': winners})
df.to_csv('results.csv',index=False)
joblib.dump(Agent1.Q, 'q_table1.pkl')
joblib.dump(Agent2.Q, 'q_table2.pkl')