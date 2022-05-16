from env import Connect4
import joblib
from agent import Agent

Agent1 = Agent(epsilon=0.1)
Agent2 = Agent(epsilon=0.1)

try:
    Q1 = joblib.load('q_table1.pkl')
    Agent1.Q = Q1
    Q2 = joblib.load('q_table2.pkl')
    Agent2.Q = Q2
    print("Loaded Q table")
except:
    print("No Q table found")

def player_input():
    col = int(input("Make your Selection(0-6):"))
    if col not in range(7):
        print("Invalid Selection")
        player_input()
    return col

def AI_input(state,p):
  while True:
    if p == 1:
      action = Agent2.take_action(state)
    else:
      action = Agent1.take_action(state)
    if env.is_valid_location(action):
      return action

env = Connect4()
while True:
  print("Choose Player")
  p = int(input())

  env.reset() 
  env.render()


  if p == 1:
    while not env.done:
      env.step(player_input())
      env.render()
      if env.done:
        break
      env.step(AI_input(env.state,p))
      env.render()
  if p == 2:
    while not env.done:
      env.step(AI_input(env.state,p))
      env.render()
      if env.done:
        break
      env.step(player_input())
      env.render()
  if p == 3:
    while not env.done:
      env.step(AI_input(env.state,1))
      env.render()
      if env.done:
        break
      env.step(AI_input(env.state,2))
      env.render()


  print(env.winner)
  print("Play Again? (Y/N)")
  if input() == 'N':
    break
