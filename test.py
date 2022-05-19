from env import Connect4
from agent import DQNAgent
from torch import load
Agent1 = DQNAgent(state_size=42,action_size=7,seed=0)
Agent2 = DQNAgent(state_size=42,action_size=7,seed=0)
env = Connect4()

try:
    Agent1.qnetwork_local.load_state_dict(load('q_dict1.pt'))
    print("Loaded Agent 1")
    Agent2.qnetwork_local.load_state_dict(load('q_dict2.pt'))
    print("Loaded Agent 2")
except:
    print("Agents not found")

def player_input():
    col = int(input("Make your Selection(0-6):"))
    if col not in range(7):
        print("Invalid Selection")
        player_input()
    return col

def AI_input(state,p):
  print("CPU Turn")
  while True:
    if p == 1:
      action = Agent2.take_action(state,eps=0.01)
    if p ==2:
      action = Agent1.take_action(state,eps=0.01)
    if env.is_valid_location(action):
      
      return action

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
      env.step(AI_input(env.board.flatten(),p))
      env.render()
  if p == 2:
    while not env.done:
      env.step(AI_input(env.board.flatten(),p))
      env.render()
      if env.done:
        break
      env.step(player_input())
      env.render()
  if p == 3:
    while not env.done:
      env.step(AI_input(env.board.flatten(),2))
      env.render()
      if env.done:
        break
      env.step(AI_input(env.board.flatten(),1))
      env.render()
  if p ==4:
      while not env.done:
          env.step(player_input())
          env.render()


  print(env.winner)
  print("Play Again? (Y/N)")
  if input() == 'N':
    break
