from game import Agent
from game import Directions
import random

class DumbAgent(Agent) :
  "An agent that goes East until it can't"
  def getAction(self, state):
    "The agent receives a GameState (defined in pacman.py)"
    print("Location: ", state.getPacmanPosition())
    print("Actions available: ", state.getLegalPacmanActions())
    if Directions.EAST in state.getLegalPacmanActions():
      print("Going East.")
      return Directions.EAST
    else:
      print("Stopping.")
      return Directions.STOP
    
class RandomAgent(Agent) :
  def getAction(self, state):
    print("Location", state.getPacmanPosition())
    print("Actions available: ", state.getLegalPacmanActions())
    
    legalActions = state.getLegalPacmanActions()
    
    if legalActions:
      direction = random.choice(legalActions)
      print(f"Going {direction}.")
      return direction
    else:
      print("Stopping.")
      return Directions.STOP

    
class BetterRandomAgent(Agent) :
  def getAction(self, state):
    print("Location", state.getPacmanPosition())
    print("Actions available: ", state.getLegalPacmanActions())
    
    legalActions = state.getLegalPacmanActions()
    
    if Directions.STOP in legalActions:
      legalActions.remove(Directions.STOP)
    
    if legalActions:
      direction = random.choice(legalActions)
      print(f"Going {direction}.")
      return direction
    else:
      print("Stopping.")
      return Directions.STOP
    
    
class ReflexAgent() :
  def getAction(self, state):
    print("Location", state.getPacmanPosition())
    print("Actions available: ", state.getLegalPacmanActions())
    
    legalActions = state.getLegalPacmanActions()
    foodPositions = state.getFood().asList()

    for action in legalActions:
      successor = state.generatePacmanSuccessor(action)
      newPosition = successor.getPacmanPosition()
      if newPosition in foodPositions:
        print(f"Go to {action}")
        return action
    
    if Directions.STOP in legalActions:
      legalActions.remove(Directions.STOP)

    if legalActions:
      direction = random.choice(legalActions)
      print(f"Going {direction}.")
      return direction
    else:
      print("Stopping.")
      return Directions.STOP
    
      
      
    
    
    
    
    