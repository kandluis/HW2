# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, search, searchAgents, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.

        Evaluation Function:
          The value of a state depends on two variables. The minimum manhattanDistance
          to the closest food item, and the number of food items.

          For the same number of food items, a lower distance is better.
          Lower number of food items is always better than more food items.

          To combine these two metrics so the above holds, we think of the
          returned value as a decimal with (food_score).(distance_score).

          Therefore, we first bound the distance_score into the interval [0,1)
          using the bounded metric 1/(1 + d). Note that larger x imply a lower
          score, which is what we want.

          The food items are already guaranteed to be integers, so we can simply
          take their count and negate it, as -d. This gives use a value in the
          set of natural numbers, and additionally, more food implies a lower score.

          However, we also consider a few edge cases. We never want to die, some
          given our position, any position that puts us in the vicinity of a ghost
          is a position we wish to avoid. Given that the grid is open and that there
          is only one ghost, we can guarantee survival by staying at least two
          steps away from non-scared ghost.
        """
        def ghostBuffer(ghost):
            x, y = ghost.getPosition()
            return [(x - 1, y), (x, y - 1), (x + 1, y), (x, y + 1), (x, y)]

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        dists = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        enemyGhosts = [ghost for ghost in newGhostStates if ghost.scaredTimer == 0]
        deathPos = [pos for ghost in enemyGhosts for pos in ghostBuffer(ghost)]
        minDist = min(dists) if dists else 0

        # Try to never die.
        if newPos in deathPos:
            return -float("Inf")
        # A slightly complicated evaluation function.
        else:
            return (1 + minDist)**(-1) - len(newFood.asList())

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def terminal(self, state):
        """
        Returns true if the state is a terminal state. We define a state to be
        terminal if the default agent has no legal moves.
        """
        return state.getLegalActions() == []

    def isMaxAgent(self, agent):
        """
        Returns true if the agent is the max agent. The assumption is that agent
        0 is the only max agent.
        """
        return agent == 0

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        nagents = gameState.getNumAgents()
        def goalFunction(agentFun, actionList):
            """
            Takes as input a list of (action, score) tuples and returns the best
            action as defined by agentFun.
            """
            return agentFun(actionList, key=lambda (action, score): score)

        def miniMax(state, depth):
            # We assume that there exists only one max agent, and a single ply
            # consists of a turn by max and all opponents.
            agentIndex = depth % nagents
            if self.terminal(state) or int(depth / nagents) >= self.depth:
                return (None, self.evaluationFunction(state))

            # Acquire the value of successor states and their actions.
            agentActionCost = [(action, miniMax(state.generateSuccessor(agentIndex, action), depth + 1)[1]) for action in state.getLegalActions(agentIndex)]
            agentFun = max if self.isMaxAgent(agentIndex) else min

            return goalFunction(agentFun, agentActionCost)

        return miniMax(gameState, 0)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        nagents = gameState.getNumAgents()
        def abMiniMax(state, depth, alpha, beta):
            """
            Implementation of MiniMax with alpha-beta pruning
            """
            agentIndex = depth % nagents
            if self.terminal(state) or int(depth / nagents) >= self.depth:
                return (None, self.evaluationFunction(state))

            if self.isMaxAgent(agentIndex):
                # Max's turn!
                value = -float("Inf")
                best_action = None
                for action in state.getLegalActions(agentIndex):
                    res = abMiniMax(state.generateSuccessor(agentIndex, action), depth + 1, alpha, beta)[1]
                    if res > value:
                        value = res
                        best_action = action
                    if value > beta:
                        return (action, value)
                    alpha = max(alpha, value)
                return (best_action, value)
            else:
                value = float("Inf")
                best_action = None
                for action in state.getLegalActions(agentIndex):
                    res = abMiniMax(state.generateSuccessor(agentIndex, action), depth + 1, alpha, beta)[1]
                    if res < value:
                        value = res
                        best_action = action
                    if value < alpha:
                        return (action, value)
                    beta = min(beta, value)
                return (best_action, value)

        return abMiniMax(gameState, 0, -float("Inf"), float("Inf"))[0]



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        nagents = gameState.getNumAgents()
        def expectiMax(state, depth):
            """
            Implementation of the ExpectiMax search algorithm.
            The min player is taken to be a chance node.
            """
            agentIndex = depth % nagents
            if self.terminal(state) or int(depth / nagents) >= self.depth:
                return (None, self.evaluationFunction(state))

            agentActionCost = [(action, expectiMax(state.generateSuccessor(agentIndex, action), depth + 1)[1]) for action in state.getLegalActions(agentIndex)]

            if self.isMaxAgent(agentIndex):
                return max(agentActionCost, key=lambda (action, score): score)
            else:  # We have a ghost, so chance node.
                return (None, float(sum([score for (_, score) in agentActionCost])) / float(len(agentActionCost)))

        return expectiMax(gameState, 0)[0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
        The evaluation function is similar to that used for q1.

        We have a few criteria for evaluating a state. The minimum
        manhattanDistance to the nearest non-scared ghost, the score of the
        game, and number of food items left on the board.

        If the minimum distance to the nearest non-scared ghost is 0, we
        return -infinity. This is to make sure that death is always penalized
        more heavily than anything else. One drawback of this approach is that
        death states become indistinguishable. However, it does guarantee we
        never select a path to a death state if one exists.

        Next, we calculated a heuristic for the expected score of the game.
        States with higher expected scores are favored. We do this by taking the
        current score and subtracting the number of food items left to acquire.
        This is a simple heuristic for what our game score will be.

        Lastly, we combine the above with the bounded metric on the nearest
        food items. Similar to q1, this implies that lower score states are
        favored, and out of those with the same expected score, states with a
        closer food item are favored.

        On the given tests, we achieve an average score of 1086.1.
    """
    # Get minimum distance to nearest food
    pos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    dists = [util.manhattanDistance(pos, food) for food in foodPos]
    minDist = min(dists) if dists else 0

    # Get ghost positions.
    newGhostStates = currentGameState.getGhostStates()
    enemyGhosts = [util.manhattanDistance(pos, ghost.getPosition()) for ghost in newGhostStates if ghost.scaredTimer == 0]

    minGhost = min(enemyGhosts) if enemyGhosts else float("Inf")
    # Avoid deaths at all cost.
    if minGhost == 0:
        return -float("Inf")

    score = currentGameState.getScore()
    return (score - len(foodPos)) + ((1+minDist)**(-1))

# Abbreviation
better = betterEvaluationFunction

