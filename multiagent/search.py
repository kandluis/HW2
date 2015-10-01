# search.py

# Authors:
# @Kevin Eskici (keskici@college.harvard.edu)
# @Luis A. Perez (luisperez@college.harvard.edu)

# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def reconstructPath(actionSet, toState):
    """
    Reconstructs the path. Assumes that actionSet maps toState => (action, prevState)
    tuples.
    """
    try:
        action, prevState = actionSet[toState]
        return reconstructPath(actionSet, prevState) + [action]
    except KeyError:
        return []

def genericSearch(problem, frontier):
    """
    Generic search algorithm which utilizes the given frontier to determine
    which states to explore next.

    We mark a state as explored only after full expansion, which guarantees
    we have found the best path (according to the frontier) to that state.

    The frontier can either update the best path to a state or duplicate paths,
    but it must decide which is better and pop that off first.
    Future paths that lead to the same state (in the case the frontier duplicates)
    are ignored because we mark that state as fully explored and skip it in
    our algorithm.

    The code is intended to work with BFS (using queue), DFS (using stack),
    and A*/UCS (using priorityQueue). No other algos were tested, but it should
    work in general with however the frontier defines "best" path as long as
    the guarantee exists that a path to a state cannot be improved by revisiting
    that state (ie, no negative edge weights).

    We also optimize the function to avoid keeping track of paths entirely,
    instead only keeping track of path cost and a list of prev pointers.

    Why go to all this hard work? To make sure the autograder gets the expeted
    results.
    """
    # Frontier stores (cost, state, (action, fromState)) tuples.
    frontier.push((0, problem.getStartState(), (None, None)))
    explored = set()
    # Maps state => (action, fromState) tuples so we can reconstruct a path.
    prev = {}
    while not frontier.isEmpty():
        (cost, state, (action, fromState)) = frontier.pop()
        if state in explored:
            continue
        prev[state] = (action, fromState)
        if problem.isGoalState(state):
            break
        for (successor, action, stepCost) in problem.getSuccessors(state):
            frontier.push((cost + stepCost, successor, (action, state)))
        explored.add(state)

    # Don't want to return initial "None" action.
    return reconstructPath(prev, state)[1:]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    return genericSearch(problem, util.Stack())

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    "*** YOUR CODE HERE ***"
    return genericSearch(problem, util.Queue())

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    return genericSearch(problem, util.PriorityQueueWithFunction(lambda (cost, state, _): cost))

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"
    return genericSearch(problem, util.PriorityQueueWithFunction(
        lambda (cost, state, _): cost + heuristic(state, problem)))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
