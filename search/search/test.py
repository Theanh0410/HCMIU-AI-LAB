from util import Queue  # Assuming Queue class is defined in util.py

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the goal.
    """
    # Initialize the frontier with the start state
    frontier = Queue()
    frontier.push((problem.getStartState(), [], []))  # (state, actions, visited nodes)

    # Initialize the explored set
    explored = set()

    while not frontier.isEmpty():
        state, actions, visited = frontier.pop()

        # If the state is the goal, return the actions that lead to it
        if problem.isGoalState(state):
            return actions

        # Skip this state if it has been explored
        if state not in explored:
            explored.add(state)

            # Add successors to the frontier
            for successor, action, step_cost in problem.getSuccessors(state):
                if successor not in visited and successor not in explored:
                    frontier.push((successor, actions + [action], visited + [state]))

    return []  # If no solution found

def depthFirstSearch(problem):
    # Initialize the frontier with the start state and an empty path
    frontier = Stack()  # Use Stack for DFS
    start_state = problem.getStartState()
    frontier.push((start_state, []))  # (state, path)
    
    # Set of visited nodes to avoid revisiting
    visited = set()

    while not frontier.isEmpty():
        # Get the current state and path
        state, path = frontier.pop()

        # If this state is the goal, return the path to it
        if problem.isGoalState(state):
            return path
        
        # If state is not visited, mark it as visited
        if state not in visited:
            visited.add(state)

            # Explore the successors (state, action, step_cost)
            for successor, action, _ in problem.getSuccessors(state):
                if successor not in visited:
                    # Add the new state to the frontier with the updated path
                    new_path = path + [action]
                    frontier.push((successor, new_path))

    return []  # Return empty list if no solution found

import util

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # Priority Queue with initial state
    frontier = util.PriorityQueue()
    start_state = problem.getStartState()
    frontier.push((start_state, [], 0), 0)  # (state, path, cost)
    
    # Dictionary to keep track of the cost to reach a state
    explored_cost = {}
    explored_cost[start_state] = 0
    
    # Keep track of visited nodes
    visited = set()

    # Number of expanded nodes
    expanded_nodes = 0

    while not frontier.isEmpty():
        # Pop the node with the lowest cost
        current_state, actions, current_cost = frontier.pop()

        # If goal state is reached, return the path and expanded nodes count
        if problem.isGoalState(current_state):
            print(f"Expanded nodes: {expanded_nodes}")
            return actions

        # If the state has been visited with a cheaper cost, skip it
        if current_state in visited:
            continue

        # Mark the current state as visited
        visited.add(current_state)
        expanded_nodes += 1

        # Expand the node and add successors to the frontier
        for successor, action, step_cost in problem.getSuccessors(current_state):
            new_cost = current_cost + step_cost
            if successor not in explored_cost or new_cost < explored_cost[successor]:
                explored_cost[successor] = new_cost
                new_actions = actions + [action]
                frontier.push((successor, new_actions, new_cost), new_cost)

    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    frontier = util.Queue()
    startState = problem.getStartState()
    frontier.push((startState, [], [])) #(state, path, visited nodes)
    
    visited = set()
    
    while not frontier.isEmpty():
        state, path, cost = frontier.pop()
        
        if problem.isGoalState(state):
            return path
            
        if state not in visited:
            visited.add(state)
            
            for successor, action, _ in problem.getSuccessors(state):
                if successor not in visited:
                    newPath = path + [action]
                    newCost = cost + [state]
                    frontier.push((successor, newPath, newCost))
                    
    
    util.raiseNotDefined()
    
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # Priority Queue with initial state
    frontier = util.PriorityQueue()
    start_state = problem.getStartState()
    frontier.push((start_state, [], 0), 0)  # (state, path, cost)
    
    # Dictionary to keep track of the cost to reach a state
    explored_cost = {}
    explored_cost[start_state] = 0
    
    # Keep track of visited nodes
    visited = set()

    # Number of expanded nodes
    expanded_nodes = 0

    while not frontier.isEmpty():
        # Pop the node with the lowest cost
        current_state, actions, current_cost = frontier.pop()

        # If goal state is reached, return the path and expanded nodes count
        if problem.isGoalState(current_state):
            print(f"Expanded nodes: {expanded_nodes}")
            return actions

        # If the state has been visited with a cheaper cost, skip it
        if current_state in visited:
            continue

        # Mark the current state as visited
        visited.add(current_state)
        expanded_nodes += 1

        # Expand the node and add successors to the frontier
        for successor, action, step_cost in problem.getSuccessors(current_state):
            new_cost = current_cost + step_cost
            if successor not in explored_cost or new_cost < explored_cost[successor]:
                explored_cost[successor] = new_cost
                new_actions = actions + [action]
                frontier.push((successor, new_actions, new_cost), new_cost)

    return []
