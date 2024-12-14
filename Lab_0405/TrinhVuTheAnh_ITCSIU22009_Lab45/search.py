"""
In search.py, you will implement Backtracking and AC3 searching algorithms
for solving Sudoku problem which is called by sudoku.py
"""

from csp import *
from copy import deepcopy
import util

def Backtracking_Search(csp):
    """
    Backtracking search initializes the assignment and 
    starts the recursive backtracking process.
    """
    "***YOUR CODE HERE ***"
    assignment = {}
    return Recursive_Backtracking(assignment, csp)

def Recursive_Backtracking(assignment, csp):
    """
    The recursive function that assigns values using backtracking.
    """
    "***YOUR CODE HERE ***"
    if isComplete(assignment):
        return assignment

    var = Select_Unassigned_Variables(assignment, csp)

    for value in Order_Domain_Values(var, assignment, csp):
        if isConsistent(var, value, assignment, csp):
            csp_copy = deepcopy(csp)

            assignment[var] = value
            forward_checking(csp, assignment, var, value)

            result = Recursive_Backtracking(assignment, csp)
            if result:
                return result

            assignment.pop(var)
            csp = csp_copy

    return None

def AC3(csp):
    """
    AC-3 algorithm for arc consistency
    """
    queue = [(x, y) for x in csp.variables for y in csp.peers[x]]
    while queue:
        (var, neighbor) = queue.pop(0)
        if revise(csp, var, neighbor):
            if len(csp.values[var]) == 0:
                return False
            for peer in csp.peers[var]:
                if peer != neighbor:
                    queue.append((peer, var))
    return True

def revise(csp, var, neighbor):
    """
    Revise the domain of var to ensure arc consistency with neighbor
    """
    revised = False
    for value in csp.values[var]:
        if not any(value != other for other in csp.values[neighbor]):
            csp.values[var] = csp.values[var].replace(value, "")
            revised = True
    return revised

def Inference(assignment, inferences, csp, var, value):
    """
    Forward checking using concept of Inferences
    """

    inferences[var] = value

    for neighbor in csp.peers[var]:
        if neighbor not in assignment and value in csp.values[neighbor]:
            if len(csp.values[neighbor]) == 1:
                return "FAILURE"

            remaining = csp.values[neighbor] = csp.values[neighbor].replace(value, "")

            if len(remaining) == 1:
                flag = Inference(assignment, inferences, csp, neighbor, remaining)
                if flag == "FAILURE":
                    return "FAILURE"

    return inferences

def Order_Domain_Values(var, assignment, csp):
    """
    Returns string of values of given variable
    """
    return csp.values[var]

def Select_Unassigned_Variables(assignment, csp):
    """
    Selects new variable to be assigned using minimum remaining value (MRV)
    """
    unassigned_variables = dict((squares, len(csp.values[squares])) for squares in csp.values if squares not in assignment.keys())
    mrv = min(unassigned_variables, key=unassigned_variables.get)
    return mrv

def isComplete(assignment):
    """
    Check if assignment is complete
    """
    return set(assignment.keys()) == set(squares)

def isConsistent(var, value, assignment, csp):
    """
    Check if assignment is consistent
    """
    for neighbor in csp.peers[var]:
        if neighbor in assignment.keys() and assignment[neighbor] == value:
            return False
    return True

def forward_checking(csp, assignment, var, value):
    csp.values[var] = value
    for neighbor in csp.peers[var]:
        csp.values[neighbor] = csp.values[neighbor].replace(value, '')

def display(values):
    """
    Display the solved sudoku on screen
    """
    for row in rows:
        if row in 'DG':
            print("-------------------------------------------")
        for col in cols:
            if col in '47':
                print(' | ', values[row + col], ' ', end=' ')
            else:
                print(values[row + col], ' ', end=' ')
        print(end='\n')

def write(values):
    """
    Write the string output of solved sudoku to file
    """
    output = ""
    for variable in squares:
        output = output + values[variable]
    return output