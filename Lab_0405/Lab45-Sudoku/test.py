class csp:
    # INITIALIZING THE CSP
    def __init__(self, domain=digits, grid=""):
        """
        Unitlist consists of the 27 lists of peers
        Units is a dictionary consisting of the keys and the corresponding lists of peers
        Peers is a dictionary consisting of the 81 keys and the corresponding set of 27 peers
        Constraints denote the various all-different constraints between the variables
        """
        self.variables = squares  # List of all cell identifiers (e.g., 'A1', 'A2', ..., 'I9')
        self.values = {v: digits for v in self.variables}  # Initialize all cells with possible digits (1-9)

        if grid: 
            self.values = self.getDict(grid)  # Populate initial values from the grid input
            
        # Create unitList consisting of all rows, columns, and 3x3 boxes
        unitList = (
            [cross(rows, c) for c in cols] +  # Rows
            [cross(r, cols) for r in rows] +  # Columns
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]  # 3x3 boxes
        )
        
        # Create units dictionary: mapping each variable to its corresponding units
        self.units = {s: [u for u in unitList if s in u] for s in self.variables}
        
        # Create peers dictionary: mapping each variable to its corresponding peers
        self.peers = {s: set(sum(self.units[s], [])) - {s} for s in self.variables}

    def getDict(self, grid=""):
        """
        Getting the string as input and returning the corresponding dictionary
        """
        i = 0
        values = dict()
        for cell in self.variables:
            if grid[i] != '0':
                values[cell] = grid[i]  # Assign the digit if it's not '0'
            else:
                values[cell] = digits  # Assign the full set of digits if it's '0'
            i += 1
        return values
    
    
def Recursive_Backtracking(assignment, csp):
    """
    The recursive function that assigns values using backtracking.
    """
    # Check if the assignment is complete
    if isComplete(assignment):
        return assignment

    # Select an unassigned variable using the Minimum Remaining Value (MRV) heuristic
    var = Select_Unassigned_Variables(assignment, csp)

    # Iterate over the domain values for the selected variable
    for value in Order_Domain_Values(var, assignment, csp):
        # Check if assigning the value is consistent with the current assignment
        if isConsistent(var, value, assignment, csp):
            # Make a copy of the CSP to track changes
            csp_copy = deepcopy(csp)

            # Assign the value
            assignment[var] = value
            forward_checking(csp, assignment, var, value)

            # Recurse with the updated assignment
            result = Recursive_Backtracking(assignment, csp)
            if result:
                return result

            # If recursion fails, undo the assignment
            assignment.pop(var)
            csp = csp_copy

    # Return failure if no value works
    return None


from csp import *
from copy import deepcopy
import util


def Backtracking_Search(csp):
    """
    Backtracking search initializes the initial assignment
    and calls the recursive backtrack function
    """
    assignment = {}
    inferences = {}
    return Recursive_Backtracking(assignment, csp)


def Recursive_Backtracking(assignment, csp):
    """
    The recursive function which assigns value using backtracking
    """
    if isComplete(assignment):
        return assignment

    var = Select_Unassigned_Variables(assignment, csp)  # Select unassigned variable
    for value in Order_Domain_Values(var, assignment, csp):  # Order domain values for the variable
        if isConsistent(var, value, assignment, csp):  # Check if the value is consistent
            assignment[var] = value
            inferences = {}
            forward_checking(csp, assignment, var, value)  # Perform forward checking

            result = Recursive_Backtracking(assignment, csp)  # Recur
            if result != "FAILURE":
                return result

            # If result is FAILURE, backtrack and remove assignment
            del assignment[var]
            csp.values[var] = csp.domain  # Reset the domain to the original domain for the variable

    return "FAILURE"


def Inference(assignment, inferences, csp, var, value):
    """
    Forward checking using the concept of Inferences
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
    Returns the string of values of given variable
    """
    return csp.values[var]  # Can improve by ordering values, for now returning the entire domain


def Select_Unassigned_Variables(assignment, csp):
    """
    Selects a new variable to be assigned using the minimum remaining value (MRV)
    """
    unassigned_variables = {squares: len(csp.values[squares]) for squares in csp.values if squares not in assignment}
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
        if neighbor in assignment and assignment[neighbor] == value:
            return False
    return True


def forward_checking(csp, assignment, var, value):
    """
    Forward checking that removes value from the domain of the neighbors
    """
    csp.values[var] = value  # Assign the value to the variable
    for neighbor in csp.peers[var]:
        csp.values[neighbor] = csp.values[neighbor].replace(value, '')  # Remove value from neighbors


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


def display(values):
    """
    Display the solved sudoku on the screen
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
