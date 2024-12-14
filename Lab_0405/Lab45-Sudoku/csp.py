# CLASS DESCRIPTION FOR CONSTRAINT SATISFACTION PROBLEM (CSP)

from util import *

class csp:

    # INITIALIZING THE CSP
    def __init__(self, domain=digits, grid=""):
        """
        Unitlist consists of the 27 lists of peers
        Units is a dictionary consisting of the keys and the corresponding lists of peers
        Peers is a dictionary consisting of the 81 keys and the corresponding set of 27 peers
        Constraints denote the various all-different constraints between the variables
        """
        "***YOUR CODE HERE ***"
        self.domain = domain
        self.variables = squares

        self.unitList = (
            [cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]
        )

        self.units = {square: [u for u in self.unitList if square in u] for square in self.variables}
        self.peers = {square: set(sum(self.units[square], [])) - {square} for square in self.variables}

        if grid:
            self.values = self.getDict(grid)
        else:
            self.values = {square: self.domain for square in self.variables}

    def getDict(self, grid=""):
        """
        Getting the string as input and returning the corresponding dictionary
        """
        i = 0
        values = dict()
        for cell in self.variables:
            if grid[i] != '0':
                values[cell] = grid[i]
            else:
                values[cell] = digits
            i = i + 1
        return values