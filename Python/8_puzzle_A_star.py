"""
Using manhattan distance + step as F()
Simple version from https://gist.github.com/thiagopnts/8015876
"""
import queue as Q
import numpy as np
import copy

class Node:
    def __init__(self, val, step, prev):
        self.val = val
        self.step = step
        self.prev = prev
        
    def f(self):
        return self.h() + self.g()
    
    def g(self): 
        return self.step
    
    def h(self):
        locm = locp = [[0,0]]*9

        # List of correct position each piece
        locp[1:] = [[(i-1)/3,(i-1)%3] for i in xrange(1,9)] 
        locp[0] = [2,2]

        # List of current position each piece
        for i in xrange(3):
            for j in xrange(3):
                locm[self.val[i][j]]=[i,j]
        return sum([abs(locp[i][1]-locm[i][1]) + abs(locp[i][0]-locm[i][0]) \
                for i in xrange(9)])

    def __eq__(self, other):
        return self.val == other.val

    def __hash__(self):
        return hash(str(self.val))

    def __lt__(self, other):
        return self.f() < other.f()

def printstep(now):
    if now.prev!=-1:
        printstep(now.prev)
    print np.matrix(now.val)

# Possible movement
#      down    right   up      left
op = [[1, 0], [0, 1], [-1,0], [0,-1]]  # operation
cn = [[2,-1], [-1,2], [0,-1], [-1,0]]  # condition

# A* algorithm
def main(initial, target):
    initial = Node(initial,0,-1)
    openlist = Q.PriorityQueue()
    openlist.put(initial)
    closedlist = set()
    kk = 0
    while not openlist.empty():
        now = openlist.get()
        if now.val==target:
            print 'solved in %d move' % now.step
            printstep(now)
            break
    
        for i in xrange(3):
            for j in xrange(3):
                if now.val[i][j]==0:
                    pos=(i,j)

        for x in xrange(4):
            i,j = pos
            if i!=cn[x][0] and j!=cn[x][1]:
                newval = copy.deepcopy(now.val)
                newval[i][j], newval[i+op[x][0]][j+op[x][1]] = \
					newval[i+op[x][0]][j+op[x][1]], newval[i][j]
                nextnode = Node(newval, now.step+1, now)
                if nextnode not in closedlist:
                    openlist.put(nextnode)

        closedlist.add(now)

# Puzzle is 2D list
if __name__=="__main__":
    puzzle = [[4, 1, 3],
              [2, 0, 6],
              [7, 5, 8]]
    # Target
    target = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]]

    main(puzzle, target)