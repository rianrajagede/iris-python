"""
Using manhattan distance + step as F()
Simple version from https://gist.github.com/thiagopnts/8015876
"""
import Queue as Q
import copy
import numpy as np

class Node:
    def __init__(self, val, step, prev):
        self.val = val     # state of the puzzle
        self.step = step   # number of step has been passed
        self.prev = prev   # previous state
        self.locm = []
        self.locp = []
        
    def f(self):
        return self.h() + self.g()
    
    def g(self): 
        return self.step
    
    def h(self):
        self.locm = [[0,0]]*9
        self.locp = [[0,0]]*9

        # List of correct position each piece
        self.locp[1:] = [[(i-1)/3,(i-1)%3] for i in xrange(1,9)] 
        self.locp[0] = [2,2]

        # List of current position each piece
        for i in xrange(3):
            for j in xrange(3):
                self.locm[self.val[i][j]]=[i,j]
				
		# Return Manhattan Distance
        return sum([abs(self.locp[i][1]-self.locm[i][1]) + abs(self.locp[i][0]-self.locm[i][0]) \
                for i in xrange(9)])

    def __eq__(self, other):
        return self.val == other.val

    def __hash__(self):
        return hash(str(self.val))

    def __lt__(self, other):
        return self.f() < other.f()

		
# Print the shortest step and F() value
def printstep(now):
    if now.prev!=-1:
        printstep(now.prev)
    print np.matrix(now.val)
    print 'F = ' + str(now.f())+ ' | g = ' + str(now.g()) + ' | h = ' + str(now.h())

# Possible movement
#      down    right   up      left
op = [[1, 0], [0, 1], [-1,0], [0,-1]]  # operation
cn = [[2,-1], [-1,2], [0,-1], [-1,0]]  # condition

# A* algorithm
def main(initial, target):
    initial = Node(initial,0,-1)
    openlist = Q.PriorityQueue()  # will always sort by F() value
    openlist.put(initial)
    closedlist = set()
    
    while not openlist.empty():
        now = openlist.get()
		
		# if done
        if now.val==target:
            print 'solved in %d move' % now.step
            printstep(now)
            break
    
		# find zero position
        for i in xrange(3):
            for j in xrange(3):
                if now.val[i][j]==0:
                    pos=(i,j)

        for x in xrange(4):
            i,j = pos
			
			# create new node
            if i!=cn[x][0] and j!=cn[x][1]:
                newval = copy.deepcopy(now.val)
                newval[i][j], newval[i+op[x][0]][j+op[x][1]] = \
					newval[i+op[x][0]][j+op[x][1]], newval[i][j]
                nextnode = Node(newval, now.step+1, now)

				# check if it in closed list
                if nextnode not in closedlist:
                    openlist.put(nextnode)

        closedlist.add(now)

# Puzzle is 2D list
if __name__=="__main__":
    simple = [[1, 0, 3],
              [4, 2, 5],
              [7, 8, 6]]

    puzzle = [[2, 7, 5],
              [0, 4, 1],
              [3, 8, 6]]
              
    target = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]]

main(puzzle, target)