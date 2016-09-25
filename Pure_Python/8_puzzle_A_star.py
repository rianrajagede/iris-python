"""
Using manhattan distance + step as F()
Simple version from https://gist.github.com/thiagopnts/8015876
"""

import queue as Q

class Node:
    def __init__(self, val, step):
        self.val = val
        self.step = step
        
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
        locm = [[self.val.index(i)/3, self.val.index(i)%3] for i in xrange(9)] 
        return sum([abs(locp[i][1]-locm[i][1]) + abs(locp[i][0]-locm[i][0]) \
				for i in xrange(9)])
    
    def __eq__(self, other):
        return self.val == other.val

    def __hash__(self):
        return hash(str(self.val))

    def __lt__(self, other):
        return self.f() < other.f()

# Position for each possible movement
pos = [[3, 4, 5, 6, 7, 8], 
       [1, 2, 4, 5, 7, 8],
       [0, 1, 3, 4, 6, 7],
       [0, 1, 2, 3, 4, 5]]

# Possible movement
op = [-3, -1, 1, 3]

# A* algorithm
def main(initial):
    initial = Node(initial,0)
    openlist = Q.PriorityQueue()
    openlist.put(initial)
    closedlist = set()
    while not openlist.empty():
        now = openlist.get()
        if now.val[:-1]==range(1,9):
            print 'solved in %d move' %now.step
            break
        i = now.val.index(0)
        for j in xrange(4):
            if i in pos[j]:
                newval = now.val[:]
                newval[i], newval[i + op[j]] = newval[i + op[j]], newval[i]
                nextnode = Node(newval, now.step+1)
                if nextnode not in closedlist:
                    openlist.put(nextnode)

# Puzzle is 1D list
if __name__=="__main__":
    puzzle = [1, 2, 3,
              4, 6, 8, 
              0, 7, 5]
    main(puzzle)
