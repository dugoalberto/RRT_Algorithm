import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
import random

from matplotlib.pyplot import rcParams
np.set_printoptions(precision=3, suppress=True)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
plt.rcParams['font.size'] = 22

class treeNode():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.children = []
        self.parent = None

class RRT():
    def __init__(self, start, goal, numIt, grid, stepSize):
        self.randomTree = treeNode(start[0],start[1])
        self.goal = treeNode(goal[0],goal[1])
        self.nearestNode = None
        self.iteration = min(numIt, 200)
        self.grid = grid
        self.rho = stepSize #Length of each branch
        self.path_distance = 0
        self.nearestDist = 1000
        self.numWayPoint = 0
        self.wayPoint = [] #i vari punti che posso percorrere

def addChild(self, x, y, parent):
    if( x == self.goal.x):
        self.goal = treeNode(self,x,y)
        self.nearestNode.children = self.goal
        self.goal.parent = self
    # add tempNode to children of nearest node
    else:
        tempNode = treeNode(x,y)
        self.nearestNode.children = tempNode
        # add tempNode to children of nearest node

#sample a random point within grid limits
def sempleAPoint(self):
    x = random.randint(1, grid.shape[1]) #genera casualmente un numero intero tra 1 (incluso) e l'altezza della griglia ~700
    y = random.randint(1, grid.shape[0]) #~400
    point = np.array([x, y])
    return point

#steer a distance stepsize from start to end location
def steerToPoint(self, locationStart, locationEnd):
    offset = self.rho * self.unitVector(locationStart, locationEnd)
    point = np.array([locationStart.x + offset[0], locationStart.y + offset[1]])
    if point[0] >= grid.shape[1]: #se la coordinata X del nuovo punto è maggiore o uguale alla larghezza della griglia, imposta la coordinata X del punto al valore massimo consentito, impedendo al punto di uscire dalla griglia
        point[0] = grid.shape[1]
    if point[1] >= grid.shape[0]:
        point[1] = grid.shape[0]
    return point
#check if obstacle lies between the start node and end point of the edge
def isInObstacle(self, locationStart, locationEnd):
    u_hat = self.unitVector(locationStart, locationEnd)
    testPoint = np.array([0.0, 0.0])
    for i in range(self.rho):
        testPoint[0] = locationStart.location + i * u
        hat[0]
    testPoint[1] = locationStart.locationY + i * u_hat[1]  # check if testPoint lies within obstacle
    if True:
        return True
    return False

#find unit vector between 2 points which form a vector
def unitVector(self, locationStart, locationEnd):
    v = np.array([locationEnd[0] - locationStart.x, locationEnd[1] - locationStart.y])
    u_hat = v / np.linalg.norm(v)
    return u_hat


#find the nearest node from a given unconnected point (Euclidean distance)
def findNearest(self, root, point):
    # return condition if root is NULL
    # find distance between root and point #if this is Lower than the nearest Distance, set this as the nearest node and update nearest distance #recursively call by iterating through the children
    for child in root.children:  # do something
        pass

#sqrt( (px-qx)^2 + (py-qy)^2 )
#find euclidean distance between a node and an XY point
def distance(self, node1, point):
    return np.sqrt((nodel.locationx - point[0]) ** 2 + (nodel.locationY - point[1]) ** 2)


#check if the goal has been reached within step size
def goalFound(self, point):
    pass

def resetNearestValue(self):
    self.nearestNode = None
    self.nearestDist = 10000


def retraceRRTPath(self, goal):
    # end the recursion when goal node reaches the start node
    # add 1 to number of waypoints
    # insert currentPoint to the Waypoints array from the beginning
    currentPoint = np.array([goal.locationX, goal.locationY])  # add step size (rho) to path distance
    # recursive call.


#end method

# Carica l'array NumPy
grid = np.load('cspace.npy')

# Definisci le variabili
start = np.array([100.0, 100.0])
goal = np.array([600.0, 400.0])
numIteration = 200
stepSize = 50

# Crea una regione obiettivo come un cerchio
goalRegion = plt.Circle((goal[0], goal[1]), stepSize, color='b', fill=False)

# Crea una figura
fig = plt.figure("rtt Algoritm")

# Visualizza l'array NumPy come un'immagine binaria
plt.imshow(grid, cmap='binary')

# Disegna il punto di partenza e il punto di destinazione
plt.plot(start[0], start[1], 'ro')
plt.plot(goal[0], goal[1], 'bo')

# Ottieni l'oggetto degli assi
ax = fig.gca()

# Aggiungi il cerchio della regione obiettivo agli assi
ax.add_patch(goalRegion)
# Mostra la figura
plt.show()
