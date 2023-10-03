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

class RRTAlgorithm():
    def __init__(self, start, goal, numIt, grid, stepSize):
        self.randomTree = treeNode(start[0],start[1])
        self.goal = treeNode(goal[0],goal[1])
        self.nearestNode = None
        self.iterations = min(numIt, 200)
        self.grid = grid
        self.rho = stepSize #Length of each branch
        self.path_distance = 0
        self.nearestDist = 1000
        self.numWayPoint = 0
        self.wayPoint = [] #i vari punti che posso percorrere
    def addChild(self, x, y):
        if(x == self.goal.x):
            self.nearestNode.children.append(self.goal)
            self.goal.parent = self.nearestNode
        # add tempNode to children of nearest node
        else:
            tempNode = treeNode(x,y)
            self.nearestNode.children.append(tempNode)
            tempNode.parent= self.nearestNode
    #sample a random point within grid limits
    def sempleAPoint(self):
        x = random.randint(1, grid.shape[1]) #genera casualmente un numero intero tra 1 (incluso) e l'altezza della griglia ~700
        y = random.randint(1, grid.shape[0]) #~400
        point = np.array([x, y])
        return point
    #steer a distance stepsize from start to end location
    def steerToPoint(self, locationStart, locationEnd):
        offset = self.rho*self.unitVector(locationStart, locationEnd)
        point = np.array([locationStart.x + offset[0], locationStart.y + offset[1]])
        if point[0] >= grid.shape[1]:
            point[0] = grid.shape[1]-1
        if point[1] >=grid.shape[0]:
            point[1] = grid.shape[0]- 1
        return point
    #check if obstacle lies between the start node and end point of the edge
    def isInObstacle(self, locationStart, locationEnd):
        u_hat = self.unitVector(locationStart, locationEnd)
        testPoint = np.array([0.0, 0.0])
        for i in range(self.rho):
            testPoint[0] = locationStart.x + i * u_hat[0]
            testPoint[1] = locationStart.y + i * u_hat[1]
            #guardo se è nero quindi è un ostacolo
        if self.grid[round(testPoint[1]),round(testPoint[0])] == 1:
            return True
        return False

    #find unit vector between 2 points which form a vector
    #prima calcolo vettore direzionale, quindi non so quanto sia lungo
    # poi lo normalizzo : u_hat
    # chatGPT:
    # Nel contesto della tua funzione unitVector, quando si calcola il "vettore direzionale (non normalizzato)",
    # si sta calcolando un vettore che rappresenta la direzione tra due punti specifici,
    # ma questo vettore può avere una lunghezza diversa a seconda della distanza tra i punti.
    # Successivamente, il vettore direzionale può essere normalizzato dividendo ogni sua componente per la sua lunghezza,
    # ottenendo così un "vettore unitario" che rappresenta solo la direzione senza tener conto della lunghezza del movimento.
    # In sintesi, il vettore direzionale contiene informazioni sulla direzione e sulla lunghezza del movimento, mentre il vettore unitario rappresenta solo la direzione e ha una lunghezza di 1 unità.
    def unitVector(self, locationStart, locationEnd):
        v = np.array([locationEnd[0] - locationStart.x, locationEnd[1] - locationStart.y])
        u_hat = v / np.linalg.norm(v)
        return u_hat


    #find the nearest node from a given unconnected point (Euclidean distance)
    def findNearest(self, root, point):
        # return condition if root is NULL
        # find distance between root and point
        #if this is Lower than the nearest Distance, set this as the nearest node and update nearest distance #recursively call by iterating through the children
        if not root:
            return
        dist = self.distance(root, point)
        if dist <= self.nearestDist:
            self.nearestNode = root
            self.nearestDist = dist
        # recursively call by iterating through the children
        for child in root.children:
            self.findNearest(child, point)
        pass

    #sqrt( (px-qx)^2 + (py-qy)^2 )
    #find euclidean distance between a node and an XY point
    def distance(self, node, point):
        return np.sqrt((node.x - point[0]) ** 2 + (node.y - point[1]) ** 2)


    #check if the goal has been reached within step size
    def goalFound(self, point):
        if self.distance(self.goal, point) <= self.rho:
            return True
        pass

    def resetNearestValues(self):
        self.nearestNode = None
        self.nearestDist = 10000


    def retraceRRTPath(self, goal):
        # end the recursion when goal node reaches the start node
        # add 1 to number of waypoints
        # insert currentPoint to the Waypoints array from the beginning
        if goal.x == self.randomTree.x:
            return
        self.numWayPoint += 1
        # insert currentPoint to the Waypoints array from the beginning
        currentPoint = np.array([goal.x, goal.y])
        self.wayPoint.insert(0, currentPoint)
        self.path_distance += self.rho
        self.retraceRRTPath(goal.parent)


#end method

# Carica l'array NumPy
grid = np.load('cspace.npy')
start = np.array([100.0, 100.0])
goal = np.array([600.0, 400.0])
numIt = 200
stepSize = 50
# Crea una regione obiettivo come un cerchio
goalRegion = plt.Circle((goal[0], goal[1]), stepSize, color='b', fill=False)

fig = plt.figure("rtt Algoritm")
plt.imshow(grid, cmap='binary')
plt.plot(start[0], start[1], 'ro')
plt.plot(goal[0], goal[1], 'bo')
ax = fig.gca()
ax.add_patch(goalRegion)

rrt = RRTAlgorithm(start, goal, numIt, grid, stepSize)

for i in range(rrt.iterations):
    # Reset nearest values
    rrt.resetNearestValues()
    print("Iteration: ", i)
    point = rrt.sempleAPoint()
    rrt.findNearest(rrt.randomTree, point)
    new = rrt.steerToPoint(rrt.nearestNode, point)
    bool = rrt.isInObstacle(rrt.nearestNode, new)
    if (bool == False):
        rrt.addChild(new[0], new[1])
        plt.pause(0.50)
        plt.plot([rrt.nearestNode.x, new[0]], [rrt.nearestNode.y, new[1]], 'go', linestyle="--")
        if (rrt.goalFound(new)):
            rrt.addChild(goal[0], goal[1])
            print("goal found")
            break
# trace back the path returned, and add start to waypoints
rrt.retraceRRTPath(rrt.goal)
rrt.wayPoint.insert(0,start)
print("Number of waypoints: ", rrt.numWayPoint)
print("Path Distance (m): ", rrt. path_distance)
print("Waypoints: ", rrt.wayPoint)

for i in range(len(rrt.wayPoint)-1):
    plt.plot([rrt.wayPoint[i][0], rrt.wayPoint[i+1][0]], [rrt.wayPoint[i][1], rrt.wayPoint[i+1][1]], 'ro', linestyle="--")
    plt.pause(0.50)
plt.show()