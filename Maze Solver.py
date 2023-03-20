import numpy as np
import math
import cv2 as cv
from collections import deque
import time
# To implement Open-List priority queue with the help of
import heapq  # list(iterable unlike the Priority Queue of queue module)
# with insertion and deletion in O(log(n)) time

from warnings import warn  # To display Warning

t1 = time.time()  # Program Begins


# Reading the Image andImage Details
IMAGE_PATH = "spiral2.png"
image = cv.imread(IMAGE_PATH, cv.IMREAD_COLOR)
NAVIGABLE_PATH_COLOR = (255, 255, 255)
OBSTACLE_COLOR = (0, 0, 0)
# Determing source and target
final_path = image.copy()
source_coordinate = ()
target_coordinate = ()
src = np.where((image[:, :, 0] <= 127) & (image[:, :, 1] >= 170) & (image[:, :, 2] <= 127))
trgt = np.where((image[:, :, 0] <= 127) & (image[:, :, 1] <= 127) & (image[:, :, 2] >= 170))
source_coordinate = (src[0][0], src[1][0])
target_coordinate = (trgt[0][0], trgt[1][0])
print(source_coordinate)
print(target_coordinate)
# Creating Window for displaying the results
cv.namedWindow("Path", cv.WINDOW_NORMAL)

# Whether to show stimulation or not: If True it will
# show how the Alogorithm grows
RUN_VISUALISATION = False

# Defining the Diagonal Distance
CHEBYSHEV_DISTANCE = True  # True directs the cost function to take the cost
# of moving one unit diagonally as sqrt(2)
# whereas as False take Octile Distance (value as 1)

# Determing Heuristic to use (Only one of them should be true)
EUCLIDEAN_DISTANCE = True
MANHATTAN_DISTANCE = False
DIAGONAL_DISTANCE = False
DIJKSTRA_ALGORITHM = False
ADMISSIBLE_HEURISTIC = False
NON_ADMISSIBLE_HEURISTIC = False


# Whether to Allow Diagonal Movement or not
ALLOW_DIAGONAL_MOVEMENT = True


# Defining a Node Class for storing information of Nodes
class Node:
    """
    Class of Nodes to store informations about Nodes
    """

    def __init__(self, parent=None, position: tuple = None):
        self.parent = parent
        self.position = position

        self.f = np.inf
        self.g = np.inf
        self.h = np.inf
        self.isInClosedList = False
        self.isInOpenList = False

    def __eq__(self, other=None):
        """
            Defines equality betweeen two Nodes
        """
        return self.position == other.position

    def __repr__(self):
        """
            Helps in Printing the Information of node in a Readable format
        """
        return f"Position={self.position}---> (g:{self.g} ,h:{self.h} ,f:{self.f})"

    def __gt__(self, other):
        """
            Method for Comparing two Nodes (For Deciding Priority)
        """
        return self.f > other.f

    def __lt__(self, other):
        """
            Method for Comparing two Nodes (For Deciding Priority)
        """
        return self.f < other.f


# Utility Function for Checking the Accesibility of the Node
def isNavigable(image: np.ndarray = None, point: tuple = None):
    v, h = point
    V, H = image.shape[:2]
    if ((v >= 0) and (v < V) and (h >= 0) and (h < H)):
        return (NAVIGABLE_PATH_COLOR == tuple(image[v, h])) or (target_coordinate == (v, h))
    return False
# Defining Heuristic function


def heuristic(current: tuple = None, end: tuple = target_coordinate):  # h>h(actual)-non adm
    # h<h(actual)-admissible
    l, m = current
    p, q = end
    if DIJKSTRA_ALGORITHM:
        return 0
    elif EUCLIDEAN_DISTANCE:
        return math.sqrt((l-p)**2+(m-q)**2)
    elif MANHATTAN_DISTANCE:
        return abs(l-p)+abs(m-q)
    elif DIAGONAL_DISTANCE:
        return max(abs(l-p), abs(m-q))
    elif ADMISSIBLE_HEURISTIC:
        return math.sqrt((l-p)**2+(m-q)**2)/2
    else:
        return (l-p)**2+(m-q)**2

# Defining Actual Cost function


def cost(current: tuple = None, next: tuple = None):
    if not CHEBYSHEV_DISTANCE:
        return 1
    elif next[0] == current[0] or next[1] == current[1]:
        return 1
    else:
        return math.sqrt(2)


def AStarSearch(image: np.ndarray = None):

    # Initialise Open List (List+ Heapq)
    openList = []

    # Initialise Closed List
    final = np.empty(image.shape[:2], dtype=object)
    x, y = image.shape[:2]
    for i in range(x):
        for j in range(y):
            final[i, j] = Node(None, None)
            final[i, j].position = (i, j)
    # Set the Values of Start Node
    final[source_coordinate[0], source_coordinate[1]].isInOpenList = True
    final[source_coordinate[0], source_coordinate[1]].h = final[source_coordinate[0],
                                                                source_coordinate[1]].f = heuristic(source_coordinate)
    final[source_coordinate[0], source_coordinate[1]].g = 0
    heapq.heappush(openList, final[source_coordinate[0], source_coordinate[1]])

    # Adding a Stop Condition to avoid infinite loop
    iterations = 0
    max_iterations = (image.shape[0]*image.shape[1])

    # A flag to cancel visulisation
    CANCEL_VISUALISATION = False

    # Main Loop for finding destination
    while(len(openList)):
        iterations += 1

        # Extract the Node with lowest F-cost
        currentNode = heapq.heappop(openList)
        i, j = currentNode.position
        # final[i,j] and currentNode are equal as their position

        # Run the Visualisation if allowed
        if RUN_VISUALISATION and (not CANCEL_VISUALISATION):
            cv.imshow("Path", final_path)
            if ord('q') == cv.waitKey(1):
                CANCEL_VISUALISATION = True

        # Put the Node in Closed
        currentNode.isInOpenList = False
        if (not final[i, j].isInClosedList) or (final[i, j].f > currentNode.f):
            currentNode.isInClosedList = True
            final[i, j] = currentNode
        # Colour the Closed List (Visited Nodes)
        final_path[i, j] = [0, 127, 255]

        # Destination found; Return the path
        if(currentNode == final[target_coordinate[0], target_coordinate[1]]):
            print("Destination found! Displaying the Path")
            path(final_path, final, final[source_coordinate[0], source_coordinate[1]],
                 final[target_coordinate[0], target_coordinate[1]])
            return

        # Possible Neighbours
        adjacent_squares = np.array(
            [[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=int)
        if ALLOW_DIAGONAL_MOVEMENT:
            adjacent_squares = np.array(
                [[1, 1], [1, 0], [0, 1], [-1, 0], [0, -1], [1, -1], [-1, 1], [-1, -1]], dtype=int)

        # Too many iterations
        if max_iterations < iterations:
            warn("Too many iterations. Cannot find the path.")
            path(final_path, final,
                 final[source_coordinate[0], source_coordinate[1]], final[i, j])
            return

        # Generate Neighbours
        adjacent_squares[:] += currentNode.position

        # Iterate over the Neighbours
        for newPos in adjacent_squares:
            l, m = newPos

            # Check for Validity of the Neighbour
            if not isNavigable(image, (l, m)):
                continue

            # New Probable f,g,h values
            gNew = final[i, j].g + cost((i, j), (l, m))  # 1/root(2)
            hNew = heuristic((l, m))
            fNew = gNew+hNew

            # If This can be a better path, Remove it
            # from Closed List(if it was there initially) and update it with
            # the f,g,h values and add it to open list
            if gNew < (final[l, m].g):
                final[l, m].isInClosedList = False
                # Set the parent to current Node
                final[l, m].parent = final[i, j]
                final[l, m].f = fNew
                final[l, m].g = gNew
                final[l, m].h = hNew
                final[l, m].isInOpenList = True
                heapq.heappush(openList, final[l, m])
                # Colour the elements of OpenList
                final_path[l, m] = [255, 0, 0]

                # Run Visualisation if allowed
                if RUN_VISUALISATION and (not CANCEL_VISUALISATION):
                    cv.imshow("Path", final_path)
                    if ord('q') == cv.waitKey(1):
                        CANCEL_VISUALISATION = True

            else:
                continue

    warn("Not able to find path")
    return

# Returns Path, Distance


def path(final_path: np.ndarray = None, final: np.ndarray = None, start: Node = None, end: Node = None):

    n = end
    print("Distance (Cost): ", end.g)
    q = deque()
    while n != start:
        i, j = n.position
        q.appendleft((i, j))
        final_path[i, j] = [0, 255, 0]
        n = n.parent
    final_path[start.position[0], start.position[1]] = [0, 255, 0]
    q.appendleft(start.position)
    print("Path: ")
    print(list(q))

    return


def main():

    AStarSearch(image)

    cv.imshow("Path", final_path)

    t2 = time.time()-t1
    print("Runtime (in seconds):", t2)
    cv.waitKey(0)  # Press any key to Proceed

    # Displating the magnified image

    v, h = image.shape[:2]
    f = 1  # factor by which we magnify our image(integer)
    magnified = np.zeros((f*v, f*h, 3), dtype=np.uint8)
    for i in range(v):
        for j in range(h):
            magnified[f*i:f+f*i, f*j:f+f*j] = final_path[i, j]

    cv.imshow("Path", magnified)
    cv.waitKey(0)  # Press any key to end the program

    # cv.imwrite("NonAdmissible1.png",magnified)


if __name__ == "__main__":
    main()
