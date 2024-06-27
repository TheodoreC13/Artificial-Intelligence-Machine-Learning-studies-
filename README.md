# Artificial-Intelligence-Machine-Learning-studies-
This is work from my introduction to Artificial Intelligence class I took in college. There are 4 projects here, the first 3 are based on the UC Berkeley pacman project (http://ai.berkeley.edu/home.html). For the first three projects you can play pacman with the command `python pacman.py` Utilizing the layout flag `-l` or `--layout` proceeding by `tinyMaze` `mediumMaze` `bigMaze` for alternate maps. For all options you can run `python pacman.py -h`. I've removed me and my partner's names from the files but the work is unchanged. 

#  ~ Project 1 ~

In project 1 the pacman agent is tasked with finding a particular location and collecting food efficiently. This is done via a collection of search algorithms. Of the flies provided, only 'search.py' and 'searchAgents.py' were written by me (and my partner). The rest were provided as part of the project.
* Finding a fixed dot using Depth First Search

`python pacman.py -l mediumMaze -p SearchAgent`

* Bredth First Search

`python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs`

* Cost function variants - by varying the cost function the optimal pathing will change. This is the uniform cost function implementation

`python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs`

* A* search - A* takes a heuristic function as an argument, the heuristic takes a state in the search problem and the problem itself. Slightly faster than UCS

`python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic`

* Find all corners - Builds on BFS

`python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem`

* Corners problem: Heuristic - non-trivial consistent heuristic for the corners problem.

`python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5`

* Eat all dots - Done with A*

`python pacman.py -l trickySearch -p AStarFoodSearchAgent`

* Suboptimal Search - "Optimal" can be hard to define and harder to implement. sometimes all you really want is speed. This search finds a reasonably good path as quickly as possible by just greedily eating the closest dot

`python pacman.py -l bigSearch -p ClosestDotSearchAgent -z .5`
# ~ Project 2 ~

# ~ Project 3 ~

# ~ Project 4 ~
