# Artificial-Intelligence-Machine-Learning-studies-
This is work from my introduction to Artificial Intelligence class I took in college. There are 4 projects here, the first 3 are based on the UC Berkeley pacman project (http://ai.berkeley.edu/home.html). For the first three projects you can play pacman with the command `python pacman.py` Utilizing the layout flag `-l` or `--layout` proceeding by `tinyMaze` `mediumMaze` `bigMaze` for alternate maps. For all options you can run `python pacman.py -h`. I've removed me and my partner's names from the files but the work is unchanged. Each of these projects is broken down into questions, which functionally segmented the large projects into portions. I have explanations on what we were supposed to implement for each part. 

#  ~ Project 1 ~

![alt text](https://github.com/TheodoreC13/Artificial-Intelligence-Machine-Learning-studies-/blob/main/Project1/maze.png)

### Files I edited:
> search.py
> 
> searchAgents.py

In project 1 the pacman agent is tasked with finding a particular location and collecting food efficiently. This is done via a collection of search algorithms.
* Question 1: Finding a fixed dot using Depth First Search

`python pacman.py -l mediumMaze -p SearchAgent`
* Question 2: Bredth First Search

`python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs`
* Question 3: Cost function variants - by varying the cost function the optimal pathing will change. This is the uniform cost function implementation

`python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs`
* Question 4: A* search - A* takes a heuristic function as an argument, the heuristic takes a state in the search problem and the problem itself. Slightly faster than UCS

`python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic`
* Question 5: Find all corners - Builds on BFS

`python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem`
* Question 6: Corners problem: Heuristic - non-trivial consistent heuristic for the corners problem.

`python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5`
* Question 7: Eat all dots - Done with A*

`python pacman.py -l trickySearch -p AStarFoodSearchAgent`
* Question 8: Suboptimal Search - "Optimal" can be hard to define and harder to implement. sometimes all you really want is speed. This search finds a reasonably good path as quickly as possible by just greedily eating the closest dot

`python pacman.py -l bigSearch -p ClosestDotSearchAgent -z .5`
# ~ Project 2 ~
### Files I edited:
> analysis.py
> 
> qlearningAgents.py
>
> valueIterationAgents.py

Project 2 is based around Markov Decision Process and Reinforcement Learning. MDP to describe a fully observable environment and Reinforcement Learning to learn an optimal policy. In this project I implement value iteration and q-learning. Our agents were first tested on gridworld then applied to a simulated robot controller and pacman. To start in manual mode `python gridworld.py -m` can be run. For full range of options run `python gridworld.py -h`. The Default agent moves randomly. Suboptimal. I didn't quite finish this project in time. In particular `def getQValue(self, state, action):` , `def update(self, state, action, nextState, reward):` , and `def final(self, state):` are empty for the "Approximate Q-Learning" section.

* Question 1: Standard Value Iteration via Bellman Upate - 
  
![alt text](https://github.com/TheodoreC13/Artificial-Intelligence-Machine-Learning-studies-/blob/main/Project2/bellman2.png)

Value iteration computes k-step estimates of the optimal values, U<sup>K</sup> . This was used to implement `computeActionFromValues(state)` and `computeQValueFromValues(state, action)`. We used the "batch" version where U<sup>K</sup> is computed from a fixed vector U<sup>K-1</sup> not the version where one single weight vector is updated in place. A state's value is updated in iteration K based on previous iteration K-1. Our goal was to return the synthesized policy π<sup>K+1</sup>. After 5 value iterations we were supposed to get the following output:

![alt text](https://github.com/TheodoreC13/Artificial-Intelligence-Machine-Learning-studies-/blob/main/Project2/value.png)

* Question 2: Bridge crossing analysis - `BridgeGrid` is a map with a low reward start state and a high reward terminal state seperated by a narrow bridge with a chasm on either side. The Agent starts at the low value state. With the starting parameters of default discount .09 and default noise .02, the optimal policy does not cross the bridge. My answer is in q2 of `analysis.py`

![alt text](https://github.com/TheodoreC13/Artificial-Intelligence-Machine-Learning-studies-/blob/main/Project2/value-q2.png)

* Question 3: Risk Policy - Consider the grid shown below. The agent will start in the yellow square. There is a row of terminal states on the bottom with an negative payoff, and 2 positive payoff terminal states of differing values 1 and 10. There are two paths. One that "risks" the cliff for a shorter route, and a second path that will avoid the cliff(risk) traveling alongside the top of the grid that takes longer. All of my answers to this are in q3 of `analysis.py`

![alt text](https://github.com/TheodoreC13/Artificial-Intelligence-Machine-Learning-studies-/blob/main/Project2/discountgrid.png)

* Question 4: Asynchronous Value Iteration - A value agent was written in `AsychronousValueIterationAgent`. The goal was to write a cyclical vlaue iteration for the specificed number of iterations before a return. We were warned to try and optimize speed here as later in this project a slow value iteration agent method would cause issues in subsequent questions.

`python gridworld.py -a asynchvalue -i 1000 -k 10`
* Question 5: Prioritized Sweeping Value Iteration - This is a simplified version of the standard prioritized sweeping algorithm, described in the paper "Memory-based Reinforcement Learning: Efficient Computation with Prioritized Sweeping" by Andrew W. Moore and Christopher G. Atkeson. We defined predecessors of a state s as all states that have a nonzero probability of reaching s by taking an action a. Theta, which is passed as a parameter, represents our error tolerance when deciding on the value of a state. This follows the following steps:
~~~~
Compute predecessors of all states
Initialize an empty priority queue
For each non terminal state, s, do:
    find absolute value of the difference between the current value of s in `self.values` and the higehst Q-Value across all possible actions from s, call this number `diff`
    push s into the priority queue with priority -diff (note: negative). Priority queue is a min heap but we want to prioritize updating states with have a higher error
For iteration in 0,1,2... self.iterations -1 do:
    If priority queue is empty -> Terminate
    Pop a state S off the priority queue
    update the value of S in self.values (if not terminal)
    for each predecessor p of s, do:
        Find the absolute value of the difference between the current value of P in self.values and the highest Q-value accross all possible actions from p, this is diff
        If Diff > Theta, push p into the priority queue with priority -diff, as long as it does not already exist in the priority queue with an equal or lower priority
~~~~
Predessors were stored in a set not a list to avoid duplicates. We were told to use util.PriorityQueue and pointed towards its `update` method. We were also warned that slow implementation here would result in difficulties later. 

`python gridworld.py -a priosweepvalue -i 1000`

* Question 6: Q-learning -  The previous value iteration agent isn't actually a learning agent, it's a reflex agent. It contemplates it's MDP model and obtains a complete policy before ever taking any actions. When it does interact with its environment it follows policy. This distinction is subtle in our simulated environment but very important in the real world where a real MDP isn't available. We wrote a Q-Learning agent. It didn't do much on construction but learned from the world around it through trial and error with its iteractions with the environment via its `update(state, action, nextState, reward)` method.

`python gridworld.py -a q -k 5 -m`  To observe the Q-Learner under manual control.
* Question 7: Epsilon greedy - An Epsilon fraction of the time a random action is picked instead of the normal optimal action. Note: This can still return the optimal action as it is any legal action, not any legal sub-optimal action. We observed behavior for the agent with the epsilon = 0.3

`python gridworld.py -a q -k 100`

`python gridworld.py -a q -k 100 --noise 0.0 -e x` To observe a particular epsilon value, change x to a particular value between 0 and 1. For example: 

`python gridworld.py -a q -k 100 --noise 0.0 -e 0.9` You can run this to see a high degree of randomness in actions taken.

With epsilon implemented the crawler `python crawler.py` should be functional.

* Question 8: Bridge Crossing Revisited - First we trained a random Q-Learner with the default learning rate on a noiseless BridgeGrid for 50 episodes and observed. `python gridworld.py -a q -k 50 -n 0 -g BridgeGrid -e 1` Then we tried the same expirement with an epsilon of 0. We tried various epsilon and learning rates looking for any combination that would give us a highly likely  (defined as >99%) chance that the optimal policy is returned within 50 iterations. My answer is in `analysis.py`. Spoiler: I didn't find it to be possible.
* Question 9: Q-learning and Pacman - Pacman runs games in two phases for Q-Learning: Training and Testing. Because training takes a long time, especially on tiny grids it is run in quiet mode with no GUI or console display. One training is done, pacman runs in Testing mode. When testing pacman's epsilon and alpha are set to 0, stopping Q-learning and disabling exploration in order to show the effectiveness of the learned policy. You shouldn't need code changes if your code was done correctly. During training you recieve output every 100 games with statistics on how pacman is doing. This is optimized for a smallgrid and does not work well with other maps.

`python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid` To run the Pacman Q-learning agent.

* Question 10: Approximate Q-Learning - I do not think I finished this portion of the project. I may have run into time constraints with other classes. I strongly remember taking Cryptography the same semester and having to do a lot of studying and work for that class. As such this portion was unfinished.

# ~ Project 3 ~
### Files I edited:
> busterAgents.py
> 
> inference.py

In this project our pacman agent will use sensors to locate and eat invisible ghosts. Pacman is equipped with sonar that provides noisy readings of the Manhattan distance to each ghost. The game ends when Pacman has eaten all of the ghosts. You can run `python busters.py` to play on your own. The blocks of color indicate where the ghost could possibly be given the noisy distance readings provided to pacman. The estimation shown is crude and we want to do better. We used Bayesian Networks to better estimate the location of the ghosts for pacman. This project was frustrating with certain solution implementation choices I made having to be refactored later, or just end up being detrimental. For this project most of the running / testing is done via the autograder. `python autograder.py -q qx` 'x' being the question number. There is also the flag `--no-graphics` to run the test without the graphic. The no-graphics flag was important as the graphic slowed the run time down tremendously.

![alt text](https://github.com/TheodoreC13/Artificial-Intelligence-Machine-Learning-studies-/blob/main/Project3/busters.png)

* Question 1: Observation Probability - Take an observation (noisy distance reading to the ghosts), pacman's position, the ghost's position, the position of ghost's jail, and return the probability of the reading. Very simply: P(noisyDistance | pacmanPosition, ghostPosition) There is a special case after a ghost has been jailed where our distance sensor will return `none`. So if the ghost's position is in jail the probability of it being in jail is 1, conversely if the distance reading is not `none` then the ghost is in jail with probability 0. 
* Question 2: Exact Ingerence Observation - We now update the agent's belief distribution given an observation from pacman's sensors. 
* Question 3: Exact Inference with Time Elapse - Observation isn't the only way we can update our agent. Pacman understands how the ghosts move and can predict to varying degree the movement of ghosts. In particular: ghosts can not move through walls, and they only move 1 step at a time. This also means if pacman recieves a bunch of readings that a ghost is near but one reading that the ghost is far, the far reading can be dismissed as pacman knows the ghost can not move that far in only one move. 
* Question 4: Exact Interface full test - Now a greedy hunting strategy is implemented and pacman will hunt down the closest ghost. Beforehand pacman was moving by randomly selecting a valid action. 
* Question 5: Approximate Inference Initialization and Beliefs - A particle filtering algorithm is implemented for tracking a single ghost.  
* Question 6: Approximate Inference Observation - Weight distribution is added to the particle filter
* Question 7: Approximate Inference with Time Elapse - Adds a time dimension to the particle filter increasing accuracy
* Question 8: Joint Particle Filter Observation - A new ghost type is added that avoids other ghosts. To track all ghosts simultaneously a dynamic Bayes Net is implemented. 
* Question 9: Joint Particle Filter Observation - Weighting and resampling for the particle filter. 
* Question 10: Joint Particle Filter Time Elapse and Full Test - Elapsed time is the final part of the Particle filter implemented.
  
![alt text](https://github.com/TheodoreC13/Artificial-Intelligence-Machine-Learning-studies-/blob/main/Project3/disperse.png)

# ~ Project 4 ~
### Files I edited
> hw4.py
>
> HW4-Answers.pdf
>
> README.txt

This was a group project on Linear Regression I worked on with a partner, name removed. `README.txt` is essentially work cited + instructions on how to run our project( we had no special instructions). For this project we implemented
* Linear Regression
* Gradient Descent
* Random Fourier Features

and wrote about our findings in `HW4-Answers.pdf`. The data used is in the folder titled `data`. This code in particular is heavily commented by the professor explaining the math and objective of different functions, as well as us for parts of our solution. There are many commented lines of other solution attempts that we left in on purpose as we were testing various solutions and implementations. 

### Linear Regression
Here are the questions we answered for linear regression in our pdf
````
Linear Regression
(a) Use your closed form implementation to fit a model to the data in 1D-no-noise-lin.txt and
2D-noisy-lin.txt. What is the loss of the fit model in both cases? Include plots for both. Note:
for some of these datasets the y-intercept will be non-zero, make sure you handle this case!
(b) What happens when you’re using the closed form solution and one of the features (columns of X)
is duplicated? Explain why. Note: you may want to test this out with your code as a useful first
step, but you should think critically about what is happening and why.
(c) Does the same thing happen if one of the training points (rows of X) is duplicated? Explain why.
(d) Does the same thing happen with Gradient Descent? Explain why.
````
### Gradient Descent
Here are the questions for gradient descent answered in our pdf
````
Gradient Descent
(a) Now use your Gradient Descent implementation to fit a model to 1D-no-noise-lin.txt with
alpha=0.05, num iters=10, and initial Theta set to the zero vector. What is the output (the
full list of (θ, loss) tuples for each of the 10 iterations)?
(b) Using the default parameters for alpha and num iters, and with initial Theta set to 0, do you
get the same model parameters as you did with the closed form solution? the same loss? Report
this for both 1D-no-noise-lin.txt and 2D-noisy-lin.txt.
(c) Find a set of values of the learning rate and iterations where you get the same answers and a set
of values where the answers are noticeably different. Explain what’s going on in both cases, and
for both datasets 1D-no-noise-lin.txt and 2D-noisy-lin.txt.
````
### Random Fourier Features
Here are the questions for random fourier features answered in our pdf. This section gave us a lot of trouble.
````
Random Fourier Features
(a) Use your Random Fourier Features implementation to fit models for 1D-exp-samp.txt, 1D-exp-uni.txt,
1D-quad-uni.txt, and 1D-quad-uni-noise.txt, each with 3 different values for K: small,
medium, and large. The small value should noticeably under-fit the data, the large value should
fit almost perfectly, and the medium value should be somewhere in between. Report the values
for K and include graphs for each of the four datasets (so you should report 12 values of K and
have 12 graphs in total).
(b) EXTRA CREDIT: What happens with Random Fourier Features if you try to predict a value
far away from any points in the training dataset? Modify the plotting functions (or write your
own) and show what the model looks like far away from the training points. Describe what you
are seeing qualitatively, compare the model’s output with the “true” output quantitatively, and
explain why this is happening. You may want to run these experiments multiple times to get a
better understanding of what is going on.
````
