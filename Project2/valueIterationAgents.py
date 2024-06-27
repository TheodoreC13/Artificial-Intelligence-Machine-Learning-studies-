# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # for i in iterations:
        #     Initialize new_U
        #         for s in states:
        #               new_U(s) = R(s) + gamma* max(sum(p(s’|a,s)*U(s’)))
        #     U = new_U
        #     Optionally break once convergence criteria met
        "*** YOUR CODE HERE ***"
        # no convergence test is needed, we do i iterations
        for i in range(self.iterations):
            iValues = util.Counter()  # dictionary of values for each state
            for m in self.mdp.getStates():  # calculate for each state
                if self.mdp.isTerminal(m):  # first we check if we're at the exit
                    continue
                else:  # else return a list of possible actions and pick best policy based on q value
                    actions = self.mdp.getPossibleActions(m)
                    iValues[m] = max([self.computeQValueFromValues(m, a) for a in actions])  # grab optimal action
            self.values = iValues  # set the values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        TSP = self.mdp.getTransitionStatesAndProbs(state, action)  # List of transition states
        value = 0  # Start at 0
        for tstate in TSP:  # for each transition state
            reward = self.mdp.getReward(state, action, tstate[0])  # get it's reward
            # use that reward to calculate the value, adding in discount and previous value
            value = value + reward + self.discount * (self.values[tstate[0]] * tstate[1])
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = util.Counter()  # Dictionary of values
        for action in self.mdp.getPossibleActions(state):  # Compute for each action
            possibleActions[action] = self.computeQValueFromValues(state, action)  # Grab Q value for the action
        actionTaken = possibleActions.argMax()  # Return action with best Q value
        return actionTaken

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()  # list of states
        stateCount = len(states)  # amount of states, used for state updating
        for i in range(self.iterations):
            st = states[i % stateCount]  # this is the state we're updating this iteration
            if self.mdp.isTerminal(st):
                continue
            actions = self.mdp.getPossibleActions(st) # list of possible actions
            optimalAction = max([self.computeQValueFromValues(st, a) for a in actions])  # grab the optimal one
            self.values[st] = optimalAction  # set the state value for that action


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        pQueue = util.PriorityQueue()
        states = self.mdp.getStates()
        predecessors = {}
        for s in states:  # for each state
            if self.mdp.isTerminal(s):  # terminal
                continue
            for a in self.mdp.getPossibleActions(s):  # does an action give us another previous state
                for st, _ in self.mdp.getTransitionStatesAndProbs(s, a):  # '_' is used to ignore the secondary value
                    if st in predecessors:
                        predecessors[st].add(s)  # if it's already in predecessors, add this state
                    else:
                        predecessors[st] = {st}  # if not, add to predecessors
        #  For each non-terminal state s, do: (note: to make the autograder work for this question, you must iterate
        #  over states in the order returned by self.mdp.getStates())
        #     Find the absolute value of the difference between the current value of s in self.values and the highest
        #     Q-value across all possible actions from s (this represents what the value should be); call this number
        #     diff. Do NOT update self.values[s] in this step. Push s into the priority queue with priority -diff
        #     (note that this is negative). We use a negative because the priority queue is a min heap,
        #     but we want to prioritize updating states that have a higher error.
        for s in self.mdp.getStates():  # grab the diff for pqueue from actions
            if self.mdp.isTerminal(s):
                continue
            maxQ = max([self.computeQValueFromValues(s, action)for action in self.mdp.getPossibleActions(s)])
            diff = abs(self.values[s] - maxQ)
            pQueue.update(s, -diff)
        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        #
        #     If the priority queue is empty, then terminate.
        #     Pop a state s off the priority queue.
        #     Update the value of s (if it is not a terminal state) in self.values.
        #     For each predecessor p of s, do:
        #         Find the absolute value of the difference between the current value of p in self.values and the
        #         highest Q-value across all possible actions from p (this represents what the value should be); call
        #         this number diff. Do NOT update self.values[p] in this step.
        #         If diff > theta, push p into the priority queue with priority -diff (note that this is negative), as
        #         long as it does not already exist in the priority queue with equal or lower priority. As before, we
        #         use a negative because the priority queue is a min heap, but we want to prioritize updating states
        #         that have a higher error.
        for i in range(self.iterations): # this is mostly all pulled from previous code; just iterated over the pQueue
            if pQueue.isEmpty():
                break
            else:
                s = pQueue.pop()
                if not self.mdp.isTerminal(s):
                    self.values[s] = max([self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)])
                for predState in predecessors[s]:
                    if self.mdp.isTerminal(predState):
                        continue
                    maxQ = max([self.computeQValueFromValues(predState, a) for a in self.mdp.getPossibleActions(predState)])
                    diff = abs(self.values[predState] - maxQ)
                    if diff > self.theta:
                        pQueue.update(predState, -diff)
