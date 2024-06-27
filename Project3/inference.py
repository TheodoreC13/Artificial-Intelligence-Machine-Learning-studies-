# inference.py
# ------------
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


import itertools
import math
import random
import busters
import game
import numpy as np

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        if self.total() == 0:
            return
        factor = 1.0/self.total()
        for key in self:
            self[key] = self[key]*factor
        return


    #TODO
    # NOTE: I have no idea how property got here, I have had a huge amount of trouble with floats with my
    # previous implementation of this method. I tried some other ways of sampling and they A. Don't distribute correctly
    # and B. Don't work or break other implementations.
    @property
    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        # original sampling code, this breaks once the tests end up using floats
        # keys = []
        # weights = []
        # for a in self:
        #     keys.append(a)
        # for a in self:
        #     weights.append(self[a])
        # key = random.sample(keys, 1, counts=weights)
        # return key[0]
        self.normalize()

        keys = []
        weights = []
        for a in self:
            keys.append(a)
        for a in self:
            weights.append(self[a])
        rand = random.random()
        # TODO ths return wasn't working correctly either, sampling has *really* fucked my project over, especially
        # TODO obnoxious for something worth 0 points
        # return random.choices(keys,weights)
        # this feels weird, sorta works better but not much
        i, total = 0, weights[0]
        while rand > total:
            i += 1
            total += weights[i]
        return keys[i]


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        #  When Ghostposition = jailposition, return 1
        if ghostPosition == jailPosition:
            # when noisydistance = none, return 1
            if noisyDistance is None:
                return 1.0
            else:
                return 0.0
        # flip them for both cases
        if noisyDistance is None:
            # when noisydistance = none, return 1
            if ghostPosition == jailPosition:
                return 1.0
            else:
                return 0.0
        """ P(noisyDistance | trueDistance)"""
        trueDistance = manhattanDistance(pacmanPosition, ghostPosition)
        return busters.getObservationProbability(noisyDistance, trueDistance)



    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        #normalize everything first
        self.beliefs.normalize()
        # get pacman's position, needed for our observational probability
        pacman = gameState.getPacmanPosition()
        # get jail position for observational probability
        jail = self.getJailPosition()
        # for each ghost position, get the current belief, the observational probability for that position
        # then update the beliefs with the new probability
        for x in self.allPositions:
            old = self.beliefs[x]
            prob = self.getObservationProb(observation, pacman, x, jail)
            new = old*prob
            self.beliefs[x] = new
        # normalize everything again
        self.beliefs.normalize()


    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        # We want to make a new key + weight table
        newDistribution = DiscreteDistribution()
        # for every old position
        for oldPos in self.allPositions:
            # "distribution over new positions for the ghost, given its previous position"
            #  newPosDist is a DiscreteDistribution object, where for each position p in self.allPositions
            #  newPosDist[p] is the probability that the ghost is at position p at time t + 1, given that
            #  the ghost is at position oldPos at time t.
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            # we want to update each position with a new probability
            for newPos, newProb in newPosDist.items():  # we want both the position and probability
                newDistribution[newPos] += self.beliefs[oldPos] * newProb
        # set all the old beliefs to new ones
        for x in self.beliefs:
            self.beliefs[x] = newDistribution[x]

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        # number of particles
        num = self.numParticles
        # number of positions
        legalCount = len(self.legalPositions)
        # particles per position
        ppp = num/legalCount
        # append positions in order, x times
        for position in self.legalPositions:
            for i in range(int(ppp)):  # this was giving me weird issues.
                self.particles.append(position)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        weightedDistribution = DiscreteDistribution()
        pacman = gameState.getPacmanPosition()
        jail = self.getJailPosition()
        # grab new weighted distributions
        for particle in self.particles:
            newProb = self.getObservationProb(observation, pacman, particle, jail)
            weightedDistribution[particle] += newProb
        # normalize for sampling
        weightedDistribution.normalize()
        # if 0, initialize uniformly
        if weightedDistribution.total() == 0:
            self.initializeUniformly(gameState)
        else:
            self.beliefs = weightedDistribution
            keys = []
            weights = []
            for a in weightedDistribution:
                keys.append(a)
            for a in weightedDistribution:
                weights.append(weightedDistribution[a])
            for i in range(int(self.numParticles)):  # same int interaction
                # new particle list sampled from the new weighted distribution
                rand = random.random()
                i, total = 0, weights[0]
                while rand > total:
                    i += 1
                    total += weights[i]
                self.particles[i] = keys[i]


    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        # mostly the same as before
        new = []
        for oldPos in self.particles:
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            # newPosition = newPosDist.sample() TODO NOT WORKING CORRECTLY
            newPosDist.normalize()
            keys = []
            weights = []
            for a in newPosDist:
                keys.append(a)
            for a in newPosDist:
                weights.append(newPosDist[a])
            rand = random.random()
            i, total = 0, weights[0]
            while rand > total:
                i += 1
                total += weights[i]
            new.append(keys[i])
        self.particles = new


    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        beliefDistribution = DiscreteDistribution()
        for p in self.particles:
            beliefDistribution[p] += 1
        beliefDistribution.normalize()
        return beliefDistribution


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        # get all the permutations
        perms = list(itertools.product(self.legalPositions, repeat=self.numGhosts))
        # shuffle for random order
        random.shuffle(perms)
        # initialize particles, this is pretty much the same as what I used previously
        for i in range(self.numParticles):
            for particle in perms:
                self.particles.append(particle)


    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        # same old
        pacman = gameState.getPacmanPosition()
        weightedDistribution = DiscreteDistribution()
        for particle in self.particles:
            prob = 1.0
            for i in range(self.numGhosts):
                prob *= self.getObservationProb(observation[i], pacman, particle[i], self.getJailPosition(i))
            weightedDistribution[particle] += prob

        weightedDistribution.normalize()
        if weightedDistribution.total() == 0:
            self.initializeUniformly(gameState)
        else:
            self.particles = [weightedDistribution.sample for _ in range(self.numParticles)]


    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            """Then, assuming that i refers to the index of the ghost, to obtain the distributions over new 
                positions for that single ghost, given the list (prevGhostPositions) of previous positions 
                of all of the ghosts, use: 
                newPosDist = self.getPositionDistribution(gameState, prevGhostPositions, i, self.ghostAgents[i])"""
            # it's currently called newParticle.
            for i in range(self.numGhosts):
                newPosDist = self.getPositionDistribution(gameState, newParticle, i, self.ghostAgents[i])
                # what's commented out should work if sample wasn't fucked
                # newPosition = newPosDist.sample()
                # newParticle[i] = newPosition
                newPosDist.normalize()
                keys = []
                weights = []
                for a in newPosDist:
                    keys.append(a)
                for a in newPosDist:
                    weights.append(newPosDist[a])
                rand = random.random()
                j, total = 0, weights[0]
                while rand > total:
                    j += 1
                    total += weights[j]
                newParticle[i] = keys[j]
            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
