# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    # Fails at 0.02, Succeeds at 0 and 0.01. Manually tested both. At no level of discount is the optimal policy to
    # cross because of how high noise is. Noise has to be reduced. Interestingly enough, even with close to no noise
    # (0.02) the penalty for falling is so great it makes the bridge not worth crossing. 0.017 seems to be the lowest
    # value of noise that still returns an optimal policy of standing still. at 0.0169 and lower the optimal policy
    # crosses the bridge. I did not test with anything more than 4 decimals.
    answerNoise = 0.01
    return answerDiscount, answerNoise

def question3a():
    # Prefer the close exit (+1), risking the cliff (-10)
    # Almost all of this was trial and error playing with numbers
    answerDiscount = 1
    answerNoise = 0.2
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    # Prefer the close exit (+1), but avoiding the cliff (-10)
    # don't reward living or it will go to the far exit when it avoids the cliff
    # high discount also makes it want to go to the far exit
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    # Prefer the distant exit (+10), risking the cliff (-10)
    # make living painful to force cliff
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    # Prefer the distant exit (+10), avoiding the cliff (-10)
    # Be safe, take your time. High discount and no living reward to take the long way to far exit
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    # Avoid both exits and the cliff (so an episode should never terminate)
    # Just set it's living reward to max, don't discount. Noise is irrelevant, the optimal policy is to stay still
    answerDiscount = .9
    answerNoise = 0.2
    answerLivingReward = 100
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    # tested a bunch of learning rates and epsilons, nothing worked with the sample size of 50
    answerEpsilon = 0.1
    answerLearningRate = 1
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
