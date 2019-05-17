# grid_world.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this assignment, you will implement three classic algorithm for 
solving Markov Decision Processes either offline or online. 
These algorithms include: value_iteration, policy_iteration and q_learning.
You will test your implementation on three grid world environments. 
You will also have the opportunity to use Q-learning to control a simulated robot 
in crawler.py

The package `matplotlib` is needed for the program to run.

The Grid World environment has discrete state and action spaces
and allows for both model-based and model-free access.

It has the following properties:
    env.observation_space.n     # the number of states
    env.action_space.n          # the number of actions
    env.trans_model             # the transition/dynamics model

In value_iteration and policy_iteration, you can access the transition model 
at a given state s and action by calling
    t = env.trans_model[s][a]
where s is an integer in the range [0, env.observation_space.n),
      a is an integer in the range [0, env.action_space.n), and
      t is a list of four-element tuples in the form of
        (p, s_, r, terminal)
where s_ is a new state reachable from the state s by taking the action a,
      p is the probability to reach s_ from s by a, i.e. p(s_|s, a),
      r is the reward of reaching s_ from s by a, and
      terminal is a boolean flag to indicate if s_ is a terminal state.

In q_learning, once a terminal state is reached the environment should be (re)initialized by
    s = env.reset()
where s is the initial state.
An experience (sample) can be collected from s by taking an action a as follows:
    s_, r, terminal, info = env.step(a)
where s_ is the resulted state by taking the action a,
      r is the reward achieved by taking the action a,
      terminal is a boolean flag to indicate if s_ is a terminal state, and
      info is just used to keep compatible with openAI gym library.


A Logger instance is provided for each function, through which you can
visualize the process of the algorithm.
You can visualize the value, v, and policy, pi, for the i-th iteration by
    logger.log(i, v, pi)
You can also only update the visualization of the v values by
    logger.log(i, v)
"""


# use random library if needed
import random
import numpy as np

def value_iteration(env, gamma, max_iterations, logger):
    """
    Implement value iteration to return a deterministic policy for all states.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of value iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint: The value iteration may converge before reaching max_iterations.  
        In this case, you may want to exit the algorithm earlier. A way to check 
        if value iteration has already converged is to check whether 
        the max over (or sum of) L1 or L2 norms between the values before and
        after an iteration is small enough. For the Grid World environment, 1e-4
        is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model

    v = [0] * NUM_STATES
    print(v)
    pi = [0] * NUM_STATES
    # Visualize the value and policy 
    logger.log(0, v, pi)
    niter = 0;
    # At each iteration, you may need to keep track of pi to perform logging
   
### Please finish the code below ##############################################

###############################################################################
    for 𝑘 in range(max_iterations): 
        niter+=1;
        vprev = v
        print(vprev)
        vk = [0] * NUM_STATES
        pik = [0] * NUM_STATES
        for state in range(NUM_STATES):
            maxValue = float(-inf)
            for action in range (NUM_ACTIONS):
                val = 0
                t = env.trans_model[state][action]
                print(t)
                for item in range(len(t)):
                    (probability, nextstate, reward, terminal) = t[item]   
                    if terminal:
                        val += probability * reward #val = reward                        
                    else:
                        val += probability * (reward + gamma * vprev[nextstate])
                if val > maxValue:
                    maxValue = val
                    pik[state] = action
            vk[state] = round(maxValue,4)
        v = vk
        pi = (pik)
        logger.log(niter, v, pi)
        if vprev == v:
                print('converged at iteration: ', k)
                return pi
 
###############################################################################
    return pi


def policy_iteration(env, gamma, max_iterations, logger):
    """
    Implement policy iteration to return a deterministic policy for all states.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the reward discount factor
    max_iterations: integer
        the maximum number of policy iterations that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
        Hint 1: Policy iteration may converge before reaching max_iterations. 
        In this case, you should exit the algorithm. A simple way to check 
        if the algorithm has already converged is by simply checking whether
        the policy at each state hasn't changed from the previous iteration.
        Hint 2: The value iteration during policy evaluation usually converges 
        very fast and policy evaluation should end upon convergence. A way to check 
        if policy evaluation has converged is to check whether the max over (or sum of) 
        L1 or L2 norm between the values before and after an iteration is small enough. 
        For the Grid World environment, 1e-4 is an acceptable tolerance.
    logger: app.grid_world.App.Logger
        a logger instance to record and visualize the iteration process.
        During policy evaluation, the V-values will be updated without changing the current policy; 
        here you can update the visualization of value by simply calling logger.log(i, v).
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    TRANSITION_MODEL = env.trans_model
    
    v = [0.0] * NUM_STATES
    pi = [random.randint(0, NUM_ACTIONS-1)] * NUM_STATES
    niter = 0
    # Visualize the initial value and policy
    logger.log(0, v, pi)

### Please finish the code below ##############################################
###############################################################################
    while True:   
        v = policy_evaluation(pi,env,gamma) #evaluate policy pi until value converges
        policy_same = True;
        niter+=1
        for state in range(NUM_STATES):
            selectedAction = pi[state]
            print('selected action is', selectedAction)
            actionValue = [0.0] * NUM_ACTIONS
            for action in range (NUM_ACTIONS):
                t = env.trans_model[state][action]
                for item in range(len(t)):
                    (probability, nextstate, reward, terminal) = t[item]
                    if terminal:
                        actionValue[action]+=probability * reward #actionValue = reward
                    else:
                        actionValue[action]+=probability * (reward + gamma * v[nextstate])       
            bestAction = np.argmax(actionValue)
            print('best action is', bestAction)            
            if selectedAction != bestAction:
                policy_same = False
            pi[state] =  bestAction 
        logger.log(niter, v, pi)
        if policy_same:
            print('policy converged at iteration: ', niter);
            return pi
###############################################################################
    return pi


def policy_evaluation(pi,env,gamma):
    NUM_STATES = env.observation_space.n  
    niter = 0
    vk = [0.0] * NUM_STATES
    vkprev = [0.0] * NUM_STATES
    while True:
        niter+=1
        delta = 0
        vkprev = vk
        for state in range(NUM_STATES):
            val = 0
            print(pi[state])
            action = pi[state]                            
            t = env.trans_model[state][action]
            for item in range(len(t)):
                (probability, nextstate, reward, terminal) = t[item]   
                if terminal:
                    val += probability * reward
                else:
                    val += probability * (reward + gamma * vk[nextstate])
            delta = max(delta, abs(val - vk[state]))
            vk[state] = val
        if delta < 1e-6:
            print('Value converged at iteration: ', niter)
            return vk                           
    return vk

def q_learning(env, gamma, max_iterations, logger):
    """
    Implement Q-learning to return a deterministic policy for all states.

    Parameters
    ----------
    env: GridWorld
        the environment
    gamma: float
        the discount factor
    max_iterations: integer
        the maximum number of iterations (training episodes) that should be performed;
        the algorithm should terminate when max_iterations is exceeded.
    logger: app.grid_world.App.Logger
        a logger instance to perform test and record the iteration process.
    
    Returns
    -------
    pi: list or dict
        pi[s] should give a valid action,
        i.e. an integer in [0, env.action_space.n),
        as the optimal policy found by the algorithm for the state s.
    """
    NUM_STATES = env.observation_space.n
    NUM_ACTIONS = env.action_space.n
    
    v = [0] * NUM_STATES
    pi = [0] * NUM_STATES
    # Visualize the initial value and policy
    logger.log(0, v, pi)
    qtable = np.zeros((NUM_STATES, NUM_ACTIONS))
    
    #########################
    # Adjust superparameters as you see fit
    #
    # parameter for the epsilon-greedy method to trade off exploration and exploitation
    #eps = 1
    # learning rate for updating q values based on sample estimates
    #alpha = 0.1
    #########################

### Please finish the code below ##############################################
###############################################################################
    
    for i in range(1,max_iterations+1):
        s = env.reset()
        #alpha = 1/i
        #for the 1st 60% of my max_iterations, I choose to explore 80%
        if i<=0.6 * max_iterations:
            eps = 0.8
            alpha = 0.3
        #for the rest 40% of my max_iterations, I choose to exploit 80%   
        else:
            eps = 0.2
            alpha = 0.01
        complete = False
        while not complete:   
            a = epsilon_greedy(qtable,eps,s)
            nextState, reward, terminal, info = env.step(a) 
            if terminal:
                complete = True
                qtable[s,a] += alpha * (reward - qtable[s, a]) #qtable[s,a] = reward
            else:
                qtable[s,a] += reward + alpha * (gamma * np.max(qtable[nextState, :]) - qtable[s, a])
            s = nextState
        pi = np.argmax(qtable,axis=1)
        v = np.max(qtable,axis=1)
        logger.log(i, v, pi)
    
###############################################################################
    return pi

def epsilon_greedy(qvalues, eps, state):
    if random.random() < eps:
        # randomly select action from state
        action = np.random.choice(len(qvalues[state])) 
    else:
        # greedily select action from state
        action = np.argmax(qvalues[state]) 
    return action

if __name__ == "__main__":
    from app.grid_world import App
    import tkinter as tk

    algs = {
        "Value Iteration": value_iteration,
        "Policy Iteration": policy_iteration,
        "Q Learning": q_learning
   }
    worlds = {
        # o for obstacle
        # s for start cell
        "world1": App.DEFAULT_WORLD,
        "world2": lambda : [
            [10, "s", "s", "s", 1],
            [-10, -10, -10, -10, -10],
        ],
        "world3": lambda : [
            ["_", "_", "_", "_", "_"],
            ["_", "o", "_", "_", "_"],
            ["_", "o",   1, "_",  10],
            ["s", "_", "_", "_", "_"],
            [-10, -10, -10, -10, -10]
        ]
    }

    root = tk.Tk()
    App(algs, worlds, root)
    tk.mainloop()