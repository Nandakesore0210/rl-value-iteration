# VALUE ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the value iteration algorithm.

## PROBLEM STATEMENT
The FrozenLake environment in OpenAI Gym is a gridworld problem that challenges reinforcement learning agents to navigate a slippery terrain to reach a goal state while avoiding hazards. Note that the environment is closed with a fence, so the agent cannot leave the gridworld.

## VALUE ITERATION ALGORITHM
* Value iteration is a method of computing an optimal MDP policy and its value.
* It begins with an initial guess for the value function, and iteratively updates it towards the optimal value function, according to the Bellman optimality equation.
* The algorithm is guaranteed to converge to the optimal value function, and in the process of doing so, also converges to the optimal policy.

## VALUE ITERATION FUNCTION
### Name: Nandakesore J
### Register Number:212223240103
```
envdesc  = ['FSFH','HFFH','HFGF', 'FFFH']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 10
P = env.env.P
```
```
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if np.max(np.abs(V-np.max(Q,axis=1)))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi
```
```
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_best_v, goal_state=goal_state)*100,
    mean_return(env, pi_best_v)))
```

## OUTPUT:

### Mention the optimal policy:
<img width="478" height="164" alt="image" src="https://github.com/user-attachments/assets/e5694506-9b82-4133-a48a-5e0a2043a236" />

### Optimal value function:
<img width="481" height="127" alt="image" src="https://github.com/user-attachments/assets/194c348b-68f3-42b5-90f8-d558799646fb" />

### Success Rate of optimal policy: 
<img width="696" height="39" alt="image" src="https://github.com/user-attachments/assets/5db33d47-a3ff-4e4a-a741-21b75c9001a2" />



## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.
