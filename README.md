
## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the SARSA control algorithm using the Gymnasium FrozenLake-v1 environment and learn an optimal or near-optimal action-value function. The agent must learn to select appropriate actions to reach the goal while avoiding the hole states.

## Software Requirements

Python 3.x
Gymnasium
NumPy
Matplotlib
Jupyter Notebook / Google Colab


## Environment Description

FrozenLake-v1 is a discrete reinforcement-learning environment consisting of a 4×4 grid. The agent starts from the initial state and must reach the goal while avoiding holes. The environment has 16 states and 4 possible actions: Left, Down, Right, and Up. A reward of 1 is obtained for reaching the goal, while other transitions normally provide zero reward.

## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm

1. Create the custom FrozenLake-v1 environment with start state 5 and goal state 10.
2. Initialize the Q-table with zeros and set α, γ, ε, ε_min, and ε_decay.
3. Reset the environment and obtain the starting state S.
4. Select action A using the epsilon-greedy policy.
5. Execute A and observe reward R and next state S'.
6. Select the next action A' using the current epsilon-greedy policy.
7. Update Q(S,A) using the SARSA update rule.
8. Set S = S' and A = A', and repeat until the episode ends.
9. Decrease epsilon using ε = max(ε_min, ε × ε_decay).
10. Repeat training for all episodes and obtain the final Q-table, policy, and rewards.


## Python Program

```python
# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

for episode in range(num_episodes):

    # Reset environment
    state, info = env.reset()

    # Starting state should be 5
    assert state == START_STATE

    # Select initial action using epsilon-greedy policy
    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Select next action using epsilon-greedy policy
        if terminated or truncated:
            next_action = None
        else:
            next_action = epsilon_greedy_action(next_state, epsilon)

        # SARSA update
        if terminated or truncated:
            td_target = reward
        else:
            td_target = reward + gamma * Q[next_state, next_action]

        Q[state, action] += alpha * (
            td_target - Q[state, action]
        )

        total_reward += reward

        # End episode if terminal
        if terminated or truncated:
            break

        # Move to next state-action pair
        state = next_state
        action = next_action

    # Store reward
    episode_rewards.append(total_reward)

    # Store current epsilon
    epsilon_history.append(epsilon)

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

print("Training completed.")
print("Final epsilon:", epsilon)
```
---

## Output

Final Q-table:

<img width="236" height="290" alt="image" src="https://github.com/user-attachments/assets/09a396a8-fc6c-4ee4-af35-729bae6609e6" />


Estimated State-Value Function:

<img width="254" height="98" alt="image" src="https://github.com/user-attachments/assets/5547b742-a553-4d1d-b1b6-38d6bc184241" />



Learned Policy:

<img width="192" height="101" alt="image" src="https://github.com/user-attachments/assets/36e71668-710e-4aae-9996-63088f092131" />


Average reward over last 1000 episodes: 

<img width="351" height="31" alt="image" src="https://github.com/user-attachments/assets/21e6493a-8121-4b30-a54d-01fdc28b6a47" />

<img width="721" height="477" alt="image" src="https://github.com/user-attachments/assets/2caa4209-b3bb-4255-bffd-db813e9023f8" />

<img width="709" height="477" alt="image" src="https://github.com/user-attachments/assets/28383ebf-9778-4f82-bc15-c53bc177bf9a" />


## Result
```text
The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment.

```

---

## Inference
```text

The experiment demonstrates that SARSA can learn an effective policy through an epsilon-greedy exploration strategy. During training, the agent initially explores different actions and gradually exploits actions with higher Q-values as epsilon decreases.

```
---

