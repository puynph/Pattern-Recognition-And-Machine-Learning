# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy as np
import time

# Environment
env = gym.make("Taxi-v3", render_mode='ansi')
state_size = env.observation_space.n
action_size = env.action_space.n

# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode
epsilon = 0.1

# Q tables for rewards
# Q_reward = -100000*numpy.ones((500,6)) # All same
Q_reward = -100000*np.random.random((state_size, action_size))
# Training w/ random sampling of actions

for episode in range(num_of_episodes):
    state = env.reset()[0]
    total_reward = 0.0
    for step in range(num_of_steps):
        curr_state = state
        if np.random.uniform() < epsilon:
            action = np.argmax(Q_reward[state, :])    # exploration
        else:
            action = np.random.randint(0, 6)          # exploitation
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        next_action = np.argmax(Q_reward[state, :])
        if done:
            Q_reward[curr_state, action] = reward
            break
        else:
            Q_reward[curr_state, action] += alpha * (
                        reward + gamma * Q_reward[state, next_action] - Q_reward[curr_state, action])

print("Training finished.\n")

# Testing

rewards = []
actions = []
for run in range(10):
    state = env.reset()[0]
    tot_reward = 0
    num_actions = 0

    for t in range(1000):
        action = np.argmax(Q_reward[state, :])
        state, reward, done, truncated, info = env.step(action)
        tot_reward += reward
        num_actions += 1
        print(env.render())
        time.sleep(1)
        if done:
            print("Total reward %d" % tot_reward)
            break
    rewards.append(tot_reward)
    actions.append(num_actions)

average_total_reward = np.mean(rewards)
average_num_actions = np.mean(actions)

print(f"Average Total Reward over 10 runs: {average_total_reward}")
print(f"Average Number of Actions over 10 runs: {average_num_actions}")

