# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy as np
import time

# Environment
env = gym.make("Taxi-v3", render_mode='ansi')

state_space = env.observation_space.n
action_space = env.action_space.n
# Training parameters for Q learning

total_episodes = 25000        # Total number of training episodes
total_test_episodes = 100     # Total number of test episodes
max_steps = 200               # Max steps per episode

learning_rate = 0.01          # Learning rate
gamma = 0.99                  # Discounting rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.001  # Minimum exploration probability
decay_rate = 0.01  # Exponential decay rate for exploration prob

# Q tables for rewards
# Q_reward = -100000*numpy.ones((500,6)) # All same
Q = np.zeros((state_space, action_space))


def epsilon_greedy_policy(Q, state):
    # if random number > greater than epsilon --> exploitation
    if (random.uniform(0, 1) > epsilon):
        action = np.argmax(Q[state])
    # else --> exploration
    else:
        action = env.action_space.sample()

    return action


for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    for step in range(max_steps):
        #
        action = epsilon_greedy_policy(Q, state)

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, truncated, info = env.step(action)
        next_action = np.argmax(Q[new_state, :])

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        Q[state, action] += learning_rate * (reward + gamma * Q[state, next_action] - Q[state, action])
        # If done : finish episode
        if done:
            break

        # Our new state is state
        state = new_state

rewards = []

frames = []
for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print("****************************************************")
    print("EPISODE ", episode)
    for step in range(max_steps):
        env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Q[state][:])
        new_state, reward, done, truncated, info = env.step(action)
        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            # print ("Score", total_rewards)
            break
        state = new_state
env.close()
print("Score over time: " + str(sum(rewards) / total_test_episodes))