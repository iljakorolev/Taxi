import gym
import random
import numpy
import time
from IPython.display import clear_output

env = gym.make("Taxi-v2")
# next_state = -1000*numpy.ones((501,6))
# next_reward = -1000*numpy.ones((501,6))
# Training
# THIS YOU NEED TO IMPLEMENT
q_table = numpy.zeros([env.observation_space.n, env.action_space.n])

total_episodes = 50000        # Total episodes
total_test_episodes = 100     # Total test episodes
max_steps = 99                # Max steps per episode

learning_rate = 0.7           # Learning rate
gamma = 0.6                # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.01             # Exponential decay rate for exploration prob

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        # First we randomize a number
        expTrade = random.uniform(0, 1)

        # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if expTrade > epsilon:
            action = numpy.argmax(q_table[state, :])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * numpy.max(q_table[new_state, :]) - q_table[state, action])

        # Our new state is state
        state = new_state

        # If done : finish episode
        if done:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*numpy.exp(-decay_rate*episode)
    if episode % 5000 == 0:
        clear_output(wait=True)
        print("Episode:", episode)

print("Training finished.\n")

print("Running simulation - testing training data.\n")
# Testing
test_tot_reward = 0
test_tot_actions = 0
past_observation = -1
observation = env.reset()
for t in range(50):
    test_tot_actions = test_tot_actions+1
    action = numpy.argmax(q_table[observation, :])
    if observation == past_observation:
        # This is done only if gets stuck
        action = random.sample(range(0, 6), 1)
        action = action[0]
    past_observation = observation
    observation, reward, done, info = env.step(action)
    test_tot_reward = test_tot_reward+reward
    env.render()
    time.sleep(1)
    if done:
        break
print("Total reward: ")
print(test_tot_reward)
print("Total actions: ")
print(test_tot_actions)
