import gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)


    # You will need to update the Q_table in your iteration
    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()

        ##########################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING ABOVE THIS LINE
        # TODO: Replace the following with Q-Learning

        while (not done):
            # action = env.action_space.sample() # currently only performs a random action.
            # obs,reward,done,info = env.step(action)
            # episode_reward += reward # update episode reward
            previousObs = obs
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                predict = np.array([Q_table[(previousObs, i)] for i in range(env.action_space.n)])
                action = np.argmax(predict)

            obs, reward, done, info = env.step(action)
            episode_reward += reward  # update episode reward
            QSA = Q_table[(previousObs, action)]

            if not done:
                previousQ = QSA
                # If we distribute (1 − α) over our given equation: Q(s, a) + α(r + γ maxa0∈A Q(s',a'))
                # We can split this into two steps where the learning steps calculates
                # α(r + γ maxa0∈A Q(s',a') - Q(s, a))
                QLearningStep = LEARNING_RATE * (reward + (DISCOUNT_FACTOR * np.max(
                    np.array([Q_table[(obs, action)] for action in range(env.action_space.n)]))) - QSA)
                # Now we take Q(s,a) + the above step to get our desired value
                Q_table[(previousObs, action)] = previousQ + QLearningStep
            else:
                previousQ = QSA
                # If we distribute (1 − α) over our given equation: Q(s, a) + αr
                # We can split it into two steps first get r - Q(s,a)
                QLearningStep = LEARNING_RATE * (reward - QSA)
                # Then we do Q(s,a) + the above step to get our desired value
                Q_table[(previousObs, action)] = previousQ + QLearningStep


        # END of TODO
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE
        ##########################################################

        # record the reward for this episode
        episode_reward_record.append(episode_reward)
        # need to update hyper params for this to work
        EPSILON = EPSILON * EPSILON_DECAY

        
        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    #### DO NOT MODIFY ######
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
    #########################