import numpy as np
import random
import gym


env = gym.make('FrozenLake-v1', map_name="4x4", desc=["SFFF", "FHFH", "FFFH", "HFFG"],is_slippery=True)



def Q_leaning(Q_values, n_spisodes = 10000, n_max_tries = 100,alpha = 0.1
                  ,gamma = 0.992,epsilon = 1,max_epsilon = 1,min_epsilon = 0.0078,epsilon_decay = 0.005):
    
    history_rewards = []
    for episode in range(n_spisodes):
    
        s =  env.reset()

        if type(s) != int:
            s = s[0]
        
        r_current_episode = 0

        for t in range(n_max_tries):

            #Exploration-exploitation trade-off

            exploit_or_explore = random.uniform(0,1)

            if exploit_or_explore > epsilon:
                a = np.argmax(Q_values[s])

            else:
                a = env.action_space.sample()

            s_new, r, terminated, truncated, info = env.step(a)
            
                
            
            Q_values[s,a] = Q_values[s,a] * (1-alpha) + alpha*(r+gamma*np.max(Q_values[s_new,:]))

            s = s_new
            r_current_episode += r

            if terminated or truncated:
                break
        
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-epsilon_decay*episode)
        history_rewards.append(r_current_episode)
    

    return np.array(history_rewards),Q_values


        