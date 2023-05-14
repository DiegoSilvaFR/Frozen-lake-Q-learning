from Q_leaning import Q_leaning
import matplotlib.pyplot as plt
import numpy as np
import gym



env = gym.make('FrozenLake-v1', map_name="4x4", desc=["SFFF", "FHFH", "FFFH", "HFFG"],is_slippery=True)

A = env.action_space.n #number of actions
S = env.observation_space.n #state space
Q_values = np.zeros((S,A))


max_step_episode = 150
n_episodes = 20000 
history , Q_values = Q_leaning(Q_values, n_spisodes=n_episodes,n_max_tries=max_step_episode,alpha=0.5)



if __name__ == "__main__":

    plt.plot(np.mean(np.split(history,n_episodes*0.1),axis = 0))
    plt.ylabel(f"Mean Reward per {n_episodes*0.1} episodes")
    plt.show()

    


