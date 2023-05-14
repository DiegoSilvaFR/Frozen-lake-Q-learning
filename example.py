from IPython.display import clear_output
import numpy as np
from Q_training import Q_values, max_step_episode
import gym
import time

env = gym.make('FrozenLake-v1', map_name="4x4", desc=["SFFF", "FHFH", "FFFH", "HFFG"],is_slippery=True,render_mode="human")


for episode in range(3):
    s = env.reset()
    
    if type(s) != int:
            s = s[0]
    
    
    print(f'Episode: {episode+1}')
    time.sleep(0.5)

    for t in range(max_step_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        a = np.argmax(Q_values[s,:])
        s_new, r, terminated, truncated, info = env.step(a)


        if terminated or truncated:
            clear_output(wait=True)
            env.render()
            if r == 1:
                print("You've got into the end!")
                time.sleep(1)
            
            else:
                print("You fell into a hole!")
                time.sleep(1)
            clear_output(wait=True)
            break
        s = s_new

env.close()

