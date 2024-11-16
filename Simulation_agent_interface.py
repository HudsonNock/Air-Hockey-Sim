import Air_hockey_sim_vectorized as sim
import numpy as np
import time
import pygame

agents = 1
Ts = 0.2 # time step between agent actions
Vmax = 24.0
N = 100 # simulation steps between each time step (recommened 1, use N = 100 for visualization)
delay = 0.06 # expected delay between collecting data and the agent choosing an action
N_delay = int(np.ceil(delay * N / Ts))
if N == 1:
    N_delay = 1
step_size = (Ts-delay) / N
step_size_delay = delay / N_delay

sim.initalize(agents, V_max=Vmax)
bounds_mallet = sim.get_mallet_bounds()

actions = 0
act_p_hour = 3600 / agents / Ts
hour = 1

clock = pygame.time.Clock()
clock.tick(60)

while True:
    #choose a random action
    random_base = np.random.uniform(0.0,1.0, (agents,2,2))
    actions_xf = bounds_mallet[:,:,:,0] + random_base * (bounds_mallet[:,:,:,1] - bounds_mallet[:,:,:,0])
    random_base = np.random.uniform(0.0, 1.0, (agents,2,2))
    actions_V = random_base * Vmax

    sim.take_action(actions_xf, actions_V)

    for i in range(N):
        mallet_pos, puck_pos = sim.step(step_size)
        
        error, index = sim.check_state()
        if error != 0:
            sim.reset_sim(index)

        sim.check_goal() #returns 1 or -1 depending on what agent scored a goal
        if N != 1:
            sim.display_state(0)
    
    for i in range(N_delay):
        sim.step(step_size_delay)

        error, index = sim.check_state()
        if error != 0:
            sim.reset_sim(index)

        sim.check_goal()
        if N != 1:
            sim.display_state(0)

    #print(actions)
    actions += 1
    if actions > act_p_hour * hour:
        print(hour)
        print(clock.tick(60) / 1000.0)
        hour += 1
