import Air_hockey_sim_vectorized as sim
import numpy as np
import time
import pygame

agents = 100
Vmax = 24.0
Ts = 0.2
N = 100
step_size = Ts / N
delay = 0.06
sim.initialize(agents)

bounds_mallet = sim.get_mallet_bounds()

steps = 3


def check_difference(actions_xf, actions_V):
    mallet_pos_past1 = np.empty((steps,agents,2,2), dtype="float32")
    puck_pos_past1 = np.empty((steps,agents,2), dtype="float32")

    puck_diff = np.zeros((steps,agents,2), dtype="float32")

    mallet_pos_past2 = np.empty((steps,agents,2,2), dtype="float32")
    puck_pos_past2 = np.empty((steps,agents,2), dtype="float32")
        

    #timer = 0
    for action_count in range(steps):
        sim.take_action(actions_xf[action_count], actions_V[action_count])
        for i in range(100):
            #print(timer)
            mallet_pos, puck_pos = sim.step(Ts/100)
            #timer += step_size
            error, index = sim.check_state()
            if error != 0:
                print(error)
                sim.display_state(index)
                print(actions_xf)
                print("---")
                print(actions_V)
                while True:
                    pass
            sim.check_goal()
            #if action_count == 1 and i == N-1:
            sim.display_state(0)
                #print(puck_pos)
            time.sleep(0.01)
            #for j in range(200000):
            #    pass
        
        mallet_pos_past1[action_count] = mallet_pos
        puck_pos_past1[action_count] = puck_pos

    sim.reset_sim()
    #print("R")
    #timer = 0
    for i in range(steps):
        if i == steps - 1:
            pass
        sim.take_action(actions_xf[i], actions_V[i])
        #print(timer)
        mallet_pos, puck_pos = sim.step(Ts)
        #timer += Ts
        error, index = sim.check_state()
        if error != 0:
            print(error)
            sim.display_state(index)
            print(actions_xf[:,index,:])
            print("---")
            print(actions_V[:,index,:])
            #print(index)
            while True:
                pass
        sim.check_goal()
        #if i == 1:
        sim.display_state(0)
            #print("---")
            #print(puck_pos)
        time.sleep(1)
        #for j in range(200000000):
        #    pass
        
        mallet_pos_past2[i] = mallet_pos
        puck_pos_past2[i] = puck_pos

    sim.reset_sim()

    puck_diff = puck_pos_past1 - puck_pos_past2

    #mallet_diff = mallet_pos_past1 - mallet_pos_past2
    #puck_diff = puck_pos_past1 - puck_pos_past2
    print(np.max(puck_diff))
    #print(puck_diff[2,0,:])
    if np.max(puck_diff) > 0.04:
        index = np.argmax(puck_diff)
        index = np.unravel_index(index, puck_diff.shape)
        index = index[1]
        print(actions_xf[:,index,:])
        print(actions_V[:,index,:])
        while True:
            pass

    #if np.max(puck_diff) > 1e-2:
    #    print(puck_diff)
    #print("Diff")
    #print(puck_diff[:,0,:])

def get_speed():
    actions = 0
    act_p_hour = 3600 / agents / Ts
    hour = 1
    clock = pygame.time.Clock()
    clock.tick(60)
    while True:
        random_base = np.random.uniform(0.0,1.0, (agents,2,2))
        actions_xf = bounds_mallet[:,:,:,0] + random_base * (bounds_mallet[:,:,:,1] - bounds_mallet[:,:,:,0])
        random_base = np.random.uniform(0.0, 1.0, (agents,2,2))
        actions_V = random_base * Vmax

        sim.take_action(actions_xf, actions_V)

        for i in range(N):
            mallet_pos, puck_pos = sim.step(Ts/N)
            sim.step(delay)
            error, index = sim.check_state()
            if error != 0:
                sim.reset_sim(index)

            sim.check_goal()
            sim.display_state(0)

        #print(actions)
        actions += 1
        if actions > act_p_hour * hour:
            print(hour)
            print(clock.tick(60) / 1000.0)
            hour += 1

def difference_loop():
    actions_xf = np.empty((steps, agents, 2, 2))
    actions_V = np.empty((steps,agents,2,2))

    while True:
        for i in range(steps):
            random_base = np.random.uniform(0.0,1.0, (agents,2,2))
            actions_xf[i,:,:,:] = bounds_mallet[:,:,:,0] + random_base * (bounds_mallet[:,:,:,1] - bounds_mallet[:,:,:,0])
            random_base = np.random.uniform(0.0, 1.0, (agents,2,2))
            actions_V[i,:,:,:] = random_base * Vmax

        

        actions_xf[:,0,:,:] = np.array([[[0.68740402, 0.71838327],\
        [1.27835089, 0.40322591]],\
        [[0.88684311, 0.75645865],\
        [1.47576895, 0.21888755]],\
        [[0.1282919,  0.81489309],\
        [1.61793476, 0.8421366 ]]])

        actions_V[:,0,:,:] = np.array([[[19.39013601,  3.33756613],\
        [21.41653566, 23.56477316]],\
        [[12.12014882, 20.53147996],\
        [20.3956797,   5.84942338]],\
        [[ 6.42990171, 15.83603031],\
        [14.95814848,  2.9594774 ]]])

        check_difference(actions_xf, actions_V)



#actions_xf = np.empty((steps, agents, 2, 2))
#actions_V = np.empty((steps,agents,2,2))


#mallet_pos = np.empty((agents,2,2), dtype="float32")
#puck_pos = np.empty((agents,2), dtype="float32")
#past_puck_pos = np.empty((agents,2), dtype="float32")
get_speed()






