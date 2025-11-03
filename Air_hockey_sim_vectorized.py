import math
from scipy.optimize import fsolve, newton, brentq
from scipy.special import lambertw
import numpy as np
import random
from shapely.geometry import Polygon, LineString
from itertools import product
from chandrupatla import chandrupatla
import time as tme
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize pygame
print("STARTING")

height = 0.992
width = 1.993

bounds = np.array([width, height])
plr = 500
screen_width = int(width*plr)
screen_height = int(height*plr)


surface = np.array([width, height])

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 100, 0)
blue = (0,0,255)

#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#out = cv2.VideoWriter("Model_play_new_env.avi", fourcc, 60.0, (screen_width, screen_height))
#frame_count = 0

def initalize(envs, mallet_r=0.05, puck_r=0.05, goal_w=0.35, V_max=24, pully_radius=0.035306, ab_vars=None, puck_inits=None):
    global game_number 
    global time
    global mallet_radius
    global puck_radius
    global goal_width
    global Vmax
    global Vmin
    global pullyR
    global puck_pos
    global puck_vel
    global mallet_pos
    global mallet_vel
    global x_0
    global x_p
    global x_pp
    global reset_mask
    global reset_sim_mask
    global margin
    global bounds_mallet
    global bounds_puck
    global drag
    global friction
    global mass
    global puck_init
    global sink_bounds
    global sinks
    global res_model
    global mallet_res_model

    time = 0

    game_number = envs
    mallet_radius = mallet_r
    puck_radius = puck_r

    sink_bounds = np.array([[puck_r, width - puck_r], [puck_r, height-puck_r]])
    sinks = np.random.rand(game_number, 2) * (sink_bounds[:,1] - sink_bounds[:,0]) + sink_bounds[:,0]
    
    goal_width = goal_w

    Vmax = V_max
    #(game, player, x/y)
    Vmin = np.zeros((game_number,2,2), dtype="float32")
    pullyR = pully_radius

    puck_pos = puck_inits.copy()
    puck_vel = np.full((game_number, 2), 0, dtype="float32")
    puck_init = puck_inits
    #puck_vel[:,0] = 1.359155
    #puck_vel[:,1] = 0.98599756

    mallet_pos = np.empty((game_number,2,2), dtype="float32")
    mallet_vel = np.empty((game_number,2,2), dtype="float32")

    x_0 = np.empty((game_number,2,2), dtype="float32")
    x_0[:,0,:] = [0.25, 0.5]
    x_0[:,1,:] = bounds - x_0[:,0,:]
    x_p = np.zeros((game_number,2,2), dtype="float32")
    x_pp = np.zeros((game_number,2,2), dtype="float32")

    reset_mask = np.full((game_number), False)
    reset_sim_mask = np.full((game_number), False)

    margin = 0.01
    #shape (2, 2, 2), player, x/y, lower/upper
    bounds_mallet = np.array([[[mallet_radius+margin, width/2 - mallet_radius - margin],\
                    [mallet_radius + margin, height - mallet_radius - margin]],\
                        [[width/2+mallet_radius+margin, width - mallet_radius - margin],\
                        [mallet_radius + margin, height - mallet_radius - margin]]])
    # game, player,x/y,lower/upper
    bounds_mallet = np.tile(np.expand_dims(bounds_mallet, axis=0), (game_number,1,1,1))

    # x/y, lower/upper
    bounds_puck = np.array([[puck_radius, width - puck_radius], [puck_radius, height - puck_radius]])
    bounds_puck = np.tile(np.expand_dims(bounds_puck, axis=0), (game_number,1,1))
                    
    drag = 0.0012273 + np.random.random((game_number,))*0.0004
    friction = 0.00103794 + np.random.random((game_number,))*0.0004
    mass = 0.00736196 + np.random.random((game_number,)) * 0.0008

    class HeteroscedasticNN(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=10, output_dim=2):
            super(HeteroscedasticNN, self).__init__()
            
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.predictions = nn.Linear(hidden_dim, output_dim)
            self.log_sigmas = nn.Linear(hidden_dim, output_dim)
            
            # Scale parameter for sigmas (initialized to 1.0, set later for denormalization)
            self.register_buffer('sigma_scale', torch.ones(output_dim))
            
        def forward(self, x):
            x = F.softplus(self.fc1(x))
            x = F.softplus(self.fc2(x))
            
            # Predictions
            preds = self.predictions(x)
            
            # Sigmas: softplus to ensure positive, add epsilon, then scale
            log_sig = self.log_sigmas(x)
            sigmas = (F.softplus(log_sig) + 1e-6) * self.sigma_scale
            
            # Concatenate predictions and sigmas
            return torch.cat([preds, sigmas], dim=1)

    res_model = HeteroscedasticNN(input_dim=2, hidden_dim=8, output_dim=2)
    res_model.load_state_dict(torch.load('collision_model_heteroscedastic.pt'))
    res_model.to('cpu')
    res_model.eval()

    mallet_res_model = HeteroscedasticNN(input_dim=2, hidden_dim=8, output_dim=2)
    mallet_res_model.load_state_dict(torch.load('mallet_collision_model_heteroscedastic.pt'))
    mallet_res_model.to('cpu')
    mallet_res_model.eval()
                
    global C5
    global C6
    global C7
    global ab
    global CE
    global A2CE
    global A3CE
    global A4CE
    global C1
    global C2
    global C3
    global C4

    #a1 = 3.579*10**(-6)
    #a22 = 0.00571
    #a3 = (0.0596+0.0467)/2
    #b1 = -1.7165*10**(-6)
    #b2 = -0.002739
    #b3 = 0

    #(game, player, x/y)
    C5 = np.empty((game_number,2,2))
    C5[:,:,0] = ab_vars[:,:,0] - ab_vars[:,:,3]
    C5[:,:,1] = ab_vars[:,:,0] + ab_vars[:,:,3] #[a1-b1, a1+b1]

    C6 = np.empty((game_number,2,2))
    C6[:,:,0] = ab_vars[:,:,1] - ab_vars[:,:,4]
    C6[:,:,1] = ab_vars[:,:,1] + ab_vars[:,:,4] #[a2-b2, a2+b2]

    C7 = np.empty((game_number,2,2))
    C7[:,:,0] = ab_vars[:,:,2] - ab_vars[:,:,5]
    C7[:,:,1] = ab_vars[:,:,2] + ab_vars[:,:,5] #[a3-b3, a3+b3]

    A = np.sqrt(np.square(C6) - 4 * C5 * C7)
    B = 2 * np.square(C7) * A

    #(gamenumber, player, x/y, a/b)
    ab = np.stack([(-C6-A)/(2*C5),(-C6+A)/(2*C5)], axis=-1)

    #(game_number, player, x/y, coefficients in A e^(at) + B e^(bt) + Ct + D)
    CE = np.stack(
        [(-np.square(C6) + A*C6+2*C5*C7)/B,
        (np.square(C6)+A*C6-2*C5*C7)/B,
        1/C7,
        -C6/np.square(C7)], axis=-1
    )

    B = 2*C7*A

    #(game number, player, x/y, coefficients in A e^(at) + Be^(bt) + C)
    A2CE = np.stack(
        [-(-C6+A)/B,
        -(C6+A)/B,
        1/C7], axis=-1
    )

    #(game number, player, x/y, coefficients in A e^(at) + Be^(bt))
    A3CE = np.stack(
        [-1/A,
        1/A], axis=-1
    )

    B = 2*C5*A

    #(game number, player, x/y, coefficients in A e^(at) + Be^(bt))
    A4CE = np.stack(
        [(C6+A)/B,
        (-C6+A)/B], axis=-1
    )

    #(game,player,x/y)
    C1 = np.full((game_number, 2, 2), 14*pullyR / 2)
    C2 = C5*x_pp + C6 * x_p + C7 * x_0
    C3 = C5 * x_p + C6 * x_0
    C4 = C5 * x_0

    global x_f
    global a
    global a2
    global Ts
    global x_over
    global overshoot_mask


    #(game, player, x/y)
    x_f = np.copy(x_0)
    #(game, player, x/y)
    a = np.full((game_number, 2, 2), 0.0)
    #a = np.array([[[0.1,0.1], [0.1,0.1]], [[0.08,0.08], [0.08,0.08]]])
    a2 = np.full((game_number, 2, 2), 0.0)

    Ts = 0.2

    x_over = np.zeros((game_number, 2, 2), dtype="float32")

    #(game, player, x/y, min/max)
    #mallet_square = np.stack([
    #    np.copy(x_0)-(mallet_radius+puck_radius),
    #    np.copy(x_0)+mallet_radius+puck_radius
    #], axis=-1)

    overshoot_mask = np.full((game_number, 2, 2), True)


def dP(t, B, f, mass, C, D):
    tpZero = get_tPZero(mass,C)
    t = np.minimum(tpZero, t)
    p = (mass/B) * np.log(np.cos((np.sqrt(B*f)*(C*mass+t))/mass)) + D
    return p

def velocity(t, B, f, mass, C):
    tpZero = get_tPZero(mass,C)
    v = np.zeros_like(t)
    v_mask = t < tpZero
    v[v_mask] = -np.sqrt(f/B)[v_mask] * np.tan(((np.sqrt(B*f)*(C*mass+t))/mass)[v_mask])
    return v

def velocity_scalar(t, B, f, mass, C):
    tpZero = -C*mass
    if t < tpZero:
        return -np.sqrt(f/B) * np.tan(((np.sqrt(B*f)*(C*mass+t))/mass))
    return 0

def get_tPZero(mass, C):
    return -C*mass

def getC(v_norm, B, f):
    return - np.arctan(v_norm * np.sqrt(B/f)) / np.sqrt(B*f)

def getD(B, mass, f, C):
    return - (mass/B) * np.log(np.cos(np.sqrt(B*f)*C))

def distance_to_wall(t, dist, B, f, mass, C, D):
    return dP(t, B, f, mass, C, D) - dist


def corner_collision(A, pos, vel, t, C, D, v_norm, dir, dPdt,B,f,mass, vt):
    a = vel[0]**2 + vel[1]**2
    b = 2*pos[0]*vel[0] + 2*pos[1]*vel[1]-2*A[0]*vel[0]-2*A[1]*vel[1]
    c = pos[0]**2+pos[1]**2+A[0]**2+A[1]**2-puck_radius**2-2*A[0]*pos[0]-2*A[1]*pos[1]

    disc = b**2 - 4*a*c
    if disc < 0:
        new_vel = [0,0]
        new_pos = [0,0]
        for j in range(2):
                new_pos[j] = pos[j] + dPdt * dir[j]
                new_vel[j] = vt * dir[j]
        return new_pos, new_vel, False, 0

    s = (-b - np.sqrt(disc)) / (2*a)

    new_pos = [0,0]
    new_pos[0] = pos[0] + s * vel[0]
    new_pos[1] = pos[1] + s * vel[1]
    wall_val = s * v_norm

    thit_min = -1
    if wall_val < 1e-8:
        thit_min = 1e-8
    elif dPdt > wall_val:
        thit_min, converged = brentq(distance_to_wall, 0, t, args=(wall_val,B,f,mass,C,D),\
                                        xtol=1e-5, maxiter=30, full_output=True, disp=False)
        if not converged.converged:
            pass
            #print("Failed to converge corner collision")
            #print(pos)
            #print(vel)
            #print(wall_val)
            #print(wall_val/v_norm)
            #print(v_norm)
            #print(C)
            #print(D)
            #print((new_pos[0] - pos[0])*tPZero[0]/(final_pos[0]-pos[0]))

    new_vel = [0,0]
    if thit_min == -1:
        for j in range(2):
                new_pos[j] = pos[j] + dPdt * dir[j]
                new_vel[j] = vt * dir[j]
        return new_pos, new_vel, False, 0

    vt_hit = velocity_scalar(thit_min, B, f, mass, C)
    for j in range(2):
        new_vel[j] = vt_hit * dir[j]

    n = np.array([new_pos[0] - A[0], new_pos[1] - A[1]])
    n = n / np.sqrt(np.dot(n, n))
    tangent = np.array([n[1], -n[0]])
    vel_r = np.array(new_vel)

    v_in = np.linalg.norm(vel_r)
    n_vel = np.dot(vel_r, n)
    t_vel = np.dot(vel_r, tangent)
    angle_in = np.degrees(np.arctan(abs(t_vel/n_vel)))

    with torch.no_grad():
        col_out = res_model(torch.FloatTensor(np.array([angle_in, v_in])[np.newaxis,:])).numpy()[0]
    angle_out = np.clip(np.random.normal(col_out[0], col_out[2]), -89, 89)
    v_out = max(np.random.normal(col_out[1], col_out[3]), 0.001)    

    new_vel = n * v_out * np.cos(angle_out * np.pi/180.0) * (-n_vel / abs(n_vel)) + tangent * v_out * np.sin(angle_out * np.pi/180.0) * (t_vel / abs(t_vel))

    return new_pos, new_vel, True, t - thit_min

def check_in_goal_corner_bounce(vel, pos, v_norm):
    A = [0,0]
    bounces = False
    w1 = [vel[1] * puck_radius, -vel[0] * puck_radius] / v_norm
    w2 = -w1
    s1 = np.inf
    s2 = np.inf

    if pos[0] < puck_radius:
        if vel[0] > 0:
            if pos[0] + w1[0] < 0:
                s1 = (-pos[0] - w1[0]) / vel[0]
                if surface[1]/2 - goal_width/2 > vel[1]*s1+pos[1]+w1[1] or vel[1]*s1+pos[1]+w1[1] >  surface[1]/2 + goal_width/2:
                    bounces = True
                else:
                    s1 = np.inf
            if pos[0] + w2[0] < 0:
                s2 = (-pos[0] - w2[0]) / vel[0]
                if surface[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w2[1] or vel[1]*s2+pos[1]+w2[1] >  surface[1]/2 + goal_width/2:
                    bounces = True
                else:
                    s2 = np.inf
        elif vel[0] < 0:
            if pos[0] + w1[0] > 0:
                s1 = (-pos[0]-w1[0]) / vel[0]
                if surface[1]/2 - goal_width/2 > vel[1]*s1+pos[1]+w1[1] or vel[1]*s1+pos[1]+w1[1] >  surface[1]/2 + goal_width/2:
                    bounces = True
                else:
                    s1 = np.inf
            if pos[0] + w2[0] > 0:
                s2 = (-pos[0]-w2[0]) / vel[0]
                if surface[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w2[1] or vel[1]*s2+pos[1]+w2[1] >  surface[1]/2 + goal_width/2:
                    bounces = True
                else:
                    s2 = np.inf
        elif vel[0] == 0:
                bounces = True
        if bounces:
            A[0] = 0
    elif pos[0] > surface[0]-puck_radius:
        if vel[0] > 0:
            if pos[0] + w1[0] < surface[0]:
                s1 = (-pos[0] - w1[0] + surface[0]) / vel[0]
                if surface[1]/2 - goal_width/2 > vel[1]*s1+pos[1]+w1[1] or vel[1]*s1+pos[1]+w1[1] >  surface[1]/2 + goal_width/2:
                    bounces = True
                else:
                    s1 = np.inf
            if pos[0] + w2[0] < surface[0]:
                s2 = (-pos[0] - w2[0] + surface[0]) / vel[0]
                if surface[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w2[1] or vel[1]*s2+pos[1]+w2[1] >  surface[1]/2 + goal_width/2:
                    bounces = True
                else:
                    s2 = np.inf
        elif vel[0] < 0:
            if pos[0] + w1[0] > surface[0]:
                s1 = (-pos[0]-w1[0]+surface[0]) / vel[0]
                if surface[1]/2 - goal_width/2 > vel[1]*s1+pos[1]+w1[1] or vel[1]*s1+pos[1]+w1[1] >  surface[1]/2 + goal_width/2:
                    bounces = True
                else:
                    s1 = np.inf
            if pos[0] + w2[0] > surface[0]:
                s2 = (-pos[0]-w2[0]+surface[0]) / vel[0]
                if surface[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w2[1] or vel[1]*s2+pos[1]+w2[1] >  surface[1]/2 + goal_width/2:
                    bounces = True
                else:
                    s2 = np.inf
        elif vel[0] == 0:
                bounces = True
        if bounces:
            A[0] = surface[0] 

    if bounces:
        if vel[0] != 0:
            if s1 < s2:
                if surface[1]/2 - goal_width/2 > vel[1]*s1+pos[1]+w1[1]:
                    A[1] = surface[1]/2 - goal_width/2
                elif vel[1]*s1+pos[1]+w1[1] >  surface[1]/2 + goal_width/2:
                    A[1] = surface[1]/2 + goal_width/2
            else:
                if surface[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w2[1]:
                    A[1] = surface[1]/2 - goal_width/2
                elif vel[1]*s2+pos[1]+w2[1] >  surface[1]/2 + goal_width/2:
                    A[1] = surface[1]/2 + goal_width/2
        else:
            if vel[1] > 0:
                A[1] = surface[1]/2 + goal_width/2
            else:
                A[1] = surface[1]/2 - goal_width/2
    return A, bounces


def puck_mallet_collision(mask, pos, vel, dir, dt_col, t_init, C, D, Bm, fm, massm, vpf, x_m):

    dt_dynamic = np.empty_like(dt_col)
    dt_sum = np.zeros_like(dt_col)

    collision_unknown = np.full((len(dt_col)), True)
    collision_mask = np.full((len(dt_col)), False)

    new_pos = np.empty_like(pos)
    new_vel = np.empty_like(vel)

    mask_indicies = np.where(mask)[0]

    CEm = CE[mask]
    A2CEm = A2CE[mask]
    A3CEm = A3CE[mask]
    A4CEm = A4CE[mask]
    abm = ab[mask]
    C1m = C1[mask]
    C2m = C2[mask]
    C3m = C3[mask]
    C4m = C4[mask]
    am = a[mask]
    a2m = a2[mask]
    full_timem = np.zeros_like(dt_col)

    v_m = get_xp_mask(t_init+dt_sum, CEm, A2CEm, A3CEm, A4CEm, abm, am, a2m, C1m, C2m, C3m, C4m)

    x_p = pos
    v_p = vel

    colided_mask = np.full_like(collision_unknown, False)

    tolerance = 1e-6
    radius_col = (puck_radius + mallet_radius + tolerance)**2

    #global_mask = mask_indicies
    mallet_mask = np.empty_like(collision_unknown)
    v_m_a2 = np.full((len(collision_unknown), 2, 2), 1.0)

    #counter = 0
    while True:

        if np.any(np.tile(full_timem[:,np.newaxis,np.newaxis], (1,2,2)) > a2m):
            v_m_a2 = get_xp_mask_a2(CEm, A2CEm, A3CEm, A4CEm, abm, am, a2m, C1m, C2m, C3m, C4m)
        else:
            v_m_a2 = np.empty_like(am)

        if not np.any(colided_mask):
            dt_dynamic[collision_unknown] = puck_mallet_collision_t(x_p, x_m, v_p,\
                                                                    v_m, vpf[collision_unknown],\
                                                                     dir[collision_unknown], dt_sum[collision_unknown],\
                                                                        v_m_a2, am, a2m, C1m, CEm)
        else:
            dt_dynamic[collision_unknown] = puck_mallet_collision_t(x_p, x_m, v_p,\
                                                                    v_m, vpf[collision_unknown],\
                                                                    dir[collision_unknown], dt_sum[collision_unknown],\
                                                                        v_m_a2[~colided_mask], am[~colided_mask], a2m[~colided_mask], C1m[~colided_mask], CEm[~colided_mask])

        mallet_mask2 = ~(dt_dynamic[collision_unknown] == -1) & ((dt_sum+np.maximum(dt_dynamic, 1e-6))[collision_unknown] <= dt_col[collision_unknown])

        collision_unknown = (collision_unknown & ~(dt_dynamic == -1))
        dt_dynamic[collision_unknown] = np.maximum(dt_dynamic[collision_unknown], 1e-6)
        dt_sum[collision_unknown] += dt_dynamic[collision_unknown]
        collision_unknown = collision_unknown & ~(dt_sum > dt_col)

        if not np.any(collision_unknown):
            break

        dpp = dP(dt_sum[collision_unknown], Bm[collision_unknown], fm[collision_unknown],\
                                           massm[collision_unknown], C[collision_unknown], D[collision_unknown])
        x_p = pos[collision_unknown] + np.tile(dpp[:,np.newaxis], (1,2)) * dir[collision_unknown]
        vpp = velocity(dt_sum[collision_unknown], Bm[collision_unknown], fm[collision_unknown], massm[collision_unknown], C[collision_unknown])
        v_p = np.tile(vpp[:,np.newaxis], (1,2)) * dir[collision_unknown]
        
        if np.any(colided_mask):
            mallet_mask3 = np.full_like(colided_mask, False)
            mallet_mask3[np.where(~colided_mask)[0][mallet_mask2]] = True
            mallet_mask = (~colided_mask) & mallet_mask3
        else:
            mallet_mask = mallet_mask2

        full_timem = t_init[collision_unknown] + dt_sum[collision_unknown]
        CEm = CEm[mallet_mask]
        A2CEm = A2CEm[mallet_mask]
        A3CEm = A3CEm[mallet_mask]
        A4CEm = A4CEm[mallet_mask]
        abm = abm[mallet_mask]
        C1m = C1m[mallet_mask]
        C2m = C2m[mallet_mask]
        C3m = C3m[mallet_mask]
        C4m = C4m[mallet_mask]
        am = am[mallet_mask]
        a2m = a2m[mallet_mask]

        #print(len(full_timem))
        #counter += 1
        #print(counter)
        if abm.shape[0] != full_timem.shape[0]:
            pass

        x_m = get_pos_mask(full_timem, CEm, A2CEm, A3CEm, A4CEm, abm, am, a2m, C1m, C2m, C3m, C4m)
        v_m = get_xp_mask(full_timem, CEm, A2CEm, A3CEm, A4CEm, abm, am, a2m, C1m, C2m, C3m, C4m)

        colided_mask_m1 = ((x_p[:,0] - x_m[:,0,0])**2 + (x_p[:,1]-x_m[:,0,1])**2) < radius_col
        colided_mask_m2 = (((x_p[:,0] - x_m[:,1,0])**2 + (x_p[:,1]-x_m[:,1,1])**2) < radius_col) & ~colided_mask_m1
    
        """
        if len(full_timem) < 6:
            screen.fill(white)
            index = 0

            positionA1 = [x_m[index,0,0] * plr, x_m[index,0,1] * plr]
            positionA2 = [x_m[index,1,0] * plr, x_m[index,1,1] * plr]

            pygame.draw.rect(screen, red, pygame.Rect(screen_width/2 - 5, 0, 10, screen_height))
            pygame.draw.circle(screen, black, (int(round(positionA1[0])), int(round(positionA1[1]))), mallet_radius*plr)
            pygame.draw.circle(screen, black, (int(round(positionA2[0])), int(round(positionA2[1]))), mallet_radius*plr)
            pygame.draw.rect(screen, red, pygame.Rect(0, screen_height/2-goal_width*plr/2, 5, goal_width*plr))
            pygame.draw.rect(screen, red, pygame.Rect(screen_width-5, screen_height/2-goal_width*plr/2, 5, goal_width*plr))
            pygame.draw.circle(screen, blue, (int(round(x_p[index,0]*plr)), int(round(x_p[index,1]*plr))), puck_radius*plr)

            pygame.display.flip()

            tme.sleep(1.0)
        """

        collision_indices_m1 = []
        collision_indices_m2 = []

        if np.any(colided_mask_m1):
            v_rel = (v_p - v_m[:,0,:])[colided_mask_m1]
            normal = (x_p - x_m[:,0,:])[colided_mask_m1]
            normal = normal / np.tile(np.sqrt(np.sum(normal**2, axis=1))[:,np.newaxis], (1,2))

            colided_mask_m1_indicies = np.where(colided_mask_m1)[0]
            passed_mask = (np.sum(v_rel*normal, axis=-1) < 0)
            failed_indicies = colided_mask_m1_indicies[np.logical_not(passed_mask)]
            colided_mask_m1[failed_indicies] = False

        if np.any(colided_mask_m1):
            v_rel = v_rel[passed_mask]
            normal = normal[passed_mask]
            tangent = np.stack([-normal[:,1], normal[:,0]], axis=1)
            collision_indices_m1 = np.where(collision_unknown)[0][colided_mask_m1]

            v_in = np.linalg.norm(v_rel, axis=1)
            n_vel = np.sum(v_rel*normal, axis=-1)
            t_vel = np.sum(v_rel*tangent, axis=-1)
            angle_in = np.degrees(np.arctan(np.abs(t_vel/n_vel)))

            with torch.no_grad():
                col_out = mallet_res_model(torch.FloatTensor(np.stack((angle_in, v_in), axis=1))).numpy()
            angle_out = np.clip(np.random.normal(col_out[:,0], col_out[:,2]), -89, 89)
            v_out = np.maximum(np.random.normal(col_out[:,1], col_out[:,3]), 0.001)    

            v_rel_col = normal * (v_out * np.cos(angle_out * np.pi/180.0) * (-n_vel / abs(n_vel)))[:,None] + tangent * (v_out * np.sin(angle_out * np.pi/180.0) * (t_vel / abs(t_vel)))[:,None]

            v_p_col = v_rel_col + v_m[:,0,:][colided_mask_m1]
            new_pos[collision_indices_m1] = x_p[colided_mask_m1]
            new_vel[collision_indices_m1] = v_p_col

        if np.any(colided_mask_m2):
            v_rel = (v_p - v_m[:,1,:])[colided_mask_m2]
            normal = (x_p - x_m[:,1,:])[colided_mask_m2]
            normal = normal / np.tile(np.sqrt(np.sum(normal**2, axis=1))[:,np.newaxis], (1,2))

            colided_mask_m2_indicies = np.where(colided_mask_m2)[0]
            passed_mask = (np.sum(v_rel*normal, axis=-1) < 0)
            failed_indicies = colided_mask_m2_indicies[np.logical_not(passed_mask)]
            colided_mask_m2[failed_indicies] = False

        if np.any(colided_mask_m2):
            v_rel = v_rel[passed_mask]
            normal = normal[passed_mask]
            tangent = np.stack([-normal[:,1], normal[:,0]], axis=1)
            collision_indices_m2 = np.where(collision_unknown)[0][colided_mask_m2]

            v_in = np.linalg.norm(v_rel, axis=1)
            n_vel = np.sum(v_rel*normal, axis=-1)
            t_vel = np.sum(v_rel*tangent, axis=-1)
            angle_in = np.degrees(np.arctan(np.abs(t_vel/n_vel)))

            with torch.no_grad():
                col_out = mallet_res_model(torch.FloatTensor(np.stack((angle_in, v_in), axis=1))).numpy()
            angle_out = np.clip(np.random.normal(col_out[:,0], col_out[:,2]), -89, 89)
            v_out = np.maximum(np.random.normal(col_out[:,1], col_out[:,3]), 0.001)    

            v_rel_col = normal * (v_out * np.cos(angle_out * np.pi/180.0) * (-n_vel / abs(n_vel)))[:,None] + tangent * (v_out * np.sin(angle_out * np.pi/180.0) * (t_vel / abs(t_vel)))[:,None]

            v_p_col = v_rel_col + v_m[:,1,:][colided_mask_m2]
            new_pos[collision_indices_m2] = x_p[colided_mask_m2]
            new_vel[collision_indices_m2] = v_p_col

        collision_indicies = np.sort(np.concatenate((np.array(collision_indices_m1, dtype="int"), np.array(collision_indices_m2, dtype="int"))))

        if len(collision_indicies) > 0:
            collision_unknown[collision_indicies] = False
            collision_mask[collision_indicies] = True

            if not np.any(collision_unknown):
                break

            colided_mask = np.logical_or(colided_mask_m1, colided_mask_m2)

            x_p = x_p[~colided_mask]
            v_p = v_p[~colided_mask]

            x_m = x_m[~colided_mask]
            v_m = v_m[~colided_mask]
        else:
            colided_mask = np.array([False])
    
    return new_pos, new_vel, dt_sum, collision_mask 


def puck_mallet_collision_t(x_p, x_m, v_p, v_m, vpf, d_p, t, v_m_a2, am, a2m, C1m, CEm):
    #(game, player)

    t = np.tile(t[:, np.newaxis, np.newaxis], (1, 2, 2))
    v_max_mallet = np.where((t > am) & (t < a2m), v_m, np.maximum(np.abs(C1m * CEm[:,:,:,2]), np.abs(v_m)))
    v_max_mallet = np.where(t > a2m, v_m_a2, v_max_mallet)
    v_max_mallet = np.sqrt(np.sum(v_max_mallet**2, axis=-1))

    vp_norm = np.tile(np.sqrt(np.sum(v_p**2, axis=1))[:, np.newaxis], (1, 2))
    vpf = np.tile(vpf[:, np.newaxis], (1,2))

    w = np.tile(np.stack([-d_p[:,1], d_p[:,0]], axis=-1)[:,np.newaxis,:], (1,2,1))
    x = x_m - np.tile(x_p[:, np.newaxis, :], (1,2, 1))
    xd = x / np.tile(np.sqrt(np.sum(x**2, axis=-1))[:,:,np.newaxis], (1,1,2))
    v = v_m - np.tile(v_p[:,np.newaxis,:], (1,2,1))
    
    d_perp = np.sum(w*x, axis=-1)
    v_perp = np.abs(np.sum(xd*v, axis=-1)) + 1e-5 
    d_par = np.sum(np.tile(d_p[:,np.newaxis,:], (1,2,1))*x, axis=-1)

    #(game,player)
    t = np.full_like(x_p, -1, dtype="float32")
    vP = np.empty_like(t)

    R = puck_radius + mallet_radius

    mask_d_perp = np.abs(d_perp) > R
    mask_d_par = d_par > 0

    vP[~mask_d_perp & mask_d_par] = vp_norm[~mask_d_perp & mask_d_par]
    vP[~mask_d_perp & ~mask_d_par] = vpf[~mask_d_perp & ~mask_d_par]

    vP[mask_d_perp] = ((np.abs(d_perp) - R) / v_max_mallet)[mask_d_perp]

    vP[vP < vpf] = vpf[vP < vpf]
    vP[vP > vp_norm] = vp_norm[vP > vp_norm]

    qa = v_max_mallet * d_perp
    qb = v_max_mallet * d_par + R*vP
    qc = -vP*d_perp

    with np.errstate(divide='ignore', invalid='ignore'):
        theta_n = np.where(qb == qc, -2 * np.arctan(qb/np.abs(qa)),  2 * np.arctan((qa-np.sqrt(qa**2+qb**2-qc**2))/(qb-qc)))
        theta_m = theta_n - np.pi / 2

        sa = vP**2 + v_max_mallet**2 - 2*v_max_mallet*vP * np.cos(theta_m)
        sb = np.where(qb == qc, 2 * (vP - v_max_mallet * np.cos(theta_m)) * (-d_par) - 2 * (-v_max_mallet * np.sin(theta_m)) * np.abs(d_perp),\
                       2 * (vP - v_max_mallet * np.cos(theta_m)) * (-d_par) - 2 * (-v_max_mallet * np.sin(theta_m)) * d_perp)
        sc = d_par**2 + d_perp**2 - R**2

        t = (-sb - np.sqrt(sb**2 - 4*sa*sc)) / (2*sa)

    t[np.isnan(t) | np.isinf(t)] = np.inf
    t[t < 0] = np.inf
    small_dt = t < 1e-3
    t = np.where(small_dt, np.maximum(t, np.minimum(1e-3, (np.sqrt(np.sum(x**2,axis=-1)) - R)/(v_perp))), t)

    t = np.min(t, axis=-1)
    t[np.isinf(t)] = -1
    return t


def update_puck(t, mask, t_init, no_M = False, noM_pos = None, noM_vel = None, indicies=False, noM_xm = None, calculate_cross_left = False):
    if no_M:
        vel = noM_vel[mask]
        pos = noM_pos[mask]
    else:
        vel = puck_vel[mask]
        pos = puck_pos[mask]
    v_norm = np.sqrt(np.sum(vel**2, axis=1))

    if not no_M:
        global_mask = mask#np.where(mask)[0]
        full_time = np.zeros((game_number))
        full_time[global_mask] = t_init
        x_m = get_pos(full_time)[global_mask]
    
        #(game, player, x/y)
        x_diff = pos[:,np.newaxis,:] - x_m
        #(game,player)
        x_diff_norm = np.linalg.norm(x_diff, axis=2)
        intersection = x_diff_norm < ((puck_radius + mallet_radius) + 1e-10)

        if np.any(intersection):
            intersect_1 = intersection[:,0]
            intersect_2 = intersection[:,1]

            if np.any(intersect_1):
                pos[intersect_1] = x_m[intersect_1][:,0,:] + (puck_radius + mallet_radius + 1e-6) * x_diff[intersect_1][:,0,:] / x_diff_norm[intersect_1][:,0,np.newaxis]

            if np.any(intersect_2):
                #x_mi = x_m[intersect_2][:,1,:]
                #dir_i = x_diff[intersect_2][:,1,:] / x_diff_norm[intersect_2][:,1,np.newaxis]
                pos[intersect_2] = x_m[intersect_2][:,1,:] + (puck_radius + mallet_radius + 1e-6) * x_diff[intersect_2][:,1,:] / x_diff_norm[intersect_2][:,1,np.newaxis]

            x_diff2 = pos[:,np.newaxis,:] - x_m
            #(game,player)
            x_diff_norm2 = np.linalg.norm(x_diff2, axis=2)
            intersection2 = x_diff_norm2 < (puck_radius + mallet_radius)
            intersection2 = intersection2[:,0] | intersection2[:,1]

            if np.any(intersection2):
                pos[intersection2] = np.array([surface[0]/2, -2])
                vel[intersection2] = np.array([0,0])

    new_pos = np.empty_like(pos)
    new_vel = np.empty_like(vel)

    recurr_mask_m = np.full((len(v_norm)), False)
    recurr_time_m = np.zeros_like(t, dtype="float32")

    v_norm_mask = (v_norm != 0) & np.logical_not(((pos[:,0] < 0) | (pos[:,0] > surface[0])))

    if np.any(~v_norm_mask):
        recurr_mask_m[~v_norm_mask] = False
        new_pos[~v_norm_mask] = pos[~v_norm_mask]
        new_vel[~v_norm_mask] = 0

    v_norm[~v_norm_mask] = 1
    dir = vel / np.tile(v_norm[:, np.newaxis], (1, 2))
    dir[~v_norm_mask] = [0,1]

    Bm = drag[mask]
    fm = friction[mask]
    massm = mass[mask]
    bounds_puckm = bounds_puck[mask]
 
    C = getC(v_norm, Bm, fm)
    D = getD(Bm, massm, fm, C)
    C[~v_norm_mask] = 0
    D[~v_norm_mask] = 0
    dPdt = dP(t, Bm, fm, massm, C, D)
    vt = velocity(t, Bm, fm, massm, C)

    #check if it lies outside the playing area (i.e. somewhat in the goal) and will collide with a corner
    in_goal_mask = np.logical_or(pos[:,0] < bounds_puck[mask][:,0,0], pos[:,0] > bounds_puck[mask][:,0,1]) & v_norm_mask

    if np.any(in_goal_mask):
        for idx in np.where(in_goal_mask)[0]:
            point_A, bounces = check_in_goal_corner_bounce(vel[idx], pos[idx], v_norm[idx])
            if bounces:
                new_pos[idx], new_vel[idx], recurr_mask_m[idx], recurr_time_m[idx] = corner_collision(point_A, pos[idx], vel[idx],\
                                                                                                t[idx], C[idx], D[idx], v_norm[idx],\
                                                                                                dir[idx], dPdt[idx],Bm[idx],fm[idx],\
                                                                                                    massm[idx], vt[idx])
                v_norm_mask[idx] = False


    #Compute which wall it will hit if it keeps moving
    vel_x_0 = (vel[:,0] == 0)
    vel_y_0 = (vel[:,1] == 0)

    #(game, x/y)
    vel[:,0] = np.where(vel_x_0, 1, vel[:,0])
    vel[:,1] = np.where(vel_y_0, 1, vel[:,1])

    #distance, bounds_puck: (game, x/y, lower/upper)
    #pos, vel: (game, x/y)
    #v_norm: (game,)
    distances = (bounds_puckm - pos[:,:,None]) / vel[:,:,None] * v_norm[:,None,None]
    distances[:,0,:] = np.where(vel_x_0[:,None], -1, distances[:,0,:])
    distances[:,1,:] = np.where(vel_y_0[:,None], -1, distances[:,1,:])

    s_xy = np.maximum(distances[:,:,0], distances[:,:,1])
    s_xy = np.where(s_xy < 0, np.inf, s_xy)

    vel[:,1] = np.where(vel_y_0, 0, vel[:,1])

    #game,
    s = np.min(s_xy, axis=1)

    hit_wall_mask = (s < dPdt) & v_norm_mask
    
    #(game, lower/upper)
    side_bounds = np.full_like(pos, (0, surface[0]))
    y_hit = vel[:,1, None] * side_bounds / vel[:,0, None] + pos[:,1,None] - vel[:,1, None] * pos[:,0,None] / vel[:,0, None]
    y_hit = np.where(vel[:,0] > 0, y_hit[:,1], y_hit[:,0]) 

    y_goal_top = (y_hit + puck_radius * v_norm / np.abs(vel[:,0]) < surface[1]/2 + goal_width/2)
    y_goal_bottom = (y_hit - puck_radius * v_norm / np.abs(vel[:,0]) > surface[1]/2 - goal_width/2)

    enters_goal = y_goal_top & y_goal_bottom
    enters_goal = np.where(vel_x_0, False, enters_goal)

    vel[:,0] = np.where(vel_x_0, 0, vel[:,0])
    
    if np.any(enters_goal):
        hit_wall_mask = hit_wall_mask & ~enters_goal
    
    corner_hit_mask = np.full_like(hit_wall_mask, False)
    
    if np.any(hit_wall_mask):
        wall_dist = s[hit_wall_mask]
        wall = np.argmin(s_xy[hit_wall_mask], axis=1)

        y_wall = pos[hit_wall_mask][:,1] + wall_dist * np.where(vel_y_0[hit_wall_mask], 0, vel[hit_wall_mask][:,1]) / v_norm[hit_wall_mask]
        corner_hit = (surface[1]/2 - goal_width/2 < y_wall) & (y_wall < surface[1]/2 + goal_width/2)

        hit_wall_mask_indices = np.where(hit_wall_mask)[0]
        corner_hit_indicies = hit_wall_mask_indices[corner_hit]

        corner_hit_mask[corner_hit_indicies] = True
        hit_wall_mask = hit_wall_mask & ~corner_hit_mask

    #if not no_M:
    #    side_wall = np.full(new_vel.shape, False)

    if np.any(hit_wall_mask):
        wall_dist = s[hit_wall_mask]
        wall = np.argmin(s_xy[hit_wall_mask], axis=1)
        #if not no_M:
        #    side_wall[hit_wall_mask] = np.stack([(wall == 0) & (vel[hit_wall_mask,0] < 0), (wall == 0) & (vel[hit_wall_mask,0] > 0)], axis=1)
        #    if np.any(side_wall):
        #        pass

        Bmw = Bm[hit_wall_mask]
        fmw = fm[hit_wall_mask]
        massmw = massm[hit_wall_mask]
        Cw = C[hit_wall_mask]
        Dw = D[hit_wall_mask]
        tw = t[hit_wall_mask]
        dirw = dir[hit_wall_mask]
        root = chandrupatla(distance_to_wall, np.zeros_like(tw), tw, args=(wall_dist, \
                                Bmw, fmw, massmw, Cw, Dw))

        #B, f, mass, C, D
        new_pos[hit_wall_mask] = pos[hit_wall_mask] + dirw * np.tile(dP(root, Bmw, fmw,\
                                                    massmw, Cw, Dw)[:, np.newaxis], (1, 2))
        
        new_vel_bc = np.tile(velocity(root, Bmw, fmw, massmw, Cw)[:, np.newaxis], (1, 2)) * dirw
        
        v_in = np.linalg.norm(new_vel_bc, axis=1)

        wall_mask = wall == 0
        normal = np.where(wall_mask[:,np.newaxis], np.full((wall.shape[0],2), np.array([1,0])), np.full((wall.shape[0],2), np.array([0,1])))
        tangent = np.stack((normal[:,1], normal[:,0]), axis=1)

        n_vel = np.where(wall == 0, new_vel_bc[:,0], new_vel_bc[:,1])
        t_vel = np.where(wall == 0, new_vel_bc[:,1], new_vel_bc[:,0])
        angle_in = np.degrees(np.arctan(np.abs(t_vel/n_vel)))

        with torch.no_grad():
            col_out = res_model(torch.FloatTensor(np.stack((angle_in, v_in), axis=1))).numpy()
        angle_out = np.clip(np.random.normal(col_out[:,0], col_out[:,2]), -89, 89)
        v_out = np.maximum(np.random.normal(col_out[:,1], col_out[:,3]), 0.001)    

        new_vel_bc = normal * (v_out * np.cos(angle_out * np.pi / 180.0) * (-n_vel / abs(n_vel)))[:,None] + tangent * (v_out * np.sin(angle_out * np.pi / 180.0) * (t_vel / abs(t_vel)))[:,None]
        
        new_vel[hit_wall_mask] = new_vel_bc 

        recurr_mask_m[hit_wall_mask] = True
        recurr_time_m[hit_wall_mask] = tw-root


    if np.any(corner_hit_mask):
        point_A = np.empty((len(corner_hit_indicies), 2))
        point_A[:,0] = np.where(vel[corner_hit_mask][:,0] > 0, surface[0], 0)
        point_A[:,1] = np.where(y_hit[corner_hit_mask] > surface[1]/2, surface[1]/2 + goal_width/2, surface[1]/2-goal_width/2)

        i = 0
        for idx in corner_hit_indicies:
            new_pos[idx], new_vel[idx], recurr_mask_m[idx], recurr_time_m[idx] = corner_collision(point_A[i], pos[idx], vel[idx],\
                                                                                                t[idx], C[idx], D[idx], v_norm[idx],\
                                                                                                dir[idx], dPdt[idx],Bm[idx],fm[idx],massm[idx], vt[idx])
            i += 1

    no_col_mask = ~hit_wall_mask & ~corner_hit_mask & v_norm_mask
    if np.any(no_col_mask):
        Bmw = Bm[no_col_mask]
        fmw = fm[no_col_mask]
        massmw = massm[no_col_mask]
        Cw = C[no_col_mask]
        Dw = D[no_col_mask]
        tw = t[no_col_mask]
        new_pos[no_col_mask] = pos[no_col_mask] + dir[no_col_mask] * np.tile(dP(tw, Bmw, fmw, massmw, Cw, Dw)[:, np.newaxis], (1, 2))
        new_vel[no_col_mask] = np.tile(vt[no_col_mask][:, np.newaxis], (1, 2)) * dir[no_col_mask]
        recurr_mask_m[no_col_mask] = False

    if not no_M:
        dt_col = np.where(recurr_mask_m, t-recurr_time_m, t)
        vpf = velocity(dt_col, Bm, fm, massm, C)
        
        #res_malletm = res[mask]
        #print("A")
        #print(len(np.where(mask)[0]))
        pm_pos, pm_vel, pm_time, pm_mask = puck_mallet_collision(mask, pos, vel, dir, dt_col, t_init,\
                                                                C, D, Bm, fm, massm, vpf, x_m)


        recurr_mask_m = np.logical_or(recurr_mask_m, pm_mask)
        recurr_time_m[pm_mask] = (t - pm_time)[pm_mask]
        new_pos[pm_mask] = pm_pos[pm_mask]
        new_vel[pm_mask] = pm_vel[pm_mask]
        #side_wall[pm_mask] = False

        #return new_pos, new_vel, recurr_time_m, recurr_mask_m #, pm_mask, side_wall

        crossed_to_left = (new_pos[:,0] < surface[0]/2 - puck_radius) & (vel[:,0] < 0) & (pos[:,0] > surface[0]/2 - puck_radius)
        crossed_to_right = (new_pos[:,0] > surface[0]/2 + puck_radius) & (vel[:,0] > 0) & (pos[:,0] < surface[0]/2 + puck_radius)

        if calculate_cross_left:
            cross_left = np.full_like(recurr_time_m, -1)
        cross_right = np.full_like(recurr_time_m, -1)

        if calculate_cross_left and np.any(crossed_to_left):
            dist = np.sqrt(np.sum((pos[crossed_to_left] - new_pos[crossed_to_left])**2, axis=1)) * (pos[crossed_to_left,0] - (surface[0]/2-puck_radius)) / (pos[crossed_to_left,0] - new_pos[crossed_to_left,0])
            speed = np.sqrt(np.sum(vel[crossed_to_left]**2, axis=1))
            Bmw = Bm[crossed_to_left]
            fmw = fm[crossed_to_left]
            massmw = massm[crossed_to_left]
            Cw = C[crossed_to_left]
            Dw = D[crossed_to_left]
            #tw = t[crossed_to_left]
            dirw = dir[crossed_to_left]
            #resw = resm[crossed_to_left]
            #root = chandrupatla(distance_to_wall, dist/(speed+0.0001), tw, args=(dist, \
            #                        Bmw, fmw, massmw, Cw, Dw),eps_m = 1e-6, eps_a=2e-6)
            root = dist / speed
            #B, f, mass, C, D
            center_pos = pos[crossed_to_left] + dirw * np.tile(dP(root, Bmw, fmw,\
                                                        massmw, Cw, Dw)[:, np.newaxis], (1, 2))

            center_vel = np.tile(velocity(root, Bmw, fmw, massmw, Cw)[:, np.newaxis], (1, 2)) * dirw

            CEm = CE[mask][crossed_to_left]
            A2CEm = A2CE[mask][crossed_to_left]
            A3CEm = A3CE[mask][crossed_to_left]
            A4CEm = A4CE[mask][crossed_to_left]
            abm = ab[mask][crossed_to_left]
            C1m = C1[mask][crossed_to_left]
            C2m = C2[mask][crossed_to_left]
            C3m = C3[mask][crossed_to_left]
            C4m = C4[mask][crossed_to_left]
            am = a[mask][crossed_to_left]
            a2m = a2[mask][crossed_to_left]

            x_mc = get_pos_mask(t_init[crossed_to_left]+root, CEm, A2CEm, A3CEm, A4CEm, abm, am, a2m, C1m, C2m, C3m, C4m)

            if indicies:
                noM_pos, _, hitM = step_noM(1.0, center_pos, center_vel, mask[crossed_to_left], x_mc)
            else:
                noM_pos, _, hitM = step_noM(1.0, center_pos, center_vel, np.where(mask)[0][crossed_to_left].astype(np.int32), x_mc)
            entered_goal = noM_pos[:,0] < 0
            cross_left[crossed_to_left] = np.where(entered_goal, np.where(hitM, -0.25, np.sum(center_vel**2, axis=1)), -0.5)

        times_scored = np.zeros_like(crossed_to_right, dtype=np.uint16)

        if np.any(crossed_to_right):
            dist = np.sqrt(np.sum((pos[crossed_to_right] - new_pos[crossed_to_right])**2, axis=1)) * (pos[crossed_to_right,0] - (surface[0]/2+puck_radius)) / (pos[crossed_to_right,0] - new_pos[crossed_to_right,0])
            speed = np.sqrt(np.sum(vel[crossed_to_right]**2, axis=1))
            Bmw = Bm[crossed_to_right]
            fmw = fm[crossed_to_right]
            massmw = massm[crossed_to_right]
            Cw = C[crossed_to_right]
            Dw = D[crossed_to_right]
            #tw = t[crossed_to_right]
            dirw = dir[crossed_to_right]
            #resw = resm[crossed_to_right]
            #root = chandrupatla(distance_to_wall, dist/(speed+0.0001), tw, args=(dist, \
            #                        Bmw, fmw, massmw, Cw, Dw), eps_m = 1e-6, eps_a=2e-6)
            root = dist/speed
            #B, f, mass, C, D
            center_pos = pos[crossed_to_right] + dirw * np.tile(dP(root, Bmw, fmw,\
                                                        massmw, Cw, Dw)[:, np.newaxis], (1, 2))

            center_vel = np.tile(velocity(root, Bmw, fmw, massmw, Cw)[:, np.newaxis], (1, 2)) * dirw

            CEm = CE[mask][crossed_to_right]
            A2CEm = A2CE[mask][crossed_to_right]
            A3CEm = A3CE[mask][crossed_to_right]
            A4CEm = A4CE[mask][crossed_to_right]
            abm = ab[mask][crossed_to_right]
            C1m = C1[mask][crossed_to_right]
            C2m = C2[mask][crossed_to_right]
            C3m = C3[mask][crossed_to_right]
            C4m = C4[mask][crossed_to_right]
            am = a[mask][crossed_to_right]
            a2m = a2[mask][crossed_to_right]

            x_mc = get_pos_mask(t_init[crossed_to_right]+root, CEm, A2CEm, A3CEm, A4CEm, abm, am, a2m, C1m, C2m, C3m, C4m)

            for puck_rollout in range(20):
                if indicies:
                    noM_pos, _, hitM = step_noM(1.0, center_pos, center_vel, mask[crossed_to_right], x_mc)
                else:
                    noM_pos, _, hitM = step_noM(1.0, center_pos, center_vel, np.where(mask)[0][crossed_to_right].astype(np.int32), x_mc)
                entered_goal = noM_pos[:,0] > surface[0]
                times_scored[crossed_to_right] += np.logical_and(entered_goal, np.logical_not(hitM))

            cross_right[crossed_to_right] = np.sum(center_vel**2, axis=1)

        if calculate_cross_left:
            return new_pos, new_vel, recurr_time_m, recurr_mask_m, cross_left, cross_right
        else:
            return new_pos, new_vel, recurr_time_m, recurr_mask_m, times_scored/20.0, cross_right 

    if noM_xm is not None:
        mpd = np.sum((noM_xm[:,0,:]-pos) * dir, axis=1)
        mpd2 = np.sum((noM_xm[:,1,:]-pos) * dir, axis=1)
        tp = 1/np.maximum(np.linalg.norm(new_pos - pos, axis=1), 0.0001) * mpd
        tp2 = 1/np.maximum(np.linalg.norm(new_pos-pos, axis=1),0.0001) * mpd2

        Dsq = (dir[:,0] * mpd - (noM_xm[:,0,:]-pos)[:,0])**2 + (dir[:,1] * mpd - (noM_xm[:,0,:]-pos)[:,1])**2
        Dsq2 = (dir[:,0] * mpd2 - (noM_xm[:,1,:]-pos)[:,0])**2 + (dir[:,1] * mpd2 - (noM_xm[:,1,:]-pos)[:,1])**2

        Dsq = np.where((0<tp)&(tp<1), Dsq, np.minimum(np.sum((noM_xm[:,0,:]-pos)**2,axis=1), np.sum((noM_xm[:,0,:]-new_pos)**2,axis=1)))
        Dsq2 = np.where((0<tp2)&(tp2<1), Dsq2, np.minimum(np.sum((noM_xm[:,1,:]-pos)**2,axis=1), np.sum((noM_xm[:,1,:]-new_pos)**2,axis=1)))

        hit = (Dsq < (mallet_radius + puck_radius-0.002)**2)
        hit2 = (Dsq2 < (mallet_radius + puck_radius-0.002)**2)

        return new_pos, new_vel, recurr_time_m, np.logical_or(hit, hit2), recurr_mask_m

    return new_pos, new_vel, recurr_time_m, recurr_mask_m

#A e^(at) + B e^(bt) + Ct + D
def f(x, exponentials):
    #x: float
    #exponentials: (gamenumber, player, x/y, a/b)
    #returns (game, player, x/y)
    return CE[:,:,:,0] * exponentials[:,:,:,0] \
           + CE[:,:,:,1] * exponentials[:,:,:,1] + CE[:,:,:,2]*x+CE[:,:,:,3]


def g(tms):
    #tms: game, player, x/y, a/b
    tms_tile = np.tile(np.expand_dims(tms, axis=-1), (1,1,1,2))
    exponentials = np.exp(ab*tms_tile)
    return f(tms, exponentials)

def g_x(tms):
    tms_tile = np.tile(np.expand_dims(tms, axis=-1), (1,1,1,2))
    exponentials =  ab * np.exp(ab*tms_tile)
    return CE[:,:,:,0] * exponentials[:,:,:,0] +\
                    CE[:,:,:,1] * exponentials[:,:,:,1] + CE[:,:,:,2]

def g_x_a(tms):
    tms_tile = np.tile(np.expand_dims(tms, axis=-1), (1,1,1,2))
    exponentials =  2 * ab * np.exp(ab*tms_tile)
    return CE[:,:,:,0] * exponentials[:,:,:,0] +\
                    CE[:,:,:,1] * exponentials[:,:,:,1] + CE[:,:,:,2]

def g_xx(tms):
    tms_tile = np.tile(np.expand_dims(tms, axis=-1), (1,1,1,2))
    exponentials =  np.square(ab) * np.exp(ab*tms_tile)
    return CE[:,:,:,0] * exponentials[:,:,:,0] +\
                    CE[:,:,:,1] * exponentials[:,:,:,1]


def get_pos(t):
    #t: (gamenumber)
    #returns: (game,player,x/y)
    #A2CE: A e^(at) + Be^(bt) + C
    #A2 = (16.7785+0.218934*math.exp(-1577.51*t)-16.9975*math.exp(-20.319*t))
    #A3 = (345.371*math.exp(-20.319*t)-345.371*math.exp(-1577.51*t))
    #A4 = (544825*math.exp(-1577.51*t)-7017.58*math.exp(-20.319*t))
    
    #(gamenumber, player, x/y, a/b)
    exponentials = np.exp(ab*np.tile(t[:,np.newaxis,np.newaxis,np.newaxis], (1,2,2,2)))

    #shape (game, player, x/y)
    A2 = A2CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A2CE[:,:,:,1] * exponentials[:,:,:,1] +\
        A2CE[:,:,:,2]
    
    A3 = A3CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A3CE[:,:,:,1] * exponentials[:,:,:,1]
    
    A4 = A4CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A4CE[:,:,:,1] * exponentials[:,:,:,1]
    
    t = np.tile(t[:,np.newaxis,np.newaxis], (1,2,2))
    #(game, player, x/y)
    f_t = f(t, exponentials)

    #(game, player, x/y)
    tms = np.maximum(t-a,0)

    #ab: #(gamenumber, player, x/y, a/b)
    #tms: (game, player, x/y, a/b)
    g_a1 = np.where(tms > 0, g(tms), 0)

    tms = np.maximum(t-a2,0)
    g_a2 = np.where(tms > 0, g(tms), 0)

    return C1 * (f_t - 2*g_a1 + g_a2) + C2 * A2 + C3 * A3 + C4 * A4

def get_pos_mask(t, CEm, A2CEm, A3CEm, A4CEm, abm, am, a2m, C1m, C2m, C3m, C4m):
    #t: (gamenumber)
    #returns: (game,player,x/y)
    #A2CE: A e^(at) + Be^(bt) + C
    #A2 = (16.7785+0.218934*math.exp(-1577.51*t)-16.9975*math.exp(-20.319*t))
    #A3 = (345.371*math.exp(-20.319*t)-345.371*math.exp(-1577.51*t))
    #A4 = (544825*math.exp(-1577.51*t)-7017.58*math.exp(-20.319*t))
    
    #(gamenumber, player, x/y, a/b)
    exponentials = np.exp(abm*np.tile(t[:,np.newaxis,np.newaxis,np.newaxis], (1,2,2,2)))

    #shape (game, player, x/y)
    A2 = A2CEm[:,:,:,0] * exponentials[:,:,:,0] +\
        A2CEm[:,:,:,1] * exponentials[:,:,:,1] +\
        A2CEm[:,:,:,2]
    
    A3 = A3CEm[:,:,:,0] * exponentials[:,:,:,0] +\
        A3CEm[:,:,:,1] * exponentials[:,:,:,1]
    
    A4 = A4CEm[:,:,:,0] * exponentials[:,:,:,0] +\
        A4CEm[:,:,:,1] * exponentials[:,:,:,1]
    
    t = np.tile(t[:,np.newaxis,np.newaxis], (1,2,2))
    #(game, player, x/y)
    f_t = CEm[:,:,:,0] * exponentials[:,:,:,0] \
           + CEm[:,:,:,1] * exponentials[:,:,:,1] + CEm[:,:,:,2]*t+CEm[:,:,:,3]

    #(game, player, x/y)
    tms = np.maximum(t-am,0)
    tms_tile = np.tile(np.expand_dims(tms, axis=-1), (1,1,1,2))
    exponentials = np.exp(abm*tms_tile)
    g_a1 = np.where(tms > 0, CEm[:,:,:,0] * exponentials[:,:,:,0] \
           + CEm[:,:,:,1] * exponentials[:,:,:,1] + CEm[:,:,:,2]*tms+CEm[:,:,:,3], 0)

    tms = np.maximum(t-a2m,0)
    tms_tile = np.tile(np.expand_dims(tms, axis=-1), (1,1,1,2))
    exponentials = np.exp(abm*tms_tile)
    g_a2 = np.where(tms > 0, CEm[:,:,:,0] * exponentials[:,:,:,0] \
           + CEm[:,:,:,1] * exponentials[:,:,:,1] + CEm[:,:,:,2]*tms+CEm[:,:,:,3], 0)

    return C1m * (f_t - 2*g_a1 + g_a2) + C2m * A2 + C3m * A3 + C4m * A4


def get_xp(t):
    #t: (gamenumber)

    #(gamenumber, player, x/y, a/b)
    exponentials = ab * np.exp(ab*np.tile(t[:,np.newaxis,np.newaxis,np.newaxis], (1,2,2,2)))

    #shape (game, player, x/y)
    A2 = A2CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A2CE[:,:,:,1] * exponentials[:,:,:,1]
    
    A3 = A3CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A3CE[:,:,:,1] * exponentials[:,:,:,1]
    
    A4 = A4CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A4CE[:,:,:,1] * exponentials[:,:,:,1]
    
    f_t = CE[:,:,:,0] * exponentials[:,:,:,0] +\
          CE[:,:,:,1] * exponentials[:,:,:,1] + CE[:,:,:,2]
    
    t = np.tile(t[:,np.newaxis,np.newaxis], (1,2,2))
    tms = np.maximum(t-a, 0)
    g_a1 = np.where(tms > 0, g_x(tms), 0)
    tms = np.maximum(t-a2,0)
    g_a2 = np.where(tms > 0, g_x(tms), 0)

    return C1 * (f_t - 2*g_a1 + g_a2) + C2 * A2 + C3 * A3 + C4 * A4


def get_xp_mask(t, CEm, A2CEm, A3CEm, A4CEm, abm, am, a2m, C1m, C2m, C3m, C4m):
    #t: (gamenumber)

    #(gamenumber, player, x/y, a/b)
    exponentials = abm * np.exp(abm*np.tile(t[:,np.newaxis,np.newaxis,np.newaxis], (1,2,2,2)))

    #shape (game, player, x/y)
    A2 = A2CEm[:,:,:,0] * exponentials[:,:,:,0] +\
        A2CEm[:,:,:,1] * exponentials[:,:,:,1]
    
    A3 = A3CEm[:,:,:,0] * exponentials[:,:,:,0] +\
        A3CEm[:,:,:,1] * exponentials[:,:,:,1]
    
    A4 = A4CEm[:,:,:,0] * exponentials[:,:,:,0] +\
        A4CEm[:,:,:,1] * exponentials[:,:,:,1]
    
    f_t = CEm[:,:,:,0] * exponentials[:,:,:,0] +\
          CEm[:,:,:,1] * exponentials[:,:,:,1] + CEm[:,:,:,2]
    
    t = np.tile(t[:,np.newaxis,np.newaxis], (1,2,2))
    tms = np.maximum(t-am, 0)
    tms_tile = np.tile(np.expand_dims(tms, axis=-1), (1,1,1,2))
    exponentials =  abm * np.exp(abm*tms_tile)
    g_a1 = np.where(tms > 0, CEm[:,:,:,0] * exponentials[:,:,:,0] +\
                    CEm[:,:,:,1] * exponentials[:,:,:,1] + CEm[:,:,:,2], 0)
    
    tms = np.maximum(t-a2m,0)
    tms_tile = np.tile(np.expand_dims(tms, axis=-1), (1,1,1,2))
    exponentials =  abm * np.exp(abm*tms_tile)
    g_a2 = np.where(tms > 0, CEm[:,:,:,0] * exponentials[:,:,:,0] +\
                    CEm[:,:,:,1] * exponentials[:,:,:,1] + CEm[:,:,:,2], 0)

    return C1m * (f_t - 2*g_a1 + g_a2) + C2m * A2 + C3m * A3 + C4m * A4

def get_xp_mask_a2(CEm, A2CEm, A3CEm, A4CEm, abm, am, a2m, C1m, C2m, C3m, C4m):
    #t: (gamenumber)

    #(gamenumber, player, x/y, a/b)
    exponentials = abm * np.exp(abm*np.tile(a2m[:,:,:,np.newaxis], (1,1,1,2)))

    #shape (game, player, x/y)
    A2 = A2CEm[:,:,:,0] * exponentials[:,:,:,0] +\
        A2CEm[:,:,:,1] * exponentials[:,:,:,1]
    
    A3 = A3CEm[:,:,:,0] * exponentials[:,:,:,0] +\
        A3CEm[:,:,:,1] * exponentials[:,:,:,1]
    
    A4 = A4CEm[:,:,:,0] * exponentials[:,:,:,0] +\
        A4CEm[:,:,:,1] * exponentials[:,:,:,1]
    
    f_t = CEm[:,:,:,0] * exponentials[:,:,:,0] +\
          CEm[:,:,:,1] * exponentials[:,:,:,1] + CEm[:,:,:,2]

    tms = a2m-am
    tms_tile = np.tile(np.expand_dims(tms, axis=-1), (1,1,1,2))
    exponentials =  abm * np.exp(abm*tms_tile)
    g_a1 = CEm[:,:,:,0] * exponentials[:,:,:,0] +\
                    CEm[:,:,:,1] * exponentials[:,:,:,1] + CEm[:,:,:,2]

    return C1m * (f_t - 2*g_a1) + C2m * A2 + C3m * A3 + C4m * A4


def get_xpp(t):

    #(gamenumber, player, x/y, a/b)
    exponentials = np.square(ab) * np.exp(ab*t)

    #shape (game, player, x/y)
    A2 = A2CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A2CE[:,:,:,1] * exponentials[:,:,:,1]
    
    A3 = A3CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A3CE[:,:,:,1] * exponentials[:,:,:,1]
    
    A4 = A4CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A4CE[:,:,:,1] * exponentials[:,:,:,1]
    
    f_t = CE[:,:,:,0] * exponentials[:,:,:,0] +\
          CE[:,:,:,1] * exponentials[:,:,:,1]
    
    tms = np.maximum(t-a, 0)
    g_a1 = np.where(tms > 0, g_xx(tms), 0)
    tms = np.maximum(t-a2,0)
    g_a2 = np.where(tms > 0, g_xx(tms), 0)

    return C1 * (f_t - 2*g_a1 + g_a2) + C2 * A2 + C3 * A3 + C4 * A4

# c = C2[i,j,k]*A2CE[i,j,k,1]+C3[i,j,k]*A3CE[i,j,k,1]+C4[i,j,k]*A4CE[i,j,k,1]
def solve_Cp1(x,i,j,k,c):
    if x <= 0:
        return C2[i,j,k]/C7[i,j,k] - bounds_mallet[i,j,k,1]
    return C2[i,j,k]/C7[i,j,k] - (x/(ab[i,j,k,1]*C7[i,j,k])*\
                np.log((x*CE[i,j,k,3])/(-x*CE[i,j,k,1]+c))) - bounds_mallet[i,j,k,1]

def solve_Cp1_prime(x,i,j,k,c):
    if x <= 0:
        return -100
    e = (-x*CE[i,j,k,1]+c)
    d = np.log((x*CE[i,j,k,3])/e)
    return np.maximum((-1/(ab[i,j,k,1]*C7[i,j,k])) * (d + 1 + x*CE[i,j,k,1]/e), -100)

def solve_Cp1_prime2(x,i,j,k,c):
    if x <= 0:
        return 0
    e = (-x*CE[i,j,k,1]+c)
    return (-1/(ab[i,j,k,1]*C7[i,j,k])) * (1/x + 2*CE[i,j,k,1]/e + x*CE[i,j,k,1]**2/e**2)

# c = C2[i,j,k]*A2CE[i,j,k,1]+C3[i,j,k]*A3CE[i,j,k,1]+C4[i,j,k]*A4CE[i,j,k,1]
def solve_Cn1(x,i,j,k,c):
    if x >= 0:
        return C2[i,j,k]/C7[i,j,k] - bounds_mallet[i,j,k,0]
    return C2[i,j,k]/C7[i,j,k] - (x/(ab[i,j,k,1]*C7[i,j,k])*\
                np.log((x*CE[i,j,k,3])/(-x*CE[i,j,k,1]+c))) - bounds_mallet[i,j,k,0]

def solve_Cn1_prime(x,i,j,k,c):
    if x >= 0:
        return 100
    e = (-x*CE[i,j,k,1]+c)
    d = np.log(x*CE[i,j,k,3]/e)
    return np.minimum((-1/(ab[i,j,k,1]*C7[i,j,k])) *\
          (d + 1 + x*CE[i,j,k,1]/e), 100)

def solve_Cn1_prime2(x,i,j,k,c):
    if x >= 0:
        return 0
    e = (-x*CE[i,j,k,1]+c)
    return (-1/(ab[i,j,k,1]*C7[i,j,k])) * (1/x + 2*CE[i,j,k,1]/e + x*CE[i,j,k,1]**2/e**2)

def a_error(x):
    #x: (game, player, x/y)
    a2x = (2*x - x_f*C7/C1+C2/C1)

    #ab #(gamenumber, player, x/y, a/b)
    a2x_tile = np.tile(np.expand_dims(a2x, axis=-1), (1,1,1,2))
    exponentials = np.exp(ab * a2x_tile)

    A2 = A2CE[:,:,:,0] * exponentials[:,:,:,0] + A2CE[:,:,:,1] * exponentials[:,:,:,1] + A2CE[:,:,:,2]
    A3 = A3CE[:,:,:,0] * exponentials[:,:,:,0] + A3CE[:,:,:,1] * exponentials[:,:,:,1]
    A4 = A4CE[:,:,:,0] * exponentials[:,:,:,0] + A4CE[:,:,:,1] * exponentials[:,:,:,1]

    f_t = f(a2x, exponentials)

    tms = np.maximum(a2x-x,0)
    g_a1 = np.where(tms > 0, g(tms), 0)

    g_a2 = CE[:,:,:,0] + CE[:,:,:,1]+CE[:,:,:,3]

    return C1*(f_t-2*g_a1+g_a2) + C2*A2+C3*A3+C4*A4 - x_f

def a_error_prime(x):
    #x: (game, player, x/y)
    a2x = (2*x - x_f*C7/C1+C2/C1)

    #ab #(gamenumber, player, x/y, a/b)
    a2x_tile = np.tile(np.expand_dims(a2x, axis=-1), (1,1,1,2))
    exponentials = np.exp(ab * a2x_tile) * ab * 2

    A2 = A2CE[:,:,:,0] * exponentials[:,:,:,0] + A2CE[:,:,:,1] * exponentials[:,:,:,1]
    A3 = A3CE[:,:,:,0] * exponentials[:,:,:,0] + A3CE[:,:,:,1] * exponentials[:,:,:,1]
    A4 = A4CE[:,:,:,0] * exponentials[:,:,:,0] + A4CE[:,:,:,1] * exponentials[:,:,:,1]

    f_t = CE[:,:,:,0] * exponentials[:,:,:,0] +\
          CE[:,:,:,1] * exponentials[:,:,:,1] + CE[:,:,:,2]

    tms = np.maximum(a2x-x,0)
    g_a1 = np.where(tms > 0, g_x_a(tms), 0)

    #err = np.where(neg_mask, 1-a2x, np.where(less_than_ax_mask, 1+))
    return C1*(f_t-2*g_a1) + C2*A2+C3*A3+C4*A4

def solve_a():
    global x_over
    x_min = np.maximum(x_f * C7/C1 - C2/C1, 0)
    converged = False
    x0 = x_min + 0.01
    #for n in range(2,11):
    for _ in range(100):
        err = a_error(x0)    

        if np.all(np.abs(err) < 1e-10):
            converged = True
            break     
            
        err_prime = a_error_prime(x0) 
        
        # Update step: x_new = x - f(x) / f'(x)
        err_prime_mask = (err_prime == 0)
        err_prime = np.where(err_prime_mask, 1, err_prime)
        delta_x = err / err_prime
        x_old = np.copy(x0)
        x0[err_prime_mask] -= 0.0001
        x0[~err_prime_mask] -= delta_x[~err_prime_mask]
        #x0 = np.where(err_prime_mask, x0-0.001, x0 - delta_x)

        neg_mask = x0 < x_min
        x0[neg_mask] = x_min[neg_mask] / 2 + x_old[neg_mask] / 2
        x0[~neg_mask] = x0[~neg_mask]
        #x0 = np.where(neg_mask, x_min + (x_old-x_min) / 2, x0)
        #if converged:
        #    break
        #converged_mask = np.abs(err) < 1e-4
        #x0 = np.where(converged_mask, x0, n * np.maximum(0.5, x_min))
    if not converged:
        indices = np.argwhere(np.abs(err) > 1e-4)
        for i,j,k in indices:
            #print("failed to converge a")
            #print(err[i,j,k])
            #print(err_prime[i,j,k])
            #print(x0[i,j,k])
            #print("---")
            #print(x_p[i,j,k])
            #print(x_pp[i,j,k])
            #print(C2[i,j,k] * A2CE[i,j,k,2])
            #print(C1[i,j,k])
            #print("---")
            #log_arg = (-C1[i,j,k]*CE[i,j,k,3])/(C1[i,j,k]*CE[i,j,k,1]+C2[i,j,k]*A2CE[i,j,k,1]+C3[i,j,k]*A3CE[i,j,k,1]+C4[i,j,k]*A4CE[i,j,k,1])
            #x_over_2 = C2[i,j,k]/C7[i,j,k] - (-C1[i,j,k]/(ab[i,j,k,1]*C7[i,j,k]) * np.log(log_arg))
            #print(overshoot_mask[i,j,k])
            #print(x_0[i,j,k])
            #print(x_f[i,j,k])
            #print(x_over[i,j,k])
            #print(x_over_2)
            #print("---")
            pass
        

    return x0

def update_path(time, xf, Vo, reset_mask):
    global C1
    global C2
    global C3
    global C4
    global x_0
    global x_p
    global x_pp
    global x_f
    global x_over
    global a
    global a2
    global overshoot_mask
    #bounds_mallet: shape (game, 2, 2, 2), game, player, x/y, lower/upper
    #(game, player, x/y)

    #random_base = np.random.uniform(0,1, (game_number,2,2))
    #x_f = bounds_mallet[:,:,:,0] + random_base * (bounds_mallet[:,:,:,1] - bounds_mallet[:,:,:,0])
    x_f = xf

    #(game, player, x/y)
    x_0 = get_pos(np.full((game_number),time))
    x_0 = np.maximum(x_0, bounds_mallet[:,:,:,0]+1e-8)
    x_0 = np.minimum(x_0, bounds_mallet[:,:,:,1]-1e-8)
    x_p = get_xp(np.full((game_number),time))
    x_pp = get_xpp(time)

    x_0[reset_mask] = mallet_pos[reset_mask]
    x_p[reset_mask] = 0
    x_pp[reset_mask] = 0

    #(game, player, x/y)
    C2 = C5*x_pp + C6 * x_p + C7 * x_0
    C3 = C5 * x_p + C6 * x_0
    C4 = C5 * x_0

    #(game, player, x/y)
    #random_base = np.random.uniform(0.0, 1.0, (game_number,2,2))
    #Vo = random_base * Vmax

    #Vo = np.full((game_number, 2, 2), 14) #Has to lie within Vx + Vx <= 2*Vmax
    C1 = Vo * pullyR / 2

    #find x_f0
    #use this to set sign of C1
    #check overshoot
    x_f0 = C2 * A2CE[:,:,:,2]
    overshoot_mask = np.logical_or(np.abs(x_0 - x_f0) > np.abs(x_0 - x_f),\
                                   np.logical_xor(x_0 > x_f0, x_0 > x_f))

    C1 = np.where(overshoot_mask & (x_f0 < x_0), -C1, C1)
    x_over = np.empty_like(x_over)
    log_arg = (C1[overshoot_mask]*CE[:,:,:,3][overshoot_mask])/(-C1[overshoot_mask]*CE[:,:,:,1][overshoot_mask]+C2[overshoot_mask]*A2CE[:,:,:,1][overshoot_mask]+C3[overshoot_mask]*A3CE[:,:,:,1][overshoot_mask]+C4[overshoot_mask]*A4CE[:,:,:,1][overshoot_mask])
    x_over[overshoot_mask] = C2[overshoot_mask]/C7[overshoot_mask] - C1[overshoot_mask]/(ab[:,:,:,1][overshoot_mask]*C7[overshoot_mask]) * np.log(log_arg)
    x_over[~overshoot_mask] = x_0[~overshoot_mask]
    #log_arg = np.where(overshoot_mask, (C1*CE[:,:,:,3])/(-C1*CE[:,:,:,1]+C2*A2CE[:,:,:,1]+C3*A3CE[:,:,:,1]+C4*A4CE[:,:,:,1]), 1)
    #x_over = np.where(overshoot_mask,\
    #                    C2/C7 - (C1/(ab[:,:,:,1]*C7) * np.log(log_arg)),\
    #                    x_0)

    outofbounds_mask_p = (x_over > bounds_mallet[:,:,:,1])
    outofbounds_mask_n = (x_over < bounds_mallet[:,:,:,0])

    indices = np.argwhere(outofbounds_mask_p)

    for i,j,k in indices:
        c_fixed = C2[i,j,k]*A2CE[i,j,k,1]+C3[i,j,k]*A3CE[i,j,k,1]+C4[i,j,k]*A4CE[i,j,k,1]

        if solve_Cp1(abs(C1[i,j,k]), i,j,k,c_fixed) > 0:
            root = Vmax * pullyR

            if solve_Cp1(Vmax*pullyR,i,j,k,c_fixed) < 0:
                root, converged = brentq(solve_Cp1, abs(C1[i,j,k]), Vmax*pullyR, args=(i,j,k,c_fixed),\
                                        xtol=1e-8, maxiter=50, full_output=True, disp=False)
                    
                if not converged.converged:
                    #print("C1p Failed to Converge")
                    #print(solve_Cp1(0.05, i,j,k,c_fixed))
                    #print(solve_Cp1(root, i,j,k,c_fixed))
                    #print(root)
                    #print(x_0[i,j,k])
                    #print("---")
                    #print(x_p[i,j,k])
                    #print(x_pp[i,j,k])
                    #print(x_over[i,j,k])
                    #print(bounds_mallet[i,j,k,1])
                    pass

            root = np.minimum(abs(root), Vmax * pullyR-0.0001)

            C1[i,j,k] = root
            Vo_root = 2*root / pullyR
            if Vo_root + 2*abs(C1[i,j,1-k]) / pullyR > 2*Vmax:
                C1[i,j,1-k] = pullyR * (2*Vmax - Vo_root) / 2

                if overshoot_mask[i,j,1-k]:
                    if x_f0[i,j,1-k] < x_0[i,j,1-k]:
                        C1[i,j,1-k] = - C1[i,j,1-k]
                    x_over[i,j,1-k] = C2[i,j,1-k]/C7[i,j,1-k] - (C1[i,j,1-k]/(ab[i,j,1-k,1]*C7[i,j,1-k]) * np.log(\
                        (C1[i,j,1-k]*CE[i,j,1-k,3])/(-C1[i,j,1-k]*CE[i,j,1-k,1]+C2[i,j,1-k]*A2CE[i,j,1-k,1]+C3[i,j,1-k]*A3CE[i,j,1-k,1]+C4[i,j,1-k]*A4CE[i,j,1-k,1])))

            x_over[i,j,k] = bounds_mallet[i,j,k,1]

    indices = np.argwhere(outofbounds_mask_n)

    for i,j,k in indices:
        c_fixed = C2[i,j,k]*A2CE[i,j,k,1]+C3[i,j,k]*A3CE[i,j,k,1]+C4[i,j,k]*A4CE[i,j,k,1]

        if solve_Cn1(-abs(C1[i,j,k]), i,j,k,c_fixed) < 0:
            root = Vmax * pullyR

            if solve_Cn1(-Vmax*pullyR,i,j,k,c_fixed) > 0:
                root, converged = brentq(solve_Cn1, -abs(C1[i,j,k]), -Vmax*pullyR, args=(i,j,k,c_fixed),\
                                        xtol=1e-8, maxiter=50, full_output=True, disp=False)
                

                if not converged.converged:
                    #print("C1n Failed to Converge")
                    #print(solve_Cn1(-0.05, i,j,k,c_fixed))
                    #print(solve_Cn1(root, i,j,k,c_fixed))
                    #print(root)
                    #print(x_0[i,j,k])
                    #print("---")
                    #print(x_p[i,j,k])
                    #print(x_pp[i,j,k])
                    #print(x_over[i,j,k])
                    #print(bounds_mallet[i,j,k,0])
                    pass

            root = np.minimum(abs(root), Vmax * pullyR-0.0001)

            C1[i,j,k] = np.abs(root)
            Vo_root = 2*np.abs(root) / pullyR
            if Vo_root + 2*abs(C1[i,j,1-k]) / pullyR > 2*Vmax:
                C1[i,j,1-k] = pullyR * (2*Vmax - Vo_root) / 2
                if overshoot_mask[i,j,1-k]:
                    if x_f0[i,j,1-k] < x_0[i,j,1-k]:
                        C1[i,j,1-k] = -C1[i,j,1-k]
                    x_over[i,j,1-k] = C2[i,j,1-k]/C7[i,j,1-k] - (C1[i,j,1-k]/(ab[i,j,1-k,1]*C7[i,j,1-k]) * np.log(\
                        (C1[i,j,1-k]*CE[i,j,1-k,3])/(-C1[i,j,1-k]*CE[i,j,1-k,1]+C2[i,j,1-k]*A2CE[i,j,1-k,1]+C3[i,j,1-k]*A3CE[i,j,1-k,1]+C4[i,j,1-k]*A4CE[i,j,1-k,1])))
                    
            x_over[i,j,k] = bounds_mallet[i,j,k,0]

    C1 = np.abs(C1)
    
    #(game, player, x/y, min/max)
    #mallet_square = np.stack([
    #    np.minimum(x_0,np.minimum(x_f,x_over))-mallet_radius-puck_radius-0.002,
    #    np.maximum(x_0,np.maximum(x_f,x_over))+mallet_radius+puck_radius+0.002
    #], axis=-1)

    C1neg_mask = np.logical_or(np.logical_not(overshoot_mask) & (x_f < x_0),\
                                overshoot_mask & (x_over > x_f))
    C1 = np.where(C1neg_mask, -C1, C1)

    #(game_number, 2, 2)
    a = solve_a()
    a2 = 2*a - x_f*C7/C1+C2/C1

def step(t, calculate_cross_left = False):
    global time
    global puck_pos
    global mallet_pos
    global mallet_vel
    global puck_vel
    recurr_mask = np.full((game_number), True)
    recurr_mask[reset_mask] = False
    recurr_time = np.full((game_number), t, dtype="float32")
    #mallet_hit = np.full((game_number, 2), False)
    #puck_wall_collision = np.full((game_number, 2), False)
    cross_right = np.full((game_number), -1.0)
    counter = 0
    indexed = False
    #print("--")

    if calculate_cross_left:
        cross_left = np.full((game_number), -1.0)
        while np.any(recurr_mask):
            counter += 1
            if counter == 100:
                puck_pos[recurr_mask] = np.array([1,-1])
                puck_vel[recurr_mask] = np.array([0,0])
            #clock.tick(60)
            if recurr_mask.sum() / len(recurr_mask) < 0.1 and not indexed:
                indexed = True
                recurr_mask = np.array(np.where(recurr_mask)[0])
            #print(indexed)
            if not indexed:
                puck_pos[recurr_mask], puck_vel[recurr_mask], recurr_time[recurr_mask], recurr_mask2, cross_left_n, cross_right_n = update_puck(recurr_time[recurr_mask], recurr_mask, time + (t-recurr_time)[recurr_mask])
                cross_right[recurr_mask] = np.maximum(cross_right_n, cross_right[recurr_mask])
                cross_left[recurr_mask] = np.maximum(cross_left_n, cross_left[recurr_mask])
                #puck_wall_collision[recurr_mask] = puck_wall_collision[recurr_mask] | puck_wc
                #if np.any(pm_mask):
                #    mallet_hit[recurr_mask,0] = np.logical_or(pm_mask, mallet_hit[recurr_mask][:,0]) & (puck_pos[recurr_mask][:,0] < surface[0]/2)
                #    mallet_hit[recurr_mask,1] = np.logical_or(pm_mask, mallet_hit[recurr_mask][:,1]) & (puck_pos[recurr_mask][:,0] > surface[0]/2)
                recurr_mask[recurr_mask] = recurr_mask2
            else:
                puck_pos[recurr_mask], puck_vel[recurr_mask], recurr_time[recurr_mask], recurr_mask2, cross_left_n, cross_right_n = update_puck(recurr_time[recurr_mask], recurr_mask, time + (t-recurr_time)[recurr_mask], indicies=True)
                cross_right[recurr_mask] = np.maximum(cross_right_n, cross_right[recurr_mask])
                cross_left[recurr_mask] = np.maximum(cross_left_n, cross_left[recurr_mask])
                #puck_wall_collision[recurr_mask] = puck_wall_collision[recurr_mask] | puck_wc
                #if np.any(pm_mask):
                #    mallet_hit[recurr_mask[pm_mask],0] = (puck_pos[recurr_mask[pm_mask]][:,0] < surface[0]/2)
                #    mallet_hit[recurr_mask[pm_mask],1] = (puck_pos[recurr_mask[pm_mask]][:,0] > surface[0]/2)
                recurr_mask = recurr_mask[recurr_mask2]
            #print(clock.tick(60)/1000.0)
            #print(counter)
            
            #print("---")
        time += t
        mallet_pos[np.logical_not(reset_mask)] = get_pos(np.full((game_number), time))[np.logical_not(reset_mask)]
        mallet_vel[np.logical_not(reset_mask)] = get_xp(np.full((game_number),time))[np.logical_not(reset_mask)]

        return mallet_pos, mallet_vel, puck_pos, puck_vel, cross_left, cross_right #mallet_hit, puck_wall_collision

    score_avg = np.full((game_number), 0.0)
    while np.any(recurr_mask):
        counter += 1
        if counter == 100:
            puck_pos[recurr_mask] = np.array([1,-1])
            puck_vel[recurr_mask] = np.array([0,0])
        if recurr_mask.sum() / len(recurr_mask) < 0.1 and not indexed:
            indexed = True
            recurr_mask = np.array(np.where(recurr_mask)[0])
        if not indexed:
            puck_pos[recurr_mask], puck_vel[recurr_mask], recurr_time[recurr_mask], recurr_mask2, score_avg_n, cross_right_n = update_puck(recurr_time[recurr_mask], recurr_mask, time + (t-recurr_time)[recurr_mask])
            cross_right[recurr_mask] = np.where(score_avg_n >= score_avg[recurr_mask], cross_right_n, cross_right[recurr_mask])
            score_avg[recurr_mask] = np.maximum(score_avg_n, score_avg[recurr_mask])
            recurr_mask[recurr_mask] = recurr_mask2
        else:
            puck_pos[recurr_mask], puck_vel[recurr_mask], recurr_time[recurr_mask], recurr_mask2, score_avg_n, cross_right_n = update_puck(recurr_time[recurr_mask], recurr_mask, time + (t-recurr_time)[recurr_mask], indicies=True)
            cross_right[recurr_mask] = np.where(score_avg_n >= score_avg[recurr_mask], cross_right_n, cross_right[recurr_mask])
            score_avg[recurr_mask] = np.maximum(score_avg_n, score_avg[recurr_mask])
            recurr_mask = recurr_mask[recurr_mask2]
    time += t
    mallet_pos[np.logical_not(reset_mask)] = get_pos(np.full((game_number), time))[np.logical_not(reset_mask)]
    mallet_vel[np.logical_not(reset_mask)] = get_xp(np.full((game_number),time))[np.logical_not(reset_mask)]

    return mallet_pos, mallet_vel, puck_pos, puck_vel, score_avg, cross_right

def impulse(epsilon, delta):
    global puck_vel
    global puck_pos
    global sinks
    global sink_bounds
    angles = np.random.uniform(0, 2 * np.pi, size=(game_number,))
    directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    sinks = sinks + epsilon * directions

    for i in range(2):  # x and y
        low = sink_bounds[i, 0]
        high = sink_bounds[i, 1]

        over = sinks[:, i] > high
        under = sinks[:, i] < low

        sinks[over, i] = 2 * high - sinks[over, i]
        sinks[under, i] = 2 * low - sinks[under, i]

    directions = sinks - puck_pos
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_dirs = directions / norms
    low_vel = np.linalg.norm(puck_vel, axis=1) < 0.3
    puck_vel[low_vel] = puck_vel[low_vel] + delta * unit_dirs[low_vel]

def step_noM(t, puck_pos_noM = None, puck_vel_noM = None, recurr_mask=None, x_mc = None):
    
    if puck_pos_noM is None:
        puck_pos_noM2 = np.copy(puck_pos)
        puck_vel_noM2 = np.copy(puck_vel)
        recurr_mask = np.full((game_number), True)
        #recurr_mask[reset_mask] = False
        puck_pos_noM2[reset_mask] = np.array([0.5,0.5])
        puck_vel_noM2[reset_mask] = np.array([0.0, 0.0])
        recurr_time = np.full((game_number), t, dtype="float32")
    else:
        recurr_mask2 = np.zeros((game_number), dtype=np.bool_)
        recurr_mask2[recurr_mask] = True
        recurr_mask = recurr_mask2

        recurr_time = np.full((game_number), t, dtype="float32")
        puck_pos_noM2 = np.full((game_number,2), 0.5)
        puck_pos_noM2[recurr_mask] = puck_pos_noM
        puck_vel_noM2 = np.zeros((game_number,2))
        puck_vel_noM2[recurr_mask] = puck_vel_noM

    recurr_mask_copy = np.copy(recurr_mask)

    if x_mc is None:
        while np.any(recurr_mask):
            puck_pos_noM2[recurr_mask], puck_vel_noM2[recurr_mask], recurr_time[recurr_mask], recurr_mask[recurr_mask] = update_puck(recurr_time[recurr_mask], recurr_mask, time + (t-recurr_time)[recurr_mask], no_M = True, noM_pos=puck_pos_noM2, noM_vel = puck_vel_noM2)
    else:
        x_mcg = np.zeros((game_number, 2, 2), dtype="float32")
        x_mcg[recurr_mask] = x_mc

        hitM = np.zeros((game_number), dtype=np.bool_)
        while np.any(recurr_mask):
            puck_pos_noM2[recurr_mask], puck_vel_noM2[recurr_mask], recurr_time[recurr_mask], hitM[recurr_mask], recurr_mask[recurr_mask] = update_puck(recurr_time[recurr_mask], recurr_mask, time + (t-recurr_time)[recurr_mask], no_M = True, noM_pos=puck_pos_noM2, noM_vel = puck_vel_noM2, noM_xm = x_mcg[recurr_mask])
        return puck_pos_noM2[recurr_mask_copy], puck_vel_noM2[recurr_mask_copy], hitM[recurr_mask_copy]

    return puck_pos_noM2[recurr_mask_copy], puck_vel_noM2[recurr_mask_copy]

def take_action(x_f, Vo):
    global time
    global reset_mask
    global reset_sim_mask
    update_path(time, x_f, Vo, reset_mask)
    #reset_mask = np.full((game_number), False)
    reset_mask[reset_sim_mask] = False
    reset_sim_mask = np.full((game_number), False)
    time = 0

def get_xf0():
    return C2 * A2CE[:,:,:,2]

def get_a():
    return a

def check_state():
    global reset_mask
    #[n, index]
    #0: no errors
    #1: puck out of bounds
    #2: puck intercets mallet
    out_of_bounds = (np.logical_or((puck_pos[:,0] < bounds_puck[:,0,0]-1e-6), (puck_pos[:,0] > bounds_puck[:,0,1]+1e-6))) &\
                    (np.logical_or(puck_pos[:,1] < surface[1]/2-goal_width/2+puck_radius, puck_pos[:,1]>surface[1]/2+goal_width/2-puck_radius))

    index_arr = set()
    if np.any(out_of_bounds):
        for index in np.where(out_of_bounds)[0]:
            if puck_vel[index,0] != 0:
                y1 = puck_pos[index,1] - puck_vel[index,1] * puck_pos[index,0] / puck_vel[index, 0]
                y2 = puck_pos[index,1] - puck_vel[index,1]*puck_pos[index,0]/puck_vel[index,0] + puck_vel[index,1] * surface[0] / puck_vel[index,0]
                #print("Check")
                #print(y1)
                #print(y2)
                #print(puck_pos[index,:])
                #print(puck_vel[index,:])
                if (((puck_pos[index,0] < 0 or puck_pos[index,0] > surface[0]) and\
                    ((y1 < surface[1]/2-goal_width/2+puck_radius or y1 > surface[1]/2+goal_width/2-puck_radius) and (y2 < surface[1]/2-goal_width/2+puck_radius or  y2 > surface[1]/2+goal_width/2-puck_radius))) or\
                    ((puck_pos[index,0]**2 + (puck_pos[index,1] - surface[1]/2-goal_width/2)**2) < puck_radius**2) or\
                    ((puck_pos[index,0]**2 + (puck_pos[index,1] - surface[1]/2+goal_width/2)**2) < puck_radius**2) or\
                    (((puck_pos[index,0]-surface[0])**2 + (puck_pos[index,1] - surface[1]/2-goal_width/2)**2) < puck_radius**2) or\
                    (((puck_pos[index,0]-surface[0])**2 + (puck_pos[index,1] - surface[1]/2+goal_width/2)**2) < puck_radius**2)) and\
                    not reset_mask[index]:
                    #return [1, index]
                    index_arr.add(index)
                
    out_of_bounds = np.logical_or((puck_pos[:,1] < bounds_puck[:,1,0]-1e-6), (puck_pos[:,1] > bounds_puck[:,1,1]+1e-6)) & ~out_of_bounds & ~reset_mask
    if np.any(out_of_bounds):
        for index in np.where(out_of_bounds)[0]:
            #return [1,index]
            index_arr.add(index)
                
    in_mallet_mask = (np.tile(puck_pos[:,np.newaxis,0], (1,2)) - mallet_pos[:,:,0])**2 + (np.tile(puck_pos[:,np.newaxis,1], (1,2)) - mallet_pos[:,:,1])**2 <\
                        (puck_radius + mallet_radius - 1e-2)**2
    in_mallet_mask = (in_mallet_mask[:,0] | in_mallet_mask[:,1]) & ~reset_mask
    
    if np.any(in_mallet_mask):
        for index in np.where(in_mallet_mask)[0]:
            index_arr.add(index)

    index_np = np.array(list(index_arr))
    if len(index_np) > 0:
        reset_mask[index_np] = True
    
    return index_np
        
def check_goal():
    global reset_mask
    entered_left_goal_mask = np.logical_and(puck_pos[:,0] < 0, np.logical_not(reset_mask))
    entered_right_goal_mask = np.logical_and(puck_pos[:,0] > surface[0], np.logical_not(reset_mask))

    entered_goal = (entered_left_goal_mask | entered_right_goal_mask)

    if np.any(entered_goal):
        reset_mask = reset_mask | entered_goal
        #mallet_pos[entered_goal] = np.array([[0.25,0.5], [1.75,0.5]])
        #x_0[entered_goal] = np.array([[0.25,0.5], [1.75,0.5]])

        #mallet_pos[entered_goal][:,1,:] = np.array([1.75,0.5])
        #x_0[entered_goal][:,1,:] = np.array([1.75,0.5])

        #x_p[entered_goal] = 0.0
        #x_pp[entered_goal] = 0.0

        #puck_vel[entered_right_goal_mask] = np.array([0.0,0.0])
        #puck_pos[entered_right_goal_mask] = np.array([0.5,0.5])

        #puck_vel[entered_left_goal_mask] = np.array([0.0,0.0])
        #puck_pos[entered_left_goal_mask] = np.array([0.5,0.5])

    return entered_left_goal_mask, entered_right_goal_mask

"""
def display_state(index, puck_pos_noM=None):
    screen.fill(white)

    positionA1 = [mallet_pos[index,0,0] * plr, mallet_pos[index,0,1] * plr]
    positionA2 = [mallet_pos[index,1,0] * plr, mallet_pos[index,1,1] * plr]

    pygame.draw.rect(screen, red, pygame.Rect(screen_width/2 - 5, 0, 10, screen_height))
    pygame.draw.circle(screen, black, (int(round(positionA1[0])), int(round(positionA1[1]))), mallet_radius*plr)
    pygame.draw.circle(screen, black, (int(round(positionA2[0])), int(round(positionA2[1]))), mallet_radius*plr)
    pygame.draw.circle(screen, red, (int(x_f[index,0,0]*plr), int(x_f[index,0,1]*plr)), 5)
    pygame.draw.circle(screen, red, (int(x_f[index,1,0]*plr), int(x_f[index,1,1]*plr)), 5)
    pygame.draw.rect(screen, red, pygame.Rect(0, screen_height/2-goal_width*plr/2, 5, goal_width*plr))
    pygame.draw.rect(screen, red, pygame.Rect(screen_width-5, screen_height/2-goal_width*plr/2, 5, goal_width*plr))
    pygame.draw.circle(screen, blue, (int(round(puck_pos[index,0]*plr)), int(round(puck_pos[index,1]*plr))), puck_radius*plr)

    if not puck_pos_noM is None:
        pygame.draw.circle(screen, green, (int(round(puck_pos_noM[0]*plr)), int(round(puck_pos_noM[1]*plr))), puck_radius*plr)

    pygame.display.flip()
"""

def display_state(index, puck_pos_noM=None):
    global frame_count
    # Create a blank white image
    img = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

    # Convert to int drawing coordinates
    def to_int_coords(pos):
        return (int(round(pos[0])), int(round(pos[1])))

    # Draw center line
    cv2.rectangle(img, 
                  (int(screen_width/2 - 5), 0), 
                  (int(screen_width/2 + 5), screen_height), 
                  (0, 0, 255), -1)  # red in BGR

    # Draw mallets
    positionA1 = [mallet_pos[index,0,0] * plr, mallet_pos[index,0,1] * plr]
    positionA2 = [mallet_pos[index,1,0] * plr, mallet_pos[index,1,1] * plr]
    cv2.circle(img, to_int_coords(positionA1), int(mallet_radius*plr), (0, 0, 0), -1)
    cv2.circle(img, to_int_coords(positionA2), int(mallet_radius*plr), (0, 0, 0), -1)

    # Draw targets
    cv2.circle(img, to_int_coords([x_f[index,0,0]*plr, x_f[index,0,1]*plr]), 5, (0, 0, 255), -1)
    cv2.circle(img, to_int_coords([x_f[index,1,0]*plr, x_f[index,1,1]*plr]), 5, (0, 0, 255), -1)

    # Draw goals
    gw = int(goal_width * plr)
    cv2.rectangle(img, (0, screen_height//2 - gw//2), (5, screen_height//2 + gw//2), (0, 0, 255), -1)
    cv2.rectangle(img, (screen_width - 5, screen_height//2 - gw//2), (screen_width, screen_height//2 + gw//2), (0, 0, 255), -1)

    # Draw puck
    puck_pos_scaled = [puck_pos[index,0] * plr, puck_pos[index,1] * plr]
    cv2.circle(img, to_int_coords(puck_pos_scaled), int(puck_radius * plr), (255, 0, 0), -1)  # blue in BGR

    # Draw puck without mallet if provided
    if puck_pos_noM is not None:
        puck_pos_noM_scaled = [puck_pos_noM[0] * plr, puck_pos_noM[1] * plr]
        cv2.circle(img, to_int_coords(puck_pos_noM_scaled), int(puck_radius * plr), (0, 255, 0), -1)

    img = np.flip(img, axis=0)
    # Show the image
    cv2.imshow('Air Hockey State', img)
    #out.write(img)
    
    cv2.waitKey(1)  # Add small delay so window updates
    #frame_count += 1
    #if frame_count == 60*15:
    #    out.release()
    #    cv2.destroyAllWindows()
    #    print(1/0.0)

def get_mallet_bounds():
    return bounds_mallet

def reset_sim(index = None, ab_vars=None):#, left_scored= None):
    if index is None:
        index = np.array([i for i in range(game_number)])
    #if left_scored is None:
    #    left_scored = index
    global reset_sim_mask
    global x_p
    global x_pp
    global puck_vel
    global puck_pos
    global mallet_vel

    reset_sim_mask[index] = True
    reset_mask[index] = True

    mallet_pos[index,0,:] = np.array([0.25,0.5])
    mallet_vel[index,:,:] = np.array([0.0, 0.0])
    x_0[index,0,:] = np.array([0.25,0.5])

    mallet_pos[index,1,:] = bounds - np.array([0.25,0.5])
    x_0[index,1,:] = bounds - np.array([0.25,0.5])

    x_p[index,:] = 0.0
    x_pp[index,:] = 0.0

    puck_vel[index,:] = np.array([0.0,0.0])
    puck_pos[index,:] = puck_init[index,:]

    #(game, player, x/y)
    C5_idx = np.empty((len(index),2,2))
    C5_idx[:,:,0] = ab_vars[:,:,0] - ab_vars[:,:,3]
    C5_idx[:,:,1] = ab_vars[:,:,0] + ab_vars[:,:,3] #[a1-b1, a1+b1]

    C6_idx = np.empty((len(index),2,2))
    C6_idx[:,:,0] = ab_vars[:,:,1] - ab_vars[:,:,4]
    C6_idx[:,:,1] = ab_vars[:,:,1] + ab_vars[:,:,4] #[a2-b2, a2+b2]

    C7_idx = np.empty((len(index),2,2))
    C7_idx[:,:,0] = ab_vars[:,:,2]
    C7_idx[:,:,1] = ab_vars[:,:,2] #[a3-b3, a3+b3]

    A = np.sqrt(np.square(C6_idx) - 4 * C5_idx * C7_idx)
    B = 2 * np.square(C7_idx) * A

    C5[index] = C5_idx
    C6[index] = C6_idx
    C7[index] = C7_idx

    #(gamenumber, player, x/y, a/b)
    ab[index] = np.stack([(-C6_idx-A)/(2*C5_idx),(-C6_idx+A)/(2*C5_idx)], axis=-1)

    #(game_number, player, x/y, coefficients in A e^(at) + B e^(bt) + Ct + D)
    CE[index] = np.stack(
        [(-np.square(C6_idx) + A*C6_idx+2*C5_idx*C7_idx)/B,
        (np.square(C6_idx)+A*C6_idx-2*C5_idx*C7_idx)/B,
        1/C7_idx,
        -C6_idx/np.square(C7_idx)], axis=-1
    )

    B = 2*C7_idx*A

    #(game number, player, x/y, coefficients in A e^(at) + Be^(bt) + C)
    A2CE[index] = np.stack(
        [-(-C6_idx+A)/B,
        -(C6_idx+A)/B,
        1/C7_idx], axis=-1
    )

    #(game number, player, x/y, coefficients in A e^(at) + Be^(bt))
    A3CE[index] = np.stack(
        [-1/A,
        1/A], axis=-1
    )

    B = 2*C5_idx*A

    #(game number, player, x/y, coefficients in A e^(at) + Be^(bt))
    A4CE[index] = np.stack(
        [(C6_idx+A)/B,
        (-C6_idx+A)/B], axis=-1
    )
