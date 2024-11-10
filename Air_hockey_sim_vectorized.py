import pygame
import math
from scipy.optimize import fsolve, newton, brentq
from scipy.special import lambertw
import numpy as np
import random
from shapely.geometry import Polygon, LineString
from itertools import product
from chandrupatla import chandrupatla

# Initialize pygame
pygame.init()

print("STARTING")

screen_width = 1000
screen_height = 500
plr = 500

surface = np.array([screen_width / plr, screen_height / plr])

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
blue = (0,0,255)

# Set up the display
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Air Hockey")

game_number = 1000
# Circle parameters
#play 1 on the left, 2 on the right

mallet_radius = 0.05
puck_radius = 0.05

goal_width = 0.45

Vmax = 24
#(game, player, x/y)
Vmin = np.zeros((game_number,2,2), dtype="float32")
pullyR = 0.035306

puck_pos = np.empty((game_number, 2), dtype="float32")
puck_pos[:,0] = 1.975222 - 1.359155 * 0.3
puck_pos[:,1] = 0.7260047 - 0.98599756 * 0.3
puck_vel = np.full((game_number, 2), -1, dtype="float32")
puck_vel[:,0] = 1.359155
puck_vel[:,1] = 0.98599756

x_0 = np.empty((game_number,2,2), dtype="float32")
x_0[:,0,:] = [100/plr, 250/plr]
x_0[:,1,:] = [900/plr, 250/plr]
x_p = np.zeros((game_number,2,2), dtype="float32")
x_pp = np.zeros((game_number,2,2), dtype="float32")

margin = 0.001
#shape (2, 2, 2), player, x/y, lower/upper
bounds_mallet = np.array([[[mallet_radius+margin, screen_width/(2*plr) - mallet_radius - margin],\
                   [mallet_radius + margin, screen_height/plr - mallet_radius - margin]],\
                    [[screen_width/(2*plr)+mallet_radius+margin, screen_width/plr - mallet_radius - margin],\
                     [mallet_radius + margin, screen_height/plr - mallet_radius - margin]]])
# game, player,x/y,lower/upper
bounds_mallet = np.tile(np.expand_dims(bounds_mallet, axis=0), (game_number,1,1,1))

# x/y, lower/upper
bounds_puck = np.array([[puck_radius, screen_width/plr - puck_radius], [puck_radius, screen_height/plr - puck_radius]])
bounds_puck = np.tile(np.expand_dims(bounds_puck, axis=0), (game_number,1,1))
                  
drag = np.full((game_number), 0.01)
friction = np.full((game_number), 0.01)
res = np.full((game_number), 0.9)
mass = np.full((game_number), 0.1)
            
a1 = 3.579*10**(-6)
a2 = 0.00571
a3 = (0.0596+0.0467)/2
b1 = -1.7165*10**(-6)
b2 = -0.002739
b3 = 0

#(game, player, x/y)
C5 = np.full((game_number,2,2), [a1+b1,a1-b1])
C6 = np.full((game_number, 2, 2), [a2+b2, a2-b2])
C7 = np.full((game_number, 2, 2), [a3+b3,a3-b3])

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

#(game, player, x/y)
x_f = np.copy(x_0)
#(game, player, x/y)
a = np.full((game_number, 2, 2), 0.0)
#a = np.array([[[0.1,0.1], [0.1,0.1]], [[0.08,0.08], [0.08,0.08]]])
a2 = np.full((game_number, 2, 2), 0.0)

Ts = 0.2

x_over = np.zeros((game_number, 2, 2), dtype="float32")

#(game, player, x/y, min/max)
mallet_square = np.stack([
    np.copy(x_0)-(mallet_radius+puck_radius),
    np.copy(x_0)+mallet_radius+puck_radius
], axis=-1)

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


def corner_collision(A, pos, vel, t, C, D, v_norm, dir, dPdt,B,f,mass, vt, res):
    a = vel[0]**2 + vel[1]**2
    b = 2*pos[0]*vel[0] + 2*pos[1]*vel[1]-2*A[0]*vel[0]-2*A[1]*vel[1]
    c = pos[0]**2+pos[1]**2+A[0]**2+A[1]**2-puck_radius**2-2*A[0]*pos[0]-2*A[1]*pos[1]

    s = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

    new_pos = [0,0]
    new_pos[0] = pos[0] + s * vel[0]
    new_pos[1] = pos[1] + s * vel[1]
    wall_val = s * v_norm

    thit_min = -1
    if dPdt > wall_val:
        thit_min, converged = brentq(distance_to_wall, 0, t, args=(wall_val,B,f,mass,C,D),\
                                        xtol=1e-5, maxiter=30, full_output=True, disp=False)
        if not converged.converged:
            print("Failed to converge corner collision")
            print(pos)
            print(vel)
            print(wall_val)
            print(wall_val/v_norm)
            print(v_norm)
            print(C)
            print(D)
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

    vel_r = np.array([-res*np.dot(vel_r, n), res*np.dot(vel_r, tangent)])
    new_vel = n * vel_r[0] + tangent * vel_r[1]

    return new_pos, new_vel, True, t - thit_min

def check_in_goal(vel, pos, v_norm):
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


def update_puck(t, mask):
    vel = puck_vel[mask]
    pos = puck_pos[mask]
    v_norm = np.sqrt(np.sum(vel**2, axis=1))

    new_pos = np.empty_like(pos)
    new_vel = np.empty_like(vel)

    recurr_mask_m = np.full((len(v_norm)), False)
    recurr_time_m = np.empty_like(t, dtype="float32")

    v_norm_mask = (v_norm != 0)

    if np.any(~v_norm_mask):
        recurr_mask_m[~v_norm_mask] = False
        new_pos[~v_norm_mask] = pos[~v_norm_mask]
        new_vel[~v_norm_mask] = 0

    v_norm[~v_norm_mask] = 1
    dir = vel / np.tile(v_norm[:, np.newaxis], (1, 2))

    Bm = drag[mask]
    fm = friction[mask]
    massm = mass[mask]
    resm = res[mask]
    bounds_puckm = bounds_puck[mask]
 
    C = getC(v_norm, Bm, fm)
    D = getD(Bm, massm, fm, C)
    dPdt = dP(t, Bm, fm, massm, C, D)
    vt = velocity(t, Bm, fm, massm, C)

    #check if it lies outside the playing area (i.e. somewhat in the goal) and will collide with a corner
    in_goal_mask = np.logical_or(pos[:,0] < bounds_puck[mask][:,0,0], pos[:,0] > bounds_puck[mask][:,0,1]) & v_norm_mask

    if np.any(in_goal_mask):
        for idx in np.where(in_goal_mask)[0]:
            point_A, bounces = check_in_goal(vel[idx], pos[idx], v_norm[idx])
            if bounces:
                new_pos[idx], new_vel[idx], recurr_mask_m[idx], recurr_time_m[idx] = corner_collision(point_A, pos[idx], vel[idx],\
                                                                                                t[idx], C[idx], D[idx], v_norm[idx],\
                                                                                                dir[idx], dPdt[idx],Bm[idx],fm[idx],\
                                                                                                    massm[idx], vt[idx], resm[idx])
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

    if np.any(hit_wall_mask):
        wall_dist = s[hit_wall_mask]
        wall = np.argmin(s_xy[hit_wall_mask], axis=1)
        Bmw = Bm[hit_wall_mask]
        fmw = fm[hit_wall_mask]
        massmw = massm[hit_wall_mask]
        Cw = C[hit_wall_mask]
        Dw = D[hit_wall_mask]
        tw = t[hit_wall_mask]
        dirw = dir[hit_wall_mask]
        resw = resm[hit_wall_mask]
        root = chandrupatla(distance_to_wall, np.zeros_like(tw), tw, args=(wall_dist, \
                                Bmw, fmw, massmw, Cw, Dw))

        #B, f, mass, C, D
        new_pos[hit_wall_mask] = pos[hit_wall_mask] + dirw * np.tile(dP(root, Bmw, fmw,\
                                                    massmw, Cw, Dw)[:, np.newaxis], (1, 2))
        
        new_vel_bc = np.tile(velocity(root, Bmw, fmw, massmw, Cw)[:, np.newaxis], (1, 2)) * dirw
        new_vel_bc[:,0] = np.where(wall == 0, -resw * new_vel_bc[:,0], resw * new_vel_bc[:,0])
        new_vel_bc[:,1] = np.where(wall == 0, resw * new_vel_bc[:,1], - resw * new_vel_bc[:,1])
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
                                                                                                dir[idx], dPdt[idx],Bm[idx],fm[idx],massm[idx], vt[idx], resm[idx])
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

    #global recurr_time
    #global recurr_mask

    #for idx in range(len(new_pos)):
    #    if ((new_pos[idx,0]**2 + (new_pos[idx,1] - surface[1]/2-goal_width/2)**2) < puck_radius**2-1e-6) or\
    #            ((new_pos[idx,0]**2 + (new_pos[idx,1] - surface[1]/2+goal_width/2)**2) < puck_radius**2-1e-6) or\
    #            (((new_pos[idx,0]-surface[0])**2 + (new_pos[idx,1] - surface[1]/2-goal_width/2)**2) < puck_radius**2-1e-6) or\
    #            (((new_pos[idx,0]-surface[0])**2 + (new_pos[idx,1] - surface[1]/2+goal_width/2)**2) < puck_radius**2-1e-6):
    #        print(idx)
    #        update_puck(recurr_time, recurr_mask)
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
    #returns: (game,player,x/y)
    #A2CE: A e^(at) + Be^(bt) + C
    #A2 = (16.7785+0.218934*math.exp(-1577.51*t)-16.9975*math.exp(-20.319*t))
    #A3 = (345.371*math.exp(-20.319*t)-345.371*math.exp(-1577.51*t))
    #A4 = (544825*math.exp(-1577.51*t)-7017.58*math.exp(-20.319*t))
    
    #(gamenumber, player, x/y, a/b)
    exponentials = np.exp(ab*t)

    #shape (game, player, x/y)
    A2 = A2CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A2CE[:,:,:,1] * exponentials[:,:,:,1] +\
        A2CE[:,:,:,2]
    
    A3 = A3CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A3CE[:,:,:,1] * exponentials[:,:,:,1]
    
    A4 = A4CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A4CE[:,:,:,1] * exponentials[:,:,:,1]
    
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


def get_xp(t):
    #(gamenumber, player, x/y, a/b)
    exponentials = ab * np.exp(ab*t)

    #shape (game, player, x/y)
    A2 = A2CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A2CE[:,:,:,1] * exponentials[:,:,:,1]
    
    A3 = A3CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A3CE[:,:,:,1] * exponentials[:,:,:,1]
    
    A4 = A4CE[:,:,:,0] * exponentials[:,:,:,0] +\
        A4CE[:,:,:,1] * exponentials[:,:,:,1]
    
    f_t = CE[:,:,:,0] * exponentials[:,:,:,0] +\
          CE[:,:,:,1] * exponentials[:,:,:,1] + CE[:,:,:,2]
    
    tms = np.maximum(t-a, 0)
    g_a1 = np.where(tms > 0, g_x(tms), 0)
    tms = np.maximum(t-a2,0)
    g_a2 = np.where(tms > 0, g_x(tms), 0)

    return C1 * (f_t - 2*g_a1 + g_a2) + C2 * A2 + C3 * A3 + C4 * A4


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

        if np.all(np.abs(err) < 1e-4):
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
            print("failed to converge a")
            print(err[i,j,k])
            print(err_prime[i,j,k])
            print(x0[i,j,k])
            print("---")
            print(x_p[i,j,k])
            print(x_pp[i,j,k])
            print(C2[i,j,k] * A2CE[i,j,k,2])
            print(C1[i,j,k])
            print("---")
            log_arg = (-C1[i,j,k]*CE[i,j,k,3])/(C1[i,j,k]*CE[i,j,k,1]+C2[i,j,k]*A2CE[i,j,k,1]+C3[i,j,k]*A3CE[i,j,k,1]+C4[i,j,k]*A4CE[i,j,k,1])
            x_over_2 = C2[i,j,k]/C7[i,j,k] - (-C1[i,j,k]/(ab[i,j,k,1]*C7[i,j,k]) * np.log(log_arg))
            print(overshoot_mask[i,j,k])
            print(x_0[i,j,k])
            print(x_f[i,j,k])
            print(x_over[i,j,k])
            print(x_over_2)
            print("---")
        

    return x0

def update_path(time):
    global C1
    global C2
    global C3
    global C4
    global x_0
    global x_p
    global x_pp
    global x_f
    global x_over
    global mallet_square
    global a
    global a2
    global overshoot_mask
    #bounds_mallet: shape (game, 2, 2, 2), game, player, x/y, lower/upper
    #(game, player, x/y)

    random_base = np.random.uniform(0,1, (game_number,2,2))
    x_f = bounds_mallet[:,:,:,0] + random_base * (bounds_mallet[:,:,:,1] - bounds_mallet[:,:,:,0])

    #(game, player, x/y)
    x_0 = get_pos(time)
    x_0 = np.maximum(x_0, bounds_mallet[:,:,:,0]+1e-8)
    x_0 = np.minimum(x_0, bounds_mallet[:,:,:,1]-1e-8)
    x_p = get_xp(time)
    x_pp = get_xpp(time)

    #(game, player, x/y)
    C2 = C5*x_pp + C6 * x_p + C7 * x_0
    C3 = C5 * x_p + C6 * x_0
    C4 = C5 * x_0

    #(game, player, x/y)
    random_base = np.random.uniform(0,1, (game_number,2,2))
    Vo = random_base * Vmax
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
                                        xtol=1e-5, maxiter=30, full_output=True, disp=False)
                    
                if not converged.converged:
                    print("C1p Failed to Converge")
                    print(solve_Cp1(0.05, i,j,k,c_fixed))
                    print(solve_Cp1(root, i,j,k,c_fixed))
                    print(root)
                    print(x_0[i,j,k])
                    print("---")
                    print(x_p[i,j,k])
                    print(x_pp[i,j,k])
                    print(x_over[i,j,k])
                    print(bounds_mallet[i,j,k,1])

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
                                        xtol=1e-5, maxiter=30, full_output=True, disp=False)
                

                if not converged.converged:
                    print("C1n Failed to Converge")
                    print(solve_Cn1(-0.05, i,j,k,c_fixed))
                    print(solve_Cn1(root, i,j,k,c_fixed))
                    print(root)
                    print(x_0[i,j,k])
                    print("---")
                    print(x_p[i,j,k])
                    print(x_pp[i,j,k])
                    print(x_over[i,j,k])
                    print(bounds_mallet[i,j,k,0])

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
    mallet_square = np.stack([
        np.minimum(x_0,np.minimum(x_f,x_over))-mallet_radius-puck_radius-0.002,
        np.maximum(x_0,np.maximum(x_f,x_over))+mallet_radius+puck_radius+0.002
    ], axis=-1)

    C1neg_mask = np.logical_or(np.logical_not(overshoot_mask) & (x_f < x_0),\
                                overshoot_mask & (x_over > x_f))
    C1 = np.where(C1neg_mask, -C1, C1)

    #(game_number, 2, 2)
    a = solve_a()
    a2 = 2*a - x_f*C7/C1+C2/C1

time = 0
N = 1
count = 0
action = 0
clock = pygame.time.Clock()
timer = clock.tick(60) / 1000.0
index = 0

while True:
    #(game, player, x/y)
    recurr_mask = np.full((game_number), True)
    recurr_time = np.full((game_number), Ts/N, dtype="float32")
    while np.any(recurr_mask):
        puck_pos[recurr_mask], puck_vel[recurr_mask], recurr_time[recurr_mask], recurr_mask[recurr_mask], = update_puck(recurr_time[recurr_mask], recurr_mask)
        

    out_of_bounds = np.logical_or(np.logical_or((puck_pos[:,1] < bounds_puck[:,1,0]-1e-6), (puck_pos[:,1] > bounds_puck[:,1,1]+1e-6)),\
                        (np.logical_or((puck_pos[:,0] < bounds_puck[:,0,0]-1e-6), (puck_pos[:,0] > bounds_puck[:,0,1]+1e-6))) &\
                        (np.logical_or(puck_pos[:,1] < surface[1]/2-goal_width/2, puck_pos[:,1]>surface[1]/2+goal_width/2)))

    if np.any(out_of_bounds):
        index = np.where(out_of_bounds)[0][0]
        y1 = puck_pos[index,1] - puck_vel[index,1]*puck_pos[index,0]/puck_vel[index,0]
        y2 = puck_pos[index,1] - puck_vel[index,1]*puck_pos[index,0]/puck_vel[index,0] + puck_vel[index,1] * surface[0] / puck_vel[index,0]
        if ((y1 < surface[1]/2-goal_width/2 or y1 > surface[1]/2+goal_width/2) and (y2 < surface[1]/2-goal_width/2 or  y2 > surface[1]/2+goal_width/2)) or\
            ((puck_pos[index,0]**2 + (puck_pos[index,1] - surface[1]/2-goal_width/2)**2) < puck_radius**2) or\
            ((puck_pos[index,0]**2 + (puck_pos[index,1] - surface[1]/2+goal_width/2)**2) < puck_radius**2) or\
            (((puck_pos[index,0]-surface[0])**2 + (puck_pos[index,1] - surface[1]/2-goal_width/2)**2) < puck_radius**2) or\
            (((puck_pos[index,0]-surface[0])**2 + (puck_pos[index,1] - surface[1]/2+goal_width/2)**2) < puck_radius**2):
            print("OUT OF BOUNDS")
            screen.fill(white)
            pygame.draw.rect(screen, red, pygame.Rect(0, screen_height/2-goal_width*plr/2, 5, goal_width*plr))
            pygame.draw.rect(screen, red, pygame.Rect(screen_width-5, screen_height/2-goal_width*plr/2, 5, goal_width*plr))
            pygame.draw.circle(screen, blue, (int(round(puck_pos[index,0]*plr)), int(round(puck_pos[index,1]*plr))), puck_radius*plr)
            pygame.display.flip()
            print(3/0)
        puck_pos = np.maximum(puck_pos, bounds_puck[:,:,0]+1e-8)
        puck_pos = np.minimum(puck_pos, bounds_puck[:,:,1]-1e-8)

    entered_goal_mask = np.logical_or(puck_pos[:,0] < 0, puck_pos[:,0] > surface[0])
    not_moving_mask = (puck_vel[:,0] == 0) & (puck_vel[:,1] == 0) 

    restart_mask = np.logical_or(entered_goal_mask, not_moving_mask)

    if np.any(restart_mask):
        puck_vel[restart_mask] = np.random.uniform(-3, 3, size=(restart_mask.sum(), 2))
        puck_pos[restart_mask] = np.random.uniform(0.4, 0.6, size=(restart_mask.sum(), 2))

    pos = get_pos(time)
    if time >= Ts:
        update_path(time)
        time = 0

    #position1 = [pos[0][0][0] * plr, pos[0][0][1] * plr]
    #position2 = [pos[0][1][0] * plr, pos[0][1][1] * plr]

    if N > 1:
        screen.fill(white)

        #pygame.draw.rect(screen, red, pygame.Rect(screen_width/2 - 5, 0, 10, screen_height))
        #pygame.draw.rect(screen, red, pygame.Rect((int(mallet_square[0,0,0,0]*plr), int(mallet_square[0,0,1,0]*plr)), (int(mallet_square[0,0,0,1]*plr - mallet_square[0,0,0,0]*plr), int(mallet_square[0,0,1,1]*plr - mallet_square[0,0,1,0]*plr))), width=5)
        #pygame.draw.rect(screen, red, pygame.Rect((int(mallet_square[0,1,0,0]*plr), int(mallet_square[0,1,1,0]*plr)), (int(mallet_square[0,1,0,1]*plr - mallet_square[0,1,0,0]*plr), int(mallet_square[0,1,1,1]*plr - mallet_square[0,1,1,0]*plr))), width=5)
        #pygame.draw.circle(screen, black, (int(round(position1[0])), int(round(position1[1]))), mallet_radius*plr)
        #pygame.draw.circle(screen, black, (int(round(position2[0])), int(round(position2[1]))), mallet_radius*plr)
        #pygame.draw.circle(screen, red, (int(x_f[0,0,0]*plr), int(x_f[0,0,1]*plr)), 5)
        #pygame.draw.circle(screen, red, (int(x_f[0,1,0]*plr), int(x_f[0,1,1]*plr)), 5)
        pygame.draw.rect(screen, red, pygame.Rect(0, screen_height/2-goal_width*plr/2, 5, goal_width*plr))
        pygame.draw.rect(screen, red, pygame.Rect(screen_width-5, screen_height/2-goal_width*plr/2, 5, goal_width*plr))
        pygame.draw.circle(screen, blue, (int(round(puck_pos[index,0]*plr)), int(round(puck_pos[index,1]*plr))), puck_radius*plr)
        pygame.display.flip()
    else:
        action += 1
        if action % int(0.1*3600/Ts) == 0:
            print(clock.tick(60) / 1000.0)
            print(action * Ts / 3600)

    time += Ts/N

#100 games: 15 sec / 0.1 hour, 2400 speedup
#500 games: 42.07 sec / 0.1 hour, 4279 speedup
#1000 games: 84 sec / 0.1 hour, 4286 speedup 

#with puck (only wall collision)
#100 games: 28.731 sec / 0.1 hour, 1253 speedup,  
#1000 games: 113 / 0.1 hour, 3185 speedup

#with puck and mallet (no puck mallet collision)
#1000 games: 98 sec / 0.1 hour, 3673 speedup
