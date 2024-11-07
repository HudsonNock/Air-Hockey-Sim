import pygame
import math
from scipy.optimize import fsolve, newton, brentq
from scipy.special import lambertw
import numpy as np
import random
from shapely.geometry import Polygon, LineString
from itertools import product

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

# Set up the display
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Air Hockey")

game_number = 1000
# Circle parameters
#game number, player number, mallet_pos (m)
#play 1 on the left, 2 on the right
mallet_pos = np.empty((game_number,2,2), dtype="float32")
mallet_pos[:,0,:] = [100/plr, 250/plr]
mallet_pos[:,1,:] = [900/plr, 250/plr]

mallet_radius = 0.05
puck_radius = 0.05

Vmax = 24
#(game, player, x/y)
Vmin = np.zeros((game_number,2,2), dtype="float32")
pullyR = 0.035306


x_0 = np.copy(mallet_pos)
x_p = np.zeros((game_number,2,2), dtype="float32")
x_pp = np.zeros((game_number,2,2), dtype="float32")

margin = 0.001
#shape (2, 2, 2), player, x/y, lower/upper
bounds = np.array([[[mallet_radius+margin, screen_width/(2*plr) - mallet_radius - margin],\
                   [mallet_radius + margin, screen_height/plr - mallet_radius - margin]],\
                    [[screen_width/(2*plr)+mallet_radius+margin, screen_width/plr - mallet_radius - margin],\
                     [mallet_radius + margin, screen_height/plr - mallet_radius - margin]]])
# game, player,x/y,lower/upper
bounds = np.tile(np.expand_dims(bounds, axis=0), (game_number,1,1,1))
                  
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
x_f = np.copy(mallet_pos)
#(game, player, x/y)
a = np.full((game_number, 2, 2), 0.0)
#a = np.array([[[0.1,0.1], [0.1,0.1]], [[0.08,0.08], [0.08,0.08]]])
a2 = np.full((game_number, 2, 2), 0.0)

Ts = 0.2

x_over = np.zeros((game_number, 2, 2), dtype="float32")

#(game, player, x/y, min/max)
mallet_square = np.stack([
    np.copy(mallet_pos)-(mallet_radius+puck_radius),
    np.copy(mallet_pos)+mallet_radius+puck_radius
], axis=-1)

overshoot_mask = np.full((game_number, 2, 2), True)

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
        return C2[i,j,k]/C7[i,j,k] - bounds[i,j,k,1]
    return C2[i,j,k]/C7[i,j,k] - (x/(ab[i,j,k,1]*C7[i,j,k])*\
                np.log((x*CE[i,j,k,3])/(-x*CE[i,j,k,1]+c))) - bounds[i,j,k,1]

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
        return C2[i,j,k]/C7[i,j,k] - bounds[i,j,k,0]
    return C2[i,j,k]/C7[i,j,k] - (x/(ab[i,j,k,1]*C7[i,j,k])*\
                np.log((x*CE[i,j,k,3])/(-x*CE[i,j,k,1]+c))) - bounds[i,j,k,0]

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
    for n in range(2,11):
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
            x0 = np.where(err_prime_mask, x0-0.001, x0 - delta_x)

            neg_mask = x0 < x_min
            x0 = np.where(neg_mask, x_min + (x_old-x_min) / 2, x0)
        if converged:
            break
        converged_mask = np.abs(err) < 1e-4
        x0 = np.where(converged_mask, x0, n * np.maximum(0.5, x_min))
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
    #bounds: shape (game, 2, 2, 2), game, player, x/y, lower/upper
    #(game, player, x/y)

    random_base = np.random.uniform(0,1, (game_number,2,2))
    x_f = bounds[:,:,:,0] + random_base * (bounds[:,:,:,1] - bounds[:,:,:,0])

    #(game, player, x/y)
    x_0 = get_pos(time)
    x_0 = np.maximum(x_0, bounds[:,:,:,0]+1e-8)
    x_0 = np.minimum(x_0, bounds[:,:,:,1]-1e-8)
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
    log_arg = np.where(overshoot_mask, (C1*CE[:,:,:,3])/(-C1*CE[:,:,:,1]+C2*A2CE[:,:,:,1]+C3*A3CE[:,:,:,1]+C4*A4CE[:,:,:,1]), 1)
    x_over = np.where(overshoot_mask,\
                        C2/C7 - (C1/(ab[:,:,:,1]*C7) * np.log(log_arg)),\
                        x_0)

    outofbounds_mask_p = (x_over > bounds[:,:,:,1])
    outofbounds_mask_n = (x_over < bounds[:,:,:,0])

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
                    print(bounds[i,j,k,1])

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

            x_over[i,j,k] = bounds[i,j,k,1]

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
                    print(bounds[i,j,k,0])

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
                    
            x_over[i,j,k] = bounds[i,j,k,0]

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
while True:
    #(game, player, x/y)
    pos = get_pos(time)
    if time >= Ts:
        update_path(time)
        time = 0

    """
    position1 = [pos[0][0][0] * plr, pos[0][0][1] * plr]
    position2 = [pos[0][1][0] * plr, pos[0][1][1] * plr]

    screen.fill(white)
    pygame.draw.rect(screen, red, pygame.Rect(screen_width/2 - 5, 0, 10, screen_height))
    pygame.draw.circle(screen, black, (int(round(position1[0])), int(round(position1[1]))), mallet_radius*plr)
    pygame.draw.circle(screen, black, (int(round(position2[0])), int(round(position2[1]))), mallet_radius*plr)
    pygame.draw.circle(screen, red, (int(x_f[0,0,0]*plr), int(x_f[0,0,1]*plr)), 5)
    pygame.display.flip()
    """

    action += 1
    if action % int(0.1*3600/Ts) == 0:
        print(clock.tick(60) / 1000.0)
        print(action * Ts / 3600)

    time += Ts/N

#100 games: 15 sec / 0.1 hour, 2400 speedup
#500 games: 42.07 sec / 0.1 hour, 4279 speedup
#1000 games: 84 sec / 0.1 hour, 4286 speedup 