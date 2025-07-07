import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from torchrl.modules import ProbabilisticActor, TanhNormal
import tensordict
from tensordict import TensorDict, TensorDictBase
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
import time
import math
from scipy.optimize import fsolve
import serial
import json
import threading
import queue
import cv2
import struct

obs_dim = 17
action_dim = 4
height = 1.9885
width = 0.9905
goal_width = 0.254

mallet_r = 0.0508
puck_r = 0.5 * 0.0618
op_mallet_r = 0.0508

Vmax = 24
table_bounds = np.array([height, width])
margin = 0.065
margin_bottom = 0.1

margin_bounds = 0.0
mallet_bounds = np.array([[margin_bounds + mallet_r, height/2  + mallet_r/2], [margin_bounds+mallet_r, width-margin_bounds-mallet_r]])

drag = 0.0014273
friction = 0.00123794
res = 0.7
res_mallet = 0.9
mass = 0.00776196

past_mallet_pos = [[0,0], [0,0]]
dts = [0,0]

current_time = time.time()
xf_past = [0.2, 0.5]
puck_vel_past = [0,0]

class ScaledNormalParamExtractor(NormalParamExtractor):
    def __init__(self, scale_factor=0.5):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        loc, scale = super().forward(x)
        #scale *= self.scale_factor
        scale = 2*self.scale_factor/ (1+torch.exp(-scale)) - self.scale_factor + 0.001
        return loc, scale

policy_net = nn.Sequential(
    nn.Linear(obs_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32, action_dim * 2),
    ScaledNormalParamExtractor(),
)

policy_module = TensorDictModule(
    policy_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": torch.tensor([mallet_r, mallet_r, 0.3, 0.3]),
        "high": torch.tensor([1-mallet_r, 1 - mallet_r, 10, 10]),
    },
    #default_interaction_type=tensordict.nn.InteractionType.RANDOM,
    return_log_prob=False,
)

policy_module.load_state_dict(torch.load("policy_weights8.pth")) #8

pullyR = 0.035306

C1 = [Vmax * pullyR / 2, Vmax * pullyR / 2] #[Vmax * pullyR / 2, Vmax * pullyR / 2]

a1 = 3.579*10**(-6)
a2 = 0.00571
a3 = (0.0596+0.0467)/2
b1 = -1.7165*10**(-6)
b2 = -0.002739
b3 = 0

C5 = [a1-b1, a1+b1]
C6 = [a2-b2, a2+b2]
C7 = [a3-b3, a3+b3]


#CE: A e^(at) + B e^(bt) + Ct + D
#A2CE: A e^(at) + Be^(bt) + C
#A3CE: A e^(at) + Be^(bt)
#A4CE: A e^(at) + Be^(bt)
CE = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
ab = [[0,0], [0,0]]
A2CE = [[0, 0, 0], [0,0,0]]
A3CE = [[0,0],[0,0]]
A4CE = [[0,0],[0,0]]
A = 0
B = 0
for i in range(2):
    A = math.sqrt(C6[i]**2-4*C5[i]*C7[i])
    B = 2*C7[i]**2*A

    ab[i][0] = (-C6[i]-A)/(2*C5[i])
    ab[i][1] = (-C6[i]+A)/(2*C5[i])

    CE[i][0] = (-C6[i]**2 + A*C6[i]+2*C5[i]*C7[i])/B
    CE[i][1] = (C6[i]**2+A*C6[i]-2*C5[i]*C7[i])/B
    CE[i][2] = 1/C7[i]
    CE[i][3] = -C6[i]/(C7[i]**2)

    B = 2*C7[i]*A
    A2CE[i][0] = -(-C6[i]+A)/B
    A2CE[i][1] = -(C6[i]+A)/B
    A2CE[i][2] = 1/C7[i]

    A3CE[i][0] = -1/A
    A3CE[i][1] = 1/A

    B = 2*C5[i]*A
    A4CE[i][0] = (C6[i]+A)/B
    A4CE[i][1] = (-C6[i]+A)/B

C2 = [0,0]
C3 = [0,0]
C4 = [0,0]

def dP(t, C, D):
    #return C*np.exp(-(B/mass)*t) + D - (friction/B)*t
    return (mass/drag) * np.log(np.cos((np.sqrt(drag*friction)*(C*mass+t))/mass)) + D

def velocity(t, C):
    #return (-drag/mass)*C*np.exp(-(drag/mass)*t) - (friction/drag)
    return -np.sqrt(friction/drag) * np.tan((np.sqrt(drag*friction)*(C*mass+t))/mass)

def get_tPZero(C):
    #return -(mass/drag)*np.log(-friction*mass/(C*drag**2))
    return -C*mass

def getC(v_norm):
    #return -(v_norm + friction/drag)*(mass/drag)
    return - np.arctan(v_norm * np.sqrt(drag/friction)) / np.sqrt(drag*friction)

def getD(C):
    #return -C
    return - (mass/drag) * np.log(np.cos(np.sqrt(drag*friction)*C))

def pos_wall(t, wall_val, C, D):
    return  dP(t,C,D) - wall_val

def corner_collision(A, pos, vel, t, final_pos, C, D, tPZero, v_norm, dPZero, dir, dPt, vt):
    a = vel[0]**2 + vel[1]**2
    b = 2*pos[0]*vel[0] + 2*pos[1]*vel[1]-2*A[0]*vel[0]-2*A[1]*vel[1]
    c = pos[0]**2+pos[1]**2+A[0]**2+A[1]**2-puck_r**2-2*A[0]*pos[0]-2*A[1]*pos[1]

    if b**2 - 4*a*c < 0:
        #print("s < 0 error")
        #s = max(-b/(2*a), 0.001)
        #print(1/0)
        return None, None
    else:
        s = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

    new_pos = [0,0]
    new_pos[0] = pos[0] + s * vel[0]
    new_pos[1] = pos[1] + s * vel[1]
    wall_val = s * v_norm
    thit_min = -1
    if dPZero > wall_val:
        val, info, ier, msg = fsolve(pos_wall, x0=wall_val/v_norm, xtol=1e-6, full_output=True, args=(wall_val,C,D))
        thit_min = val[0]
        if ier != 1 and abs(pos_wall(val)) > 1e-4:
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
    if thit_min == -1 or t < thit_min:
        thit = thit_min
        if thit == -1 or t < thit_min:
            thit = t
        for j in range(2):
            if t > tPZero:
                new_pos[j] = final_pos[j]
            else:
                new_pos[j] = pos[j] + dPt * dir[j]
                new_vel[j] = vt * dir[j]
        return new_pos, new_vel

    vt_hit = velocity(thit_min,C)
    for j in range(2):
        new_vel[j] = vt_hit * dir[j]

    n = np.array([new_pos[0] - A[0], new_pos[1] - A[1]])
    n = n / np.sqrt(np.dot(n, n))
    tangent = np.array([n[1], -n[0]])
    vel_r = np.array(new_vel)

    vel_r = np.array([-res*np.dot(vel_r, n), res*np.dot(vel_r, tangent)])
    new_vel = n * vel_r[0] + tangent * vel_r[1]

    return predict_puck(t-thit_min, new_pos, new_vel)

def check_in_goal(vel, pos, w):
    A = 0
    bounces = False
    s2 = 0
    if pos[0] < puck_r:
        if vel[0] > 0:
            if w[0] > 0:
                w *= -1
            if pos[0] + w[0] < 0:
                s2 = (-pos[0] - w[0]) / vel[0]
                if table_bounds[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w[1] or vel[1]*s2+pos[1]+w[1] >  table_bounds[1]/2 + goal_width/2:
                    bounces = True
        elif vel[0] < 0:
            if w[0] < 0:
                w *= -1
            if pos[0] + w[0] > 0:
                s2 = (-pos[0]-w[0]) / vel[0]
                if table_bounds[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w[1] or vel[1]*s2+pos[1]+w[1] >  table_bounds[1]/2 + goal_width/2:
                    bounces = True
        if bounces:
            A = 0
    elif pos[0] > table_bounds[0]-puck_r:
        if vel[0] > 0:
            if w[0] > 0:
                w *= -1
            if pos[0] + w[0] < table_bounds[0]:
                s2 = (-pos[0] - w[0] + table_bounds[0]) / vel[0]
                if table_bounds[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w[1] or vel[1]*s2+pos[1]+w[1] >  table_bounds[1]/2 + goal_width/2:
                    bounces = True
        elif vel[0] < 0:
            if w[0] < 0:
                w *= -1
            if pos[0] + w[0] > table_bounds[0]:
                s2 = (-pos[0]-w[0]+table_bounds[0]) / vel[0]
                if table_bounds[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w[1] or vel[1]*s2+pos[1]+w[1] >  table_bounds[1]/2 + goal_width/2:
                    bounces = True
        if bounces:
            A = table_bounds[0] 

    return A, bounces, s2

def predict_puck(t, pos, vel):
    new_pos = np.zeros((2,), dtype=np.float32)
    new_vel = np.zeros((2,), dtype=np.float32)
    thit = -1

    w = np.array([vel[1], -vel[0]])
    v_norm = np.linalg.norm(w)
    if v_norm != 0:
        w = puck_r * w / v_norm
    else:
        return pos, vel
    
    dir = vel/v_norm

    C = getC(v_norm)
    D = getD(C)
    tPZero = get_tPZero(C)
    dPZero = dP(tPZero,C,D)
    final_pos = pos + dir * dPZero

    #check if it lies outside the playing area (i.e. somewhat in the goal) and will collide with a corner
    bounces = False
    A_0, bounces, s2 = check_in_goal(vel, pos, w)
    dPt = dP(t,C,D)
    vt = velocity(t,C)
    
    if bounces:

        if table_bounds[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w[1]:
            A_1 = table_bounds[1]/2 - goal_width/2
        elif vel[1]*s2+pos[1]+w[1] >  table_bounds[1]/2 + goal_width/2:
            A_1 = table_bounds[1]/2 + goal_width/2
        A = np.array([A_0, A_1])

        return corner_collision(A, pos, vel, t, final_pos, C, D, tPZero, v_norm, dPZero, dir, dPt, vt)
        
    
    #Check if its going into the goal without hitting a corner
    s1 = 0
    s2 = 0
    y1 = 0
    y2 = 0
    if abs(vel[0]) != 0:
        if vel[0] < 0:
            s1 = (-pos[0] - w[0]) / vel[0]
            s2 = (-pos[0] + w[0]) / vel[0]
        elif vel[0] > 0:
            s1 = (-pos[0] - w[0] + table_bounds[0]) / vel[0]
            s2 = (-pos[0] + w[0] + table_bounds[0]) / vel[0]
        y1 = pos[1] + w[1] + s1*vel[1]
        y2 = pos[1] - w[1] + s2*vel[1]

        #both side rays enter goal
        if table_bounds[1]/2 - goal_width/2 < y1 < table_bounds[1]/2 + goal_width/2 and table_bounds[1]/2 - goal_width/2 < y2 < table_bounds[1]/2 + goal_width/2:
            for j in range(2):
                if t > tPZero:
                    new_pos[j] = final_pos[j]
                else:
                    new_pos[j] = pos[j] + dPt * dir[j]
                    new_vel[j] = vt * dir[j]

            return new_pos, new_vel
        

    #Compute which wall it will hit if it keeps moving
    wallx = [puck_r, table_bounds[0] - puck_r]
    wally = [puck_r,table_bounds[1] - puck_r]
    wall_idx = 0

    s = 0
    if vel[0] != 0 and vel[1] != 0:
        idx = 0
        if vel[0] > 0:
            idx = 1
        s1 = (wallx[idx] - pos[0]) / vel[0]
        idx = 0
        if vel[1] > 0:
            idx = 1
        s2 = (wally[idx] - pos[1]) / vel[1]
        if s2 < s1:
            wall_idx = 1
            s = s2 * v_norm
        else:
            s = s1 * v_norm
    elif vel[0] == 0:
        idx = 0
        if vel[1] > 0:
            idx = 1
        wall_idx = 1
        s2 = (wally[idx] - pos[1]) / vel[1]
        s = s2 * v_norm
    else:
        idx = 0
        if vel[0] > 0:
            idx = 1
        wall_idx = 0
        s1 = (wallx[idx] - pos[0]) / vel[0]
        s = s1 * v_norm

    #Check if it will hit the wall

    if s < dPZero:
        wall_val = s
        val, info, ier, msg = fsolve(pos_wall, x0=wall_val/v_norm, xtol=1e-4, full_output=True, args=(wall_val,C,D))
        thit = val[0]
        if ier != 1 and abs(pos_wall(val, wall_val, C, D)) > 1e-4:
            return None, None
            print("Failed to converge V>=0")
            print(pos)
            print(v_norm)
            print(wall_val)
            print(wall_val/v_norm)
            print(val)
            print(pos_wall(val))

    thit_min = -1
    if thit > 0:
        thit_min = thit


    #If it doesn't hit the wall in time, use the normal equations
    if thit_min == -1 or t < thit_min:
        for j in range(2):
            if t > tPZero:
                new_pos[j] = final_pos[j]
            else:
                new_pos[j] = pos[j] + dPt * dir[j]
                new_vel[j] = vt * dir[j]

        return new_pos, new_vel
    

    #Check if it will hit the corner or not
    if wall_idx == 0 and  table_bounds[1]/2 - goal_width/2 < vel[1] * s1 + pos[1] < table_bounds[1]/2 + goal_width/2:
        A = [0,table_bounds[1]/2 - goal_width/2]
        if vel[0] > 0:
            A[0] = table_bounds[0]
        if (y1+y2)/2 > table_bounds[1]/2:
            A[1] = table_bounds[1]/2 + goal_width/2

        return corner_collision(A, pos, vel, t, final_pos, C, D, tPZero, v_norm, dPZero, dir, dPt, vt)
    
    #Go to where it would hit the wall, set position and vel opposit of what they would be
    dPt_hit = dP(thit_min,C,D)
    vt_hit = velocity(thit_min,C)
    for j in range(2):
        if j == wall_idx:
            new_pos[j] = pos[j] + dPt_hit * dir[j]
            new_vel[j] = -res * vt_hit * dir[j]
        else:
            new_pos[j] = pos[j] + dPt_hit * dir[j]
            new_vel[j] = res * vt_hit * dir[j]

    return predict_puck(t-thit_min, new_pos, new_vel)

#A e^(at) + B e^(bt) + Ct + D
def f(x, i, eat, ebt):
    return CE[i][0] * eat \
           + CE[i][1] * ebt + CE[i][2]*x+CE[i][3]
    #return 16.7785*x-0.0001387*math.exp(-1577.51*x) \
    #       +0.836531*math.exp(-20.319*x)-0.836393


def g(tms, i):
    if tms < 0:
        return 0
    eat = math.exp(ab[i][0]*tms)
    ebt = math.exp(ab[i][1]*tms)
    return f(tms, i, eat, ebt)


def ax_error(ax, x_f):

    a2x = (2*ax[0] - x_f[0]*C7[0]/C1[0]+C2[0]/C1[0])
    if a2x < 0:
        return 1-a2x
    if a2x < ax[0]:
        return 1 + a2x - ax[0]
    t = a2x

    eat = math.exp(ab[0][0]*t)
    ebt = math.exp(ab[0][1]*t)
    A2 = A2CE[0][0] * eat + A2CE[0][1]*ebt + A2CE[0][2]
    A3 = A3CE[0][0] * eat + A3CE[0][1]*ebt
    A4 = A4CE[0][0] * eat + A4CE[0][1]*ebt
    f_t = f(t, 0, eat, ebt)
    g_a1 = g(t-ax[0], 0)
    g_a2 = CE[0][0] + CE[0][1]+CE[0][3]
    return C1[0]*(f_t-2*g_a1+g_a2) + C2[0]*A2+C3[0]*A3+C4[0]*A4 - x_f[0]

def ay_error(ay, x_f):
    a2y = (2*ay[0] - x_f[1]*C7[1]/C1[1]+C2[1]/C1[1])
    if a2y < 0:
        return 1-a2y
    if a2y < ay[0]:
        return 1 + a2y - ay[0]
    t = a2y

    eat = math.exp(ab[1][0]*t)
    ebt = math.exp(ab[1][1]*t)
    A2 = A2CE[1][0] * eat + A2CE[1][1]*ebt + A2CE[1][2]
    A3 = A3CE[1][0] * eat + A3CE[1][1]*ebt
    A4 = A4CE[1][0] * eat + A4CE[1][1]*ebt
    f_t = f(t, 1, eat, ebt)
    g_a1 = g(t-ay[0], 1)
    g_a2 = CE[1][0] + CE[1][1]+CE[1][3]
    return C1[1]*(f_t-2*g_a1+g_a2) + C2[1]*A2+C3[1]*A3+C4[1]*A4 - x_f[1]

def solve_vt1(x_f):
    #ax = [x_f[0] / (C1[0]*16.7785) - C2[0]/C1[0]]
    #ay = [x_f[1] / (C1[1]*16.7785) - C2[1]/C1[1]]
    x0=x_f[0] * C7[0]/C1[0] - C2[0]/C1[0]
    ax = 0.5
    if x0 == 0:
        ax = [0]
    else:
        ax, info, ier, msg = fsolve(ax_error, x0, xtol=1e-4, full_output=True, args=(x_f))
        if ier != 1 and abs(ax_error(ax, x_f)) > 1e-4:
            for n in range(2,11):
                ax, info, ier, msg = fsolve(ax_error, n*x0, xtol=1e-4, full_output=True, args=(x_f))
                if ier == 1:
                    break
            if ier != 1 and abs(ax_error(ax, x_f)) > 1e-4:
                print("failed to converge ax")
    
    x0=x_f[1] * C7[1]/C1[1] - C2[1]/C1[1]
    ay = 0.5
    if x0 == 0:
        ay = [0]
    else:
        ay, info, ier, msg = fsolve(ay_error, x0, xtol=1e-4, full_output = True, args=(x_f)) #2*(abs(x_f[1]-x_0[1]))/math.sqrt(abs(C1[1]*2/R)), xtol=1e-4)
        if ier != 1 and abs(ay_error(ay, x_f)) > 1e-4:
            for n in range(2,11):
                ay, info, ier, msg = fsolve(ay_error, n*x0, xtol=1e-4, full_output=True, args=(x_f))
                if ier == 1:
                    break
            if ier != 1 and abs(ay_error(ay, x_f)) > 1e-4:
                print("failed to converge ay")
    ax = np.float32(ax)
    ay = np.float32(ay)
    return [ax, ay]

def solve_C1p(x, k):
    if x <= 0:
        return 1 - x
    return C2[k]/C7[k] - (x[0]/(ab[k][1]*C7[k])*\
                math.log((x[0]*CE[k][3])/(-x[0]*CE[k][1]+C2[k]*A2CE[k][1]+C3[k]*A3CE[k][1]+C4[k]*A4CE[k][1]))) - mallet_bounds[k][1]

def solve_C1n(x, k):
    if x[0] >= 0:
        return 1 + x[0]
    return C2[k]/C7[k] - (x[0]/(ab[k][1]*C7[k])*\
                math.log((x[0]*CE[k][3])/(-x[0]*CE[k][1]+C2[k]*A2CE[k][1]+C3[k]*A3CE[k][1]+C4[k]*A4CE[k][1]))) - mallet_bounds[k][0]

def update_path(x_0, x_p, x_pp, x_f, Vo, idle=False):
    global C1
    global C2
    global C3
    global C4

    C2 = [C5[0]*x_pp[0]+C6[0]*x_p[0]+C7[0]*x_0[0], C5[1]*x_pp[1]+C6[1]*x_p[1]+C7[1]*x_0[1]]
    C3 = [C5[0]*x_p[0]+C6[0]*x_0[0],C5[1]*x_p[1]+C6[1]*x_0[1]]
    C4 = [C5[0]*x_0[0],C5[1]*x_0[1]]

    Vmin = [0,0]

    for j in range(2):
        if x_p[j] > 0 and C2[j]/C7[j] > mallet_bounds[j][1]:
            val, info, ier, msg = fsolve(solve_C1p, x0=0.05, xtol=1e-4, full_output=True, args=(j))
            if ier != 1:
                for n in range(1,11):
                    val, info, ier, msg = fsolve(solve_C1p, x0=0.05/(n*10), xtol=1e-4, full_output=True, args=(j))
                    if ier == 1:
                        break

                if ier != 1:
                    print("C1p failed corvergence")

            C1[j] = val[0]
            Vmin[j] = C1[j] * 2/pullyR

            #solve C1 > 0 so that x_over[j] = bounds[1]
        elif x_p[j] < 0 and C2[j]/C7[j] < mallet_bounds[j][0]:
            val, info, ier, msg = fsolve(solve_C1n, x0=-0.05, xtol=1e-4, full_output=True, args=(j))
            if ier != 1:
                for n in range(1,11):
                    val, info, ier, msg = fsolve(solve_C1n, x0=-0.05/(n*10), xtol=1e-4, full_output=True, args=(j))
                    if ier == 1:
                        break

                if ier != 1:
                    print("C1n failed corvergence")

            C1[j] = val[0]

            Vmin[j] = abs(C1[j]) * 2/pullyR
        # set magntiude of C1
        
    Vf = [0,0]

    Vmin[0] = max(Vmin[0], 0.01)
    Vmin[1] = max(Vmin[1], 0.01)
    if Vmin[0] + Vmin[1] > 2*Vmax:
        sum = Vmin[0] + Vmin[1]
        Vmin[0] *= 2*Vmax/sum
        Vmin[1] *= 2*Vmax/sum

    err_str = "None"
    if Vo[0] > Vmin[0] and Vo[1] > Vmin[1] and Vo[1] + Vo[0] < 2*Vmax:
        Vf[0] = Vo[0]
        Vf[1] = Vo[1]
        err_str = "A"
    elif Vo[1] < Vo[0] + 2*Vmin[1]-2*Vmax:
        Vf[1] = Vmin[1]
        Vf[0] = 2*Vmax-Vmin[1]
        err_str = "B"
    elif Vo[1] > Vo[0] - 2*Vmin[0]+2*Vmax:
        Vf[0]=Vmin[0]
        Vf[1]=2*Vmax-Vmin[0]
        err_str = "C"
    elif Vo[1] + Vo[0] > 2*Vmax:
        Vf[1] = Vmax + Vo[1]/2 - Vo[0]/2
        Vf[0] = Vmax - Vo[1]/2 + Vo[0]/2
        err_str = "D"
    elif Vo[0] < Vmin[0] and Vo[1] > Vmin[1]:
        Vf[1] = Vmin[0] + Vo[1] - Vo[0]
        Vf[0] = Vmin[0]
        err_str = "E"
    elif Vo[1] < Vmin[1] and Vo[0] > Vmin[0]:
        Vf[1] = Vmin[1]
        Vf[0] = Vmin[1] + Vo[0] - Vo[1]
        err_str = "F"
    elif Vo[1] < Vmin[1] and Vo[0] < Vmin[0]:
        Vf[1] = Vmin[1]
        Vf[0] = Vmin[0]
        err_str = "G"

    if Vf[0] +0.0001 < Vmin[0]:
        print("ERROR A")
        print(Vf[0])
        print(Vmin[0])
        print(Vmin[1])
        print(2*Vmax)
        print(err_str)
    if Vf[1] +0.0001< Vmin[1]:
        print("ERROR B")
        print(err_str)
    if Vf[0] + Vf[1] > 2*Vmax + 0.001:
        print("ERROR C")
        print(err_str)

    C1 = [Vf[0] * pullyR/2, Vf[1] * pullyR/2]

    x_over = [0,0]
    for j in range(2):
        if x_f[j] > x_0[j]:
            C1[j] = abs(C1[j])
        elif x_f[j] < x_0[j]:
            C1[j] = - abs(C1[j])
        else:
            if x_p[j] > 0:
                C1[j] = - abs(C1[j])
            elif x_p[j] < 0:
                C1[j] = abs(C1[j])
            else:
                if x_pp[j] > 0:
                    C1[j] = - abs(C1[j])
                else:
                    C1[j] = abs(C1[j])

        if (x_p[j] < 0 and x_f[j] > x_0[j]) or (x_p[j] > 0 and x_f[j] < x_0[j]):
            pass
        else:
            try:
                #x_over[j] = -C1[j]*16.7785*(math.log((-C1[j]*0.836393)/(-C1[j]*0.836531-C2[j]*16.9975+C3[j]*345.371-C4[j]*7017.58))/(-20.319)-(C2[j]/C1[j]))
                x_over[j] = C2[j]/C7[j] - (C1[j]/(ab[j][1]*C7[j])*\
                        math.log((C1[j]*CE[j][3])/(-C1[j]*CE[j][1]+C2[j]*A2CE[j][1]+C3[j]*A3CE[j][1]+C4[j]*A4CE[j][1])))
                if x_f[j] > x_0[j] and x_f[j] < x_over[j]:
                    C1[j] = - abs(C1[j])
                elif x_f[j] < x_0[j] and x_f[j] > x_over[j]:
                    C1[j] = abs(C1[j])
            except ValueError:
                pass

    

    vt_1 = solve_vt1(x_f)
    vt_2 = [int((2*vt_1[0] - x_f[0]*C7[0]/C1[0]+C2[0]/C1[0])*10000), int((2*vt_1[1]-x_f[1]*C7[1]/C1[1]+C2[1]/C1[1])*10000)]
    vt_1 = [int((vt_1[0][0]) * 10000), int((vt_1[1][0])*10000)]

    Vf  = [int(2*C1[0]/pullyR*10000), int(2*C1[1]/pullyR*10000)]

    C2 = [int(val*100000000) for val in C2]
    C3 = [int(val*100000000) for val in C3]
    C4 = [int(val*100000000) for val in C4]

    sum = vt_1[0] + vt_1[1] + vt_2[0] + vt_2[1] + Vf[0] + Vf[1] + C2[0] + C2[1] + C3[0] + C3[1] + C4[0] + C4[1]
    
    data = struct.pack('<iiiiiiiiiiiii?',\
                       np.int32(vt_1[0]), np.int32(vt_1[1]),\
                       np.int32(vt_2[0]), np.int32(vt_2[1]),\
                       np.int32(Vf[0]), np.int32(Vf[1]),\
                       np.int32(C2[0]), np.int32(C2[1]),\
                       np.int32(C3[0]), np.int32(C3[1]),\
                       np.int32(C4[0]), np.int32(C4[1]),\
                       np.int32(sum), np.int8(idle))
    
    #Vx/y = u(t) Vf - 2 u(t-vt1) Vf + u(t-vt2) Vf
    #return {"a": vt_1, "b": vt_2, "v": Vf, "i": C2, "j": C3, "k": C4, "s": sum, "d": False}
    return data

#copied from tracker
def get_mallet(ser):
    FMT = '<hhhhhhB'    # 6×int16, 1×uint8
    FRAME_SIZE = struct.calcsize(FMT)
    def deq(q, xmin, xmax):
        y = q/32767
        return (y+1)/2 * (xmax - xmin) + xmin

    while ser.read(1) != b'\xAA':
        if not ser.in_waiting:
            return None, None, None, False

    raw = ser.read(FRAME_SIZE)
    if len(raw)!=FRAME_SIZE or ser.read(1)!=b'\x55':
        return None, None, None, False

    p0, p1, v0, v1, a0, a1, chk = struct.unpack(FMT, raw)

    # verify checksum
    c = 0
    for b in raw[:-1]:
        c ^= b
    if c!=chk:
        print("bad checksum", c, chk)
        return None, None, None, False
        

    pos = np.array([deq(p0, -1, 2), deq(p1, -1, 2)])
    vel = np.array([deq(v0, -30, 30), deq(v1, -30, 30)])
    acc = np.array([deq(a0, -150, 150), deq(a1, -150, 150)])
    return pos, vel, acc, True

def take_action(vision_data, mallet_data, mallet_buffer, writeQueue):
    #Transition to mallet coordinate system
    #vision data delay: time.time() - vision_data[0] + this function time + com + bluepill + ?
    obs = np.hstack((...))

    obs = TensorDict({"observation": torch.tensor(obs, dtype=torch.float32)})
                    
    policy_out = policy_module(obs)
    action = policy_out["action"].detach().numpy()
    xf = action[:2]
    xf[0] = np.maximum(margin_bottom+mallet_r, xf[0])
    xf[0] = np.minimum(table_bounds[0]/2-mallet_r, xf[0])

    xf[1] = np.maximum(margin+mallet_r, xf[1])
    xf[1] = np.minimum(table_bounds[1]-margin-mallet_r, xf[1])

    Vo = action[2:]

    #read from shared memory to get mallet pos, vel, acc
    new_mallet = mallet_buffer.read(True)

    data = update_path(new_mallet[1:3], new_mallet[3:5], new_mallet[5:7], xf, Vo)
    writeQueue.put(b'\n' + data + b'\n')

    #generate_top_down_view(vision_data[1:3], mallet_data[1:3], vision_data[3:5])

def mallet_calibration(ser):
    print("mallet_calibration")
    latest_msg = None
    while latest_msg is None:
        if ser.in_waiting:
            latest_msg = ser.read(ser.in_waiting).decode('utf-8')

    print(latest_msg) # ready to home

    print("State > ^ [enter]")
    input()
    print("Place extrusions [enter]")
    input()
    print("Place mallet bottom right [enter]")
    input()
    ser.write("x\n".encode())

    latest_msg = None
    while latest_msg is None:
        if ser.in_waiting:
            latest_msg = ser.read(ser.in_waiting).decode('utf-8')
    print(latest_msg) #confirmation of X

    print("Place mallet bottom left [enter]")
    input()
    ser.write("x\n".encode())

    latest_msg = None
    while latest_msg is None:
        if ser.in_waiting:
            latest_msg = ser.read(ser.in_waiting).decode('utf-8')
    print(latest_msg) #confirmation of Y

def choose_mode(ser):
    print("Choose a mode:")
    print("1: homing only")
    print("2: feedback only on final position")
    print("3: feedback only on path")
    print("4: feedforward only")
    print("5: feedback + feedforward")
    
    mode_str = input()
    mode = int(mode_str)
    mode_str += '\n'

    if ser.in_waiting:
        ser.read(ser.in_waiting).decode('utf-8')

    ser.write(mode_str.encode())

    print("Running mode:")
    latest_msg = None
    while latest_msg is None:
        if ser.in_waiting:
            latest_msg = ser.readline().decode().strip()
    print(latest_msg) #confirmation of mode

    if mode != 1:
        print("If starting, remove extrusions and ensure table is cleared [enter]")
        input()
        print("Power motors (> >) [enter]")
        input()
        print("Confirm to start [enter]")
        input()

        ser.write("begin\n".encode())

    return mode

def nn_mode(ser):
    mode_str = "5\n"
    time.sleep(1)

    if ser.in_waiting:
        latest_msg = ser.read(ser.in_waiting).decode('utf-8')

    ser.write(mode_str.encode())

    latest_msg = None
    while latest_msg is None:
        if ser.in_waiting:
            latest_msg = ser.read(ser.in_waiting).decode('utf-8')
    print("Running mode")
    print(latest_msg)

    print("If starting, remove extrusions and ensure table is cleared [enter]")
    input()
    print("Power motors (> >) [enter]")
    input()
    print("Confirm to start [enter]")
    input()

    ser.write("begin\n".encode())

def generate_top_down_view(puck_pos, mallet_pos, op_mallet_pos):

    top_down_image = np.ones((int(width * 500), int(height * 500), 3), dtype=np.uint8) * 255
    # Convert world coordinates to image coordinates.
    x_img = int((height - puck_pos[0]) * 500)  # scale factor for x
    y_img = int(puck_pos[1] * 500)  # invert y-axis for display
    cv2.circle(top_down_image, (x_img, y_img), int(puck_r * 500), (0, 255, 0), -1)

    x_img = int((height - op_mallet_pos[0]) * 500)  # scale factor for x
    y_img = int(op_mallet_pos[1] * 500)  # invert y-axis for display
    cv2.circle(top_down_image, (x_img, y_img), int(op_mallet_r * 500), (255, 255, 0), -1)

    x_img = int((height-mallet_pos[0]) * 500)  # scale factor for x
    y_img = int(mallet_pos[1] * 500)  # invert y-axis for display
    cv2.circle(top_down_image, (x_img, y_img), int(mallet_r* 500), (0, 255, 0), -1)
    cv2.imshow("top_down_table", top_down_image)
    cv2.waitKey(1)

def generate_mallet_topdown(mallet_pos):

    top_down_image = np.ones((int(width * 500), int(height * 500), 3), dtype=np.uint8) * 255

    x_img = int((height-mallet_pos[0]) * 500)  # scale factor for x
    y_img = int(mallet_pos[1] * 500)  # invert y-axis for display
    cv2.circle(top_down_image, (x_img, y_img), int(mallet_r* 500), (0, 255, 0), -1)
    cv2.imshow("top_down_table", top_down_image)
    cv2.waitKey(1)

if __name__ == "__main__":
    occilate = True

    ser = serial.Serial('COM3', 460800, timeout=0)

    if ser.in_waiting:
        ser.read(ser.in_waiting).decode('utf-8')

    ser.write(b'1')

    mallet_calibration(ser)

    while True:
        idleing = False
        mode = choose_mode(ser)

        pos = None
        vel = None
        acc = None
        passes = False
        while not passes:
            pos, vel, acc, passes = get_mallet(ser)

        ser.write("got_init\n".encode())

        time.sleep(0.2)

        if ser.in_waiting:
            ser.read(ser.in_waiting).decode('utf-8')

        if mode == 1:
            while True:

                pos, vel, acc, passes = get_mallet(ser)

                if passes:
                    generate_mallet_topdown(pos)
        elif mode == 2:
            xf = [[0.5, 0.5], [0.8,0.5]]
            i = 0
            current_time = time.time()

            data = {"xf": xf[i]}
            json_data = json.dumps(data)
            ser.write((json_data + "\n").encode())

            while True:
                if occilate and time.time() - current_time > 2:
                    current_time = time.time()
                    if i == 0:
                        i = 1
                    else:
                        i = 0

                    data = {"xf": xf[i]}
                    json_data = json.dumps(data)
                    ser.write((json_data + "\n").encode())

                pos, vel, acc, passes = get_mallet(ser)

                if passes:
                    generate_mallet_topdown(pos)
        else:
            xf = [[0.5, 0.5], [0.8,0.5]]
            Vo = np.array([5, 5])
            i = 0

            current_time = time.time()

            data = update_path(pos, vel, acc, xf[i], Vo)
            print("SENDING DATA")
            ser.write(b'\n' + data + b'\n')
            print(b'\n' + data)

            while True:
                if time.time() - current_time > 0.5:
                    current_time = time.time()
                    if occilate:
                        if i == 0:
                            i = 1
                        else:
                            i = 0

                    data = update_path(pos, vel, acc, xf[i], Vo)
                    ser.write(b'\n' + data + b'\n')

                pos, vel, acc, passes = get_mallet(ser)

                if passes:
                    generate_mallet_topdown(pos)
