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

obs_dim = 38
action_dim = 4
height = 1.993
width = 0.992

Vmax = 24 * 0.8
table_bounds = np.array([height, width])

mallet_r = 0.1011 / 2
margin_bounds = 0.04
mallet_bounds = np.array([[margin_bounds + mallet_r, table_bounds[0]/2  - mallet_r/2], [margin_bounds+mallet_r, table_bounds[1]-margin_bounds-mallet_r]])


class ScaledNormalParamExtractor(NormalParamExtractor):
    def __init__(self, scale_factor=1.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        loc, scale = super().forward(x)
        #scale *= self.scale_factor
        scale = 2*self.scale_factor/ (1+torch.exp(-scale)) - self.scale_factor + 0.001
        return loc, scale

policy_net = nn.Sequential(
    nn.Linear(obs_dim, 1024),
    nn.LayerNorm(1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.LayerNorm(1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.LayerNorm(512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, action_dim * 2),
    ScaledNormalParamExtractor(),
)

policy_module = TensorDictModule(
    policy_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

low = torch.tensor([mallet_r + 0.01, mallet_r + 0.01, 0, -1], dtype=torch.float32)
high = torch.tensor([table_bounds[0]/2-mallet_r-0.01, table_bounds[1]-mallet_r-0.01, 1, 1], dtype=torch.float32)

policy_module = ProbabilisticActor(
    module=policy_module,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": low,
        "high": high,
    },
    #default_interaction_type=tensordict.nn.InteractionType.RANDOM,
)
# Vx + Vy < 2*(Vmax)
# sqrt(2* (2*Vmax)^2)

checkpoint = torch.load("model_209.pth", map_location="cpu")
policy_module.load_state_dict(checkpoint['policy_state_dict'])
del checkpoint
#policy_module.load_state_dict(torch.load("model_183.pth"), map_location=torch.device("cpu")) #8
#policy_module.eval()

#pullyR = 0.035755
#a1 = 2.725e-05 
#a2 = 7.575e-03  
#a3 = 6.969e-02 
#b1 = -1.996e-05 
#b2 = -2.838e-03 
#b3 = 3.688e-03 

pullyR = 0.035755
#a1 = 3.124e-05 
#a2 = 9.974e-03  
#a3 = 9.175e-02 
#b1 = -3.026e-05 
#b2 = -3.803e-03 
#b3 = 1.025e-02 

#Mauro coeffs
a1 = 1.664e-05 
a2 = 6.802e-03  
a3 = 6.703e-02 
b1 = -1.113e-05 
b2 = -2.796e-03 
b3 = 6.535e-03 

C1 = [Vmax * pullyR / 2, Vmax * pullyR / 2]
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

Vf = [0,0]
vt_1 = [0,0]
vt_2 = [0,0]

#A e^(at) + B e^(bt) + Ct + D
def f(x, i, eat, ebt):
    return CE[i][0] * eat \
           + CE[i][1] * ebt + CE[i][2]*x+CE[i][3]

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
    if x[0] <= 0:
        return 1 - x[0]
    return C2[k]/C7[k] - (x[0]/(ab[k][1]*C7[k])*\
                math.log((x[0]*CE[k][3])/(-x[0]*CE[k][1]+C2[k]*A2CE[k][1]+C3[k]*A3CE[k][1]+C4[k]*A4CE[k][1]))) - mallet_bounds[k][1]

def solve_C1n(x, k):
    if x[0] >= 0:
        return 1 + x[0]
    return C2[k]/C7[k] - (x[0]/(ab[k][1]*C7[k])*\
                math.log((x[0]*CE[k][3])/(-x[0]*CE[k][1]+C2[k]*A2CE[k][1]+C3[k]*A3CE[k][1]+C4[k]*A4CE[k][1]))) - mallet_bounds[k][0]
                
def get_IC(t):
    pos = [0,0]
    vel = [0,0]
    acc = [0,0]
    
    for i in range(2):
        eat = np.exp(ab[i][0] * t)
        ebt = np.exp(ab[i][1] * t)
        
        eatp = ab[i][0] * eat
        ebtp = ab[i][1] * ebt
        
        eatpp = ab[i][0] * eatp
        ebtpp = ab[i][1] * ebtp
        
        A_pos = [A2CE[i][0] * eat + A2CE[i][1] * ebt + A2CE[i][2],\
                 A3CE[i][0] * eat + A3CE[i][1] * ebt,\
                 A4CE[i][0] * eat + A4CE[i][1] * ebt]
                 
        A_vel = [A2CE[i][0] * eatp + A2CE[i][1] * ebtp,\
                 A3CE[i][0] * eatp + A3CE[i][1] * ebtp,\
                 A4CE[i][0] * eatp + A4CE[i][1] * ebtp]
        
        A_acc = [A2CE[i][0] * eatpp + A2CE[i][1] * ebtpp,\
                 A3CE[i][0] * eatpp + A3CE[i][1] * ebtpp,\
                 A4CE[i][0] * eatpp + A4CE[i][1] * ebtpp]
        
        #A_2 = A2CE[i][0] * eat + A2CE[i][1] * ebt + A2CE[i][2]
        #A_3 = A3CE[i][0] * eat + A3CE[i][1] * ebt
        #A_4 = A4CE[i][0] * eat + A4CE[i][1] * ebt
        
        f_t = [CE[i][0] * eat + CE[i][1] * ebt + CE[i][2]*t+CE[i][3],\
               CE[i][0] * eatp + CE[i][1] * ebtp + CE[i][2],\
               CE[i][0] * eatpp + CE[i][1] * ebtpp]
    
        g_a1 = [0,0,0]
        g_a2 = [0,0,0]
        
        
        tms = t-vt_1[i]
        if tms > 0:
            eatms = np.exp(ab[i][0]*tms)
            ebtms = np.exp(ab[i][1]*tms)
            g_a1[0] = CE[i][0] * eatms + CE[i][1] * ebtms + CE[i][2]*tms+CE[i][3]
            g_a1[1] = CE[i][0] * ab[i][0] * eatms + CE[i][1] * ab[i][1] * ebtms + CE[i][2]
            g_a1[2] = CE[i][0] * ab[i][0]**2 * eatms + CE[i][1] * ab[i][1]**2 * ebtms
        
        tms = t-vt_2[i]
        if tms > 0:
            eatms = np.exp(ab[i][0]*tms)
            ebtms = np.exp(ab[i][1]*tms)
            g_a2[0] = CE[i][0] * eatms + CE[i][1] * ebtms + CE[i][2]*tms+CE[i][3]
            g_a2[1] = CE[i][0] * ab[i][0] * eatms + CE[i][1] * ab[i][1] * ebtms + CE[i][2]
            g_a2[2] = CE[i][0] * ab[i][0]**2 * eatms + CE[i][1] * ab[i][1]**2 * ebtms
        
        pos[i] = 0.5 * Vf[i] * pullyR * (f_t[0] - 2 * g_a1[0] + g_a2[0]) + C2[i] * A_pos[0] + C3[i] * A_pos[1] + C4[i] * A_pos[2]
        vel[i] = 0.5 * Vf[i] * pullyR * (f_t[1] - 2 * g_a1[1] + g_a2[1]) + C2[i] * A_vel[0] + C3[i] * A_vel[1] + C4[i] * A_vel[2]
        acc[i] = 0.5 * Vf[i] * pullyR * (f_t[2] - 2 * g_a1[2] + g_a2[2]) + C2[i] * A_acc[0] + C3[i] * A_acc[1] + C4[i] * A_acc[2]

    return np.array(pos), np.array(vel), np.array(acc)
    
    

def update_path(x_0, x_p, x_pp, x_f, Vo):
    global C1
    global C2
    global C3
    global C4
    global Vf
    global vt_1
    global vt_2

    C2 = [C5[0]*x_pp[0]+C6[0]*x_p[0]+C7[0]*x_0[0], C5[1]*x_pp[1]+C6[1]*x_p[1]+C7[1]*x_0[1]]
    C3 = [C5[0]*x_p[0]+C6[0]*x_0[0],C5[1]*x_p[1]+C6[1]*x_0[1]]
    C4 = [C5[0]*x_0[0],C5[1]*x_0[1]]

    Vmin = [0,0]

    for j in range(2):
        if x_p[j] > 0 and C2[j]/C7[j] > mallet_bounds[j][1]:
            if x_0[j] > mallet_bounds[j][1] or solve_C1p([Vmax*pullyR], j) > 0:
                Vmin[j] = 2*Vmax
                C1[j] = Vmin[j] * pullyR/2
            else:
                val, info, ier, msg = fsolve(solve_C1p, x0=0.05, xtol=1e-4, full_output=True, args=(j))
                if ier != 1:
                    val, info, ier, msg = fsolve(solve_C1p, x0=0.005, xtol=1e-4, full_output=True, args=(j))
                    if ier != 1:
                        Vmin[j] = 0.2
                        C1[j] = Vmin[j] * pullyR/2
                        print("C1p failed corvergence")

                C1[j] = val[0]
                Vmin[j] = C1[j] * 2/pullyR

            #solve C1 > 0 so that x_over[j] = bounds[1]
        elif x_p[j] < 0 and C2[j]/C7[j] < mallet_bounds[j][0]:
            if x_0[j] < mallet_bounds[j][0] or solve_C1n([-Vmax*pullyR], j) < 0:
                Vmin[j] = 2*Vmax
                C1[j] = -Vmin[j] * pullyR/2
            else:
                val, info, ier, msg = fsolve(solve_C1n, x0=-0.05, xtol=1e-4, full_output=True, args=(j))
                if ier != 1:
                    val, info, ier, msg = fsolve(solve_C1n, x0=-0.05/(10), xtol=1e-4, full_output=True, args=(j))

                    if ier != 1:
                        Vmin[j] = 0.2
                        C1[j] = -Vmin[j] * pullyR/2
                        print("C1n failed corvergence")

                C1[j] = val[0]

                Vmin[j] = abs(C1[j]) * 2/pullyR
        # set magntiude of C1
        
    Vf = [0,0]

    Vmin[0] = max(Vmin[0], 0.01)
    Vmin[1] = max(Vmin[1], 0.01)
    Vmin[1] = min(Vmin[1], 2*Vmax)
    Vmin[0] = min(Vmin[0], 2*Vmax)
    if Vmin[0] + Vmin[1] > 2*Vmax:
        sum = Vmin[0] + Vmin[1] + 0.0001
        Vmin[0] *= 2*Vmax/sum
        Vmin[1] *= 2*Vmax/sum
    #print("--")   
    #print(Vmin)
    #print(Vo)

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
    vt_1 = [vt_1[0][0], vt_1[1][0]]
    vt_2 = [2*vt_1[0] - x_f[0]*C7[0]/C1[0]+C2[0]/C1[0], 2*vt_1[1]-x_f[1]*C7[1]/C1[1]+C2[1]/C1[1]]
    vt_2_int = [int(val*10000) for val in vt_2]
    
    vt_1_int = [int(val*10000) for val in vt_1]

    Vf = [2*C1[0]/pullyR, 2*C1[1]/pullyR]
    Vf_int  = [int(val*10000) for val in Vf]
    
    #print("data")
    #print(x_0)
    #print(x_p)
    #print(x_pp)
    #print(np.array(vt_1)/10000.0)
    #print(np.array(vt_2)/10000.0)
    #print(np.array(Vf)/10000.0)
    
    C2_int = [int(val*100000000) for val in C2]
    C3_int = [int(val*100000000) for val in C3]
    C4_int = [int(val*100000000) for val in C4]

    checksum = vt_1_int[0] ^ vt_1_int[1] ^ vt_2_int[0] ^ vt_2_int[1] ^ Vf_int[0] ^ Vf_int[1] ^ C2_int[0] ^ C2_int[1] ^ C3_int[0] ^ C3_int[1] ^ C4_int[0] ^ C4_int[1]
    
    data = struct.pack('<iiiiiiiiiiiii',\
                       np.int32(vt_1_int[0]), np.int32(vt_1_int[1]),\
                       np.int32(vt_2_int[0]), np.int32(vt_2_int[1]),\
                       np.int32(Vf_int[0]), np.int32(Vf_int[1]),\
                       np.int32(C2_int[0]), np.int32(C2_int[1]),\
                       np.int32(C3_int[0]), np.int32(C3_int[1]),\
                       np.int32(C4_int[0]), np.int32(C4_int[1]),\
                       np.int32(checksum))
    
    return data

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
