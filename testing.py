import tracker
import time
import cv2
import numpy as np
import threading
import queue
import multiprocessing as mp
import os
import psutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
import tensordict
from tensordict import TensorDict, TensorDictBase
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
from torchrl.objectives.value import GAE
from collections import deque
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
import time

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
import os
import ctypes
from ctypes import wintypes
import threading

def set_high_performance_mode():
    """Set Windows to high performance mode and disable CPU throttling"""
    try:
        # Set thread priority to real-time
        kernel32 = ctypes.windll.kernel32
        thread_handle = kernel32.GetCurrentThread()
        kernel32.SetThreadPriority(thread_handle, 15)  # THREAD_PRIORITY_TIME_CRITICAL
        
        # Set process priority to high
        process_handle = kernel32.GetCurrentProcess()
        kernel32.SetPriorityClass(process_handle, 0x00000080)  # HIGH_PRIORITY_CLASS
        
        return True
    except Exception as e:
        print(f"Failed to set high performance mode: {e}")
        return False

def disable_windows_optimizations():
    """Disable Windows features that can cause timing jitter"""
    try:
        # Disable CPU parking (requires admin rights)
        os.system('powercfg -setacvalueindex scheme_current sub_processor PROCTHROTTLEMIN 100')
        os.system('powercfg -setactive scheme_current')
        
        # Set timer resolution to 1ms (highest precision)
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod(1)
        
        return True
    except Exception as e:
        print(f"Failed to disable Windows optimizations: {e}")
        return False

obs_dim = 40
action_dim = 4
mallet_r = 0.04

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

policy_module = ProbabilisticActor(
    module=policy_module,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": torch.tensor([mallet_r, mallet_r, 0.3, 0.3]),
        "high": torch.tensor([1-mallet_r, 1 - mallet_r, 10, 10]),
    },
    #default_interaction_type=tensordict.nn.InteractionType.RANDOM,
)


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

def take_action_test(camera_tracker, img):
    #Transition to mallet coordinate system
    #vision data delay: time.time() - vision_data[0] + this function time + com + bluepill + ?
    
    puck_pos, op_mallet_pos = camera_tracker.process_frame(img)
    puck_pos = np.array([1.0 - puck_pos[0], 1.0 - puck_pos[1]])
    op_mallet_pos = np.array([2.0 - op_mallet_pos[0], 1.0 - op_mallet_pos[1]])

    obs = torch.rand((obs_dim))

    obs = TensorDict({"observation": obs})
                    
    policy_out = policy_module(obs)
    action = policy_out["action"].detach().cpu().numpy()
    xf = action[:2]
    xf[0] = np.maximum(margin_bottom+mallet_r, xf[0])
    xf[0] = np.minimum(table_bounds[0]/2-mallet_r, xf[0])

    xf[1] = np.maximum(margin+mallet_r, xf[1])
    xf[1] = np.minimum(table_bounds[1]-margin-mallet_r, xf[1])

    Vo = action[2:]

    #read from shared memory to get mallet pos, vel, acc
    new_mallet = torch.rand((7,)).numpy()
    new_mallet[1:3] = (new_mallet[1:3] + 0.3) * 0.3
    new_mallet[3:5] *= 10

    data = update_path(new_mallet[1:3], new_mallet[3:5], new_mallet[5:7], xf, Vo)
    #writeQueue.put(b'\n' + data + b'\n')

def benchmark_worker(repeats, result_queue):
    """Worker function that runs in a separate process"""
    
    # Force garbage collection before starting
    
    current_process = psutil.Process()
    current_process.cpu_affinity([2,3,4,5,6])
    current_process.nice(psutil.HIGH_PRIORITY_CLASS)

    track = tracker.SetupCamera()

    # Load and run extrinsics
    img = cv2.imread("jump_bright.bmp", cv2.IMREAD_GRAYSCALE)
    track.run_extrinsics(img)

    track = tracker.CameraTracker(track.rotation_matrix, track.translation_vector, track.z_pixel_map, (120.94)*10**(-3))
    
    times = np.empty(repeats, dtype=np.float64)

    op_mallet_z = (120.94)*10**(-3)
    camera_tracker = tracker.CameraTracker(track.rotation_matrix, track.translation_vector, track.z_pixel_map, op_mallet_z)

    img = cv2.imread("jump.bmp", cv2.IMREAD_GRAYSCALE)
    
    # Extended warm-up to stabilize caches and CPU frequency
    #for _ in range(50):  # Increased from 5
    #    puck_pos, op_mallet_pos = camera_tracker.process_frame(img)
    #    # Force CPU to stay at high frequency
    #    dummy = np.random.rand(1000) @ np.random.rand(1000)

    # Pre-allocate all arrays to avoid memory allocation during timing
    puck_pos_array = np.empty(2)
    op_mallet_pos_array = np.empty(2)
    obs = torch.rand((obs_dim))
    obs = TensorDict({"observation": obs})
    new_mallet = torch.rand((7,)).numpy()
    xf = np.empty(2)


    for i in range(50):
        # Use more precise timing
        t0 = time.perf_counter_ns()  # Nanosecond precision
        
        puck_pos, op_mallet_pos = camera_tracker.process_frame(img)
        puck_pos_array[0] = 1.0 - puck_pos[0]
        puck_pos_array[1] = 1.0 - puck_pos[1]
        op_mallet_pos_array[0] = 2.0 - op_mallet_pos[0]
        op_mallet_pos_array[1] = 1.0 - op_mallet_pos[1]

        # Reuse pre-allocated tensors instead of creating new ones
        obs["observation"] = torch.rand((obs_dim))
                        
        policy_out = policy_module(obs)
        action = policy_out["action"].detach().cpu().numpy()
        
        xf[0] = action[0]
        xf[1] = action[1]
        xf[0] = np.maximum(margin_bottom+mallet_r, xf[0])
        xf[0] = np.minimum(table_bounds[0]/2-mallet_r, xf[0])
        xf[1] = np.maximum(margin+mallet_r, xf[1])
        xf[1] = np.minimum(table_bounds[1]-margin-mallet_r, xf[1])

        Vo = action[2:]

        # Reuse pre-allocated array
        new_mallet[:] = torch.rand((7,)).numpy()
        new_mallet[1:3] = (new_mallet[1:3] + 0.3) * 0.3
        new_mallet[3:5] *= 10

        data = update_path(new_mallet[1:3], new_mallet[3:5], new_mallet[5:7], xf, Vo)
        times[i] = (time.perf_counter_ns() - t0) / 1e6  # Convert to milliseconds
        
    
    for i in range(repeats):
        # Use more precise timing
        t0 = time.perf_counter_ns()  # Nanosecond precision
        
        puck_pos, op_mallet_pos = camera_tracker.process_frame(img)
        puck_pos_array[0] = 1.0 - puck_pos[0]
        puck_pos_array[1] = 1.0 - puck_pos[1]
        op_mallet_pos_array[0] = 2.0 - op_mallet_pos[0]
        op_mallet_pos_array[1] = 1.0 - op_mallet_pos[1]
        
        # Reuse pre-allocated tensors instead of creating new ones
        obs["observation"] = torch.rand((obs_dim))
                        
        policy_out = policy_module(obs)
        action = policy_out["action"].detach().cpu().numpy()
        
        xf[0] = action[0]
        xf[1] = action[1]
        xf[0] = np.maximum(margin_bottom+mallet_r, xf[0])
        xf[0] = np.minimum(table_bounds[0]/2-mallet_r, xf[0])
        xf[1] = np.maximum(margin+mallet_r, xf[1])
        xf[1] = np.minimum(table_bounds[1]-margin-mallet_r, xf[1])

        Vo = action[2:]

        # Reuse pre-allocated array
        new_mallet[:] = torch.rand((7,)).numpy()
        new_mallet[1:3] = (new_mallet[1:3] + 0.3) * 0.3
        new_mallet[3:5] *= 10

        data = update_path(new_mallet[1:3], new_mallet[3:5], new_mallet[5:7], xf, Vo)
        times[i] = (time.perf_counter_ns() - t0) / 1e6  # Convert to milliseconds
        
    
    # Re-enable garbage collection
    
    print(times)
    result = {
        'min': np.min(times),
        'avg': np.mean(times),
        'std': np.std(times),
        'max': np.max(times)
    }
    
    result_queue.put(result)

def benchmark_single_cpu(repeats=50):
    """Run benchmark on a single CPU core"""
    # Create a queue for results
    result_queue = mp.Queue()
    
    # Create process with single CPU affinity
    process = mp.Process(target=benchmark_worker, args=(repeats, result_queue))
    
    # Start the process
    process.start()
    
    # Set CPU affinity to core 0 (you can change this to any core
    
    # Wait for process to complete and get results
    process.join()
    
    if not result_queue.empty():
        result = result_queue.get()
        print(f"Min: {result['min']:.3f} ms | "
              f"Avg: {result['avg']:.3f} ms | "
              f"Std: {result['std']:.3f} ms | "
              f"Max: {result['max']:.3f} ms")
    else:
        print("No results received from benchmark process")

#def listen_for_input():
#    """Runs in a separate thread to get user input and add it to the queue."""
#    user_input = input("\n stop and or change modes [enter]")  # Get user input
#    input_queue.put(user_input)

if __name__ == '__main__':
    # Set single threading for main process as well
    
    #input_queue = queue.Queue()
    #input_thread = threading.Thread(target=listen_for_input, daemon=True)
    #input_thread.start()
    
    # Run benchmark on single CPU
    benchmark_single_cpu(repeats=500)
  