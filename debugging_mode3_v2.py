import pygame
import math
from scipy.optimize import fsolve
from scipy.special import lambertw
import numpy as np
import random
# Initialize pygame
pygame.init()

print("STARTING")

# Screen dimensions
screen_width = 1000
screen_height = 500
plr = 500

surface = [screen_width / plr, screen_height / plr]

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)

# Set up the display
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Air Hockey")
# Circle parameters
mallet_pos = [500, 200]  # Starting position of the circle
mallet_radius = 0.05
circle_radius_m = int(mallet_radius * plr)
k=0

puck_pos = [np.float64(0.26301700197405026), np.float64(0.38141603103510346)]
puck_vel = [np.float64(0.04046566572581313), np.float64(0.25595676486620184)]
goal_width = 0.3

B = 0.01 #drag
friction = 0.01 #friction
res = 0.9 #restitution
mass = 0.1

puck_radius = 0.05
circle_pos = [puck_pos[0] * plr, puck_pos[0] * plr]
circle_radius = puck_radius * plr

Vmax = 24
Vmin = [0,0]
pullyR = 0.035306
x_0 = [np.float64(0.06865115513830858), np.float64(0.56509673880989)]
x_p = [np.float64(0.0004461810708884942), np.float64(6.651376845717271e-05)]
x_pp = [np.float64(-0.008073583677799853), np.float64(-0.0004200799622089413)]
mallet_pos = [x_0[0] * plr, x_0[1] * plr]
margin = 0.001
bounds = [[mallet_radius+margin, (screen_width - circle_radius_m) / plr - margin],[mallet_radius + margin, (screen_height - circle_radius_m)/plr - margin]]

C1 = [14 * pullyR / 2, 14 * pullyR / 2] #[Vmax * pullyR / 2, Vmax * pullyR / 2]
a1 = 3.579*10**(-6)
a2 = 0.00571
a3 = (0.0596+0.0467)/2
b1 = -1.7165*10**(-6)
b2 = -0.002739
b3 = 0
C5 = [a1+b1, a1-b1]
C6 = [a2+b2, a2-b2]
C7 = [a3+b3, a3-b3]

C = 0
D = 0
wall_val = 0

count_err = 0


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



C2 = [C5[0]*x_pp[0]+C6[0]*x_p[0]+C7[0]*x_0[0], C5[1]*x_pp[1]+C6[1]*x_p[1]+C7[1]*x_0[1]]
C3 = [C5[0]*x_p[0]+C6[0]*x_0[0],C5[1]*x_p[1]+C6[1]*x_0[1]]
C4 = [C5[0]*x_0[0],C5[1]*x_0[1]]
x_f = [0.5,0.5]
a = [0.01, 0.01]
# Marker list to store mouse click positions

# Time tracking
clock = pygame.time.Clock()
Ts = 0.2  # 0.2 seconds between each marker move

# Main loop
running = True

def dP(t):
    #return C*np.exp(-(B/mass)*t) + D - (friction/B)*t
    return (mass/B) * np.log(np.cos((np.sqrt(B*friction)*(C*mass+t))/mass)) + D

def velocity(t):
    #return (-B/mass)*C*np.exp(-(B/mass)*t) - (friction/B)
    return -np.sqrt(friction/B) * np.tan((np.sqrt(B*friction)*(C*mass+t))/mass)

def get_tPZero():
    #return -(mass/B)*np.log(-friction*mass/(C*B**2))
    return -C*mass

def getC(v_norm):
    #return -(v_norm + friction/B)*(mass/B)
    return - np.arctan(v_norm * np.sqrt(B/friction)) / np.sqrt(B*friction)

def getD():
    #return -C
    return - (mass/B) * np.log(np.cos(np.sqrt(B*friction)*C))


def pos_wall(t):
    return  dP(t) - wall_val


def corner_collision(A, pos, vel, t, final_pos, C, D, tPZero, v_norm, dPZero, dir, dPt, vt, t_init):
    global wall_val
    a = vel[0]**2 + vel[1]**2
    b = 2*pos[0]*vel[0] + 2*pos[1]*vel[1]-2*A[0]*vel[0]-2*A[1]*vel[1]
    c = pos[0]**2+pos[1]**2+A[0]**2+A[1]**2-puck_radius**2-2*A[0]*pos[0]-2*A[1]*pos[1]

    if b**2 - 4*a*c < 0:
        print(puck_TP)
        print(puck_VTP)
        print(x_0P)
        print(x_pP)
        print(x_ppP)
        print(markerP)
        print(marker)
    s = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

    new_pos = [0,0]
    new_pos[0] = pos[0] + s * vel[0]
    new_pos[1] = pos[1] + s * vel[1]
    wall_val = s * v_norm
    thit_min = -1
    if dPZero > wall_val:
        val, info, ier, msg = fsolve(pos_wall, x0=wall_val/v_norm, xtol=1e-6, full_output=True)
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
        arr = puck_mallet_collision(pos, vel, velocity(t), thit, t_init)
        #x_p, x_m, v_p, v_m, dt_sum
        if len(arr) > 1:
            new_pos = arr[0]
            new_vel = arr[2]
            return update_puck(t-arr[4], new_pos, new_vel, t_init+arr[4])
        for j in range(2):
            if t > tPZero:
                new_pos[j] = final_pos[j]
            else:
                new_pos[j] = pos[j] + dPt * dir[j]
                new_vel[j] = vt * dir[j]
        return new_pos, new_vel
    
    arr = puck_mallet_collision(pos, vel, velocity(t), thit_min, t_init)
    #x_p, x_m, v_p, v_m, dt_sum
    if len(arr) > 1:
        new_pos = arr[0]
        new_vel = arr[2]
        return update_puck(t-arr[4], new_pos, new_vel, t_init+arr[4])

    vt_hit = velocity(thit_min)
    for j in range(2):
        new_vel[j] = vt_hit * dir[j]

    n = np.array([new_pos[0] - A[0], new_pos[1] - A[1]])
    n = n / np.sqrt(np.dot(n, n))
    tangent = np.array([n[1], -n[0]])
    vel_r = np.array(new_vel)

    vel_r = np.array([-res*np.dot(vel_r, n), res*np.dot(vel_r, tangent)])
    new_vel = n * vel_r[0] + tangent * vel_r[1]

    return update_puck(t-thit_min, new_pos, new_vel, t_init+thit_min)

def check_in_goal(vel, pos, w):
    A = 0
    bounces = False
    s2 = 0
    if pos[0] < puck_radius:
        if vel[0] > 0:
            if w[0] > 0:
                w *= -1
            if pos[0] + w[0] < 0:
                s2 = (-pos[0] - w[0]) / vel[0]
                if surface[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w[1] or vel[1]*s2+pos[1]+w[1] >  surface[1]/2 + goal_width/2:
                    bounces = True
        elif vel[0] < 0:
            if w[0] < 0:
                w *= -1
            if pos[0] + w[0] > 0:
                s2 = (-pos[0]-w[0]) / vel[0]
                if surface[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w[1] or vel[1]*s2+pos[1]+w[1] >  surface[1]/2 + goal_width/2:
                    bounces = True
        if bounces:
            A = 0
    elif pos[0] > surface[0]-puck_radius:
        if vel[0] > 0:
            if w[0] > 0:
                w *= -1
            if pos[0] + w[0] < surface[0]:
                s2 = (-pos[0] - w[0] + surface[0]) / vel[0]
                if surface[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w[1] or vel[1]*s2+pos[1]+w[1] >  surface[1]/2 + goal_width/2:
                    bounces = True
        elif vel[0] < 0:
            if w[0] < 0:
                w *= -1
            if pos[0] + w[0] > surface[0]:
                s2 = (-pos[0]-w[0]+surface[0]) / vel[0]
                if surface[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w[1] or vel[1]*s2+pos[1]+w[1] >  surface[1]/2 + goal_width/2:
                    bounces = True
        if bounces:
            A = surface[0] 

    return A, bounces, s2

def check_line_intersects_square(x_bounds, y_bounds, initial, final):
    # Define the square using the given bounds
    square = Polygon([
        (x_bounds[0], y_bounds[0]),
        (x_bounds[1], y_bounds[0]),
        (x_bounds[1], y_bounds[1]),
        (x_bounds[0], y_bounds[1])
    ])
    
    # Define the line segment
    line = LineString([initial, final])
    
    # Check for intersection
    return line.intersects(square)

def puck_mallet_collision(x_p,v_p, vpf, dt,time):
    global mallet_pos
    global puck_radius
    global mallet_radius
    global count_err
    global C, D
    global mallet_x_bounds, mallet_y_bounds

    x_p = np.array(x_p)
    v_p = np.array(v_p)
    dt_dynamic = 0
    dt_sum = 0
    v_norm = np.sqrt(v_p[0]**2 + v_p[1]**2)
    dir = np.array([0,1])
    if v_norm != 0:
        dir = np.array([v_p[0] / v_norm, v_p[1] / v_norm])

    if not check_line_intersects_square([mallet_x_bounds[0]-puck_radius, mallet_x_bounds[1]+puck_radius], [mallet_y_bounds[0]-puck_radius,mallet_y_bounds[1]+puck_radius], x_p, x_p + dP(dt)*dir):
        return [-1]

    x_m = get_pos(time+dt_sum)
    v_m = get_xp(time+dt_sum)

    while True:
        
        dt_dynamic = puck_mallet_collision_t(x_p, x_m, v_p, v_m, vpf)
        dt_sum += dt_dynamic
        if dt_sum > dt or dt_dynamic == -1:
            return [-1]
        
        C = getC(np.sqrt(v_p[0]**2 + v_p[1]**2))
        D = getD()

        x_p += dP(dt_dynamic) * dir
        v_p = velocity(dt_dynamic) * dir
        x_m = get_pos(time+dt_sum)
        v_m = get_xp(time+dt_sum)
        if np.sqrt((x_p[0] - x_m[0])**2 + (x_p[1] - x_m[1])**2) < puck_radius + mallet_radius + 1e-4:
            v_rel = np.array([v_p[0] - v_m[0], v_p[1] - v_m[1]])
            normal = np.array([x_p[0] - x_m[0], x_p[1] - x_m[1]])
            normal = normal / np.sqrt(np.dot(normal,normal))
            tangent = np.array([-normal[1], normal[0]])

            if np.dot(v_rel, normal) < 0:
                v_rel_col = tangent * np.dot(v_rel, tangent) - res * normal * np.dot(v_rel, normal)
                v_p = [v_rel_col[0] + v_m[0], v_rel_col[1] + v_m[1]]
                return [x_p, x_m, v_p, v_m, dt_sum]

            

def puck_mallet_collision_t(x_p, x_m, v_p, v_m, vpf):
    v_max_mallet = max(np.sqrt((C1[0]*CE[0][2])**2+(C1[1]*CE[1][2])**2), np.sqrt(v_m[0]**2 + v_m[1]**2))
    vp_norm = np.sqrt(v_p[0]**2 + v_p[1]**2)
    
    d_p = [1,0]
    if vp_norm != 0:
        d_p = [v_p[0]/vp_norm, v_p[1]/vp_norm]
    w = np.array([-d_p[1], d_p[0]])
    x = np.array([x_m[0] - x_p[0], x_m[1] - x_p[1]])
    
    d_perp = np.dot(w,x)
    d_par = np.dot(d_p, x)
    vP = 0
    R = puck_radius + mallet_radius
    
    if abs(d_perp) > R:
        if v_max_mallet == 0:
            return -1
        t_col = (abs(d_perp) - R) / v_max_mallet
        vP = d_par / t_col
        if vP < vpf:
            vP = vpf
        elif vP > vp_norm:
            vP = vp_norm
    else:
        if d_par > 0:
            vP = vp_norm
        else:
            vP = vpf

    a = v_max_mallet * d_perp
    b = v_max_mallet * d_par + R*vP
    c = -vP*d_perp

    t = 0
    try:
        theta_n = 0
        if b != c:
            theta_n = 2 * math.atan((a-math.sqrt(a**2+b**2-c**2))/(b-c))
        else:
            theta_n = -2 * math.atan(b/a) + math.pi / math.sqrt(2)
        if abs(theta_n) != np.pi/2:
            t = math.sqrt((abs(d_perp)-R*abs(math.cos(theta_n)))**2 + (math.sin(theta_n))**2 * (abs(d_perp/math.cos(theta_n)) - R)**2) / v_max_mallet
        else:
            t = (-math.sin(theta_n) * R - d_par) / (math.sin(theta_n)*v_max_mallet-vP)
    except:
        return -1

    return t

def update_puck(t, pos, vel, t_init):
    global wall_val
    global C
    global D
    global count_err
    new_pos = [0,0]
    new_vel = [0,0]
    final_pos = [0,0]
    thit = -1
    tPZero = 0
    w = np.array([vel[1], -vel[0]])
    v_norm = np.sqrt(np.dot(w,w))
    if v_norm != 0:
        w = puck_radius * w / v_norm
    else:
        arr = puck_mallet_collision(pos, vel, velocity(t), t, t_init)
        #x_p, x_m, v_p, v_m, dt_sum
        if len(arr) > 1:
            new_pos = arr[0]
            new_vel = arr[2]
            return update_puck(t-arr[4], new_pos, new_vel, t_init + arr[4])
        return pos, vel
    dir = [vel[0] / v_norm, vel[1] / v_norm]

    #Compute constants for movement
    
    """
    if v_norm != 0:
        w = puck_radius * w / v_norm
        f[0] = friction * vel[0] / v_norm
        f[1] = friction * vel[1] / v_norm
    else:
        return pos, vel
    for j in range(2):
        if abs(vel[j]) > 0:
            C[j] = -(vel[j] + f[j]/B)*(mass/B)
            D[j] = pos[j] - C[j]
            tPZero[j] = tStop(C, j)
            final_pos[j] = C[j]*np.exp(-(B/mass)*tPZero[j]) + D[j] - (f[j]/B)*tPZero[j]
        else:
            C[j] = 0
            D[j] = pos[j] - C[j]
            tPZero[j] = 0
            final_pos[j] = pos[j]
    """

    C = getC(v_norm)
    D = getD()
    tPZero = get_tPZero()
    dPZero = dP(tPZero)
    final_pos[0] = pos[0] + dir[0] * dPZero
    final_pos[1] = pos[1] + dir[1] * dPZero

    #check if it lies outside the playing area (i.e. somewhat in the goal) and will collide with a corner
    bounces = False
    A = [0,0]
    A[0], bounces, s2 = check_in_goal(vel, pos, w)
    dPt = dP(t)
    vt = velocity(t)
    
    if bounces:
        if surface[1]/2 - goal_width/2 > vel[1]*s2+pos[1]+w[1]:
            A[1] = surface[1]/2 - goal_width/2
        elif vel[1]*s2+pos[1]+w[1] >  surface[1]/2 + goal_width/2:
            A[1] = surface[1]/2 + goal_width/2

        return corner_collision(A, pos, vel, t, final_pos, C, D, tPZero, v_norm, dPZero, dir, dPt, vt, t_init)
        
    
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
            s1 = (-pos[0] - w[0] + surface[0]) / vel[0]
            s2 = (-pos[0] + w[0] + surface[0]) / vel[0]
        y1 = pos[1] + w[1] + s1*vel[1]
        y2 = pos[1] - w[1] + s2*vel[1]

        #both side rays enter goal
        if surface[1]/2 - goal_width/2 < y1 < surface[1]/2 + goal_width/2 and surface[1]/2 - goal_width/2 < y2 < surface[1]/2 + goal_width/2:
            for j in range(2):
                if t > tPZero:
                    new_pos[j] = final_pos[j]
                else:
                    new_pos[j] = pos[j] + dPt * dir[j]
                    new_vel[j] = vt * dir[j]

            arr = puck_mallet_collision(pos, vel, velocity(t), t, t_init)
            #x_p, x_m, v_p, v_m, dt_sum
            if len(arr) > 1:
                new_pos = arr[0]
                new_vel = arr[2]
                return update_puck(t-arr[4], new_pos, new_vel, t_init+arr[4])

            return new_pos, new_vel
        

    #Compute which wall it will hit if it keeps moving
    wallx = [puck_radius, surface[0] - puck_radius]
    wally = [puck_radius,surface[1] - puck_radius]
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

    if  s < dPZero:
        wall_val = s
        val, info, ier, msg = fsolve(pos_wall, x0=wall_val/v_norm, xtol=1e-4, full_output=True)
        thit = val[0]
        if ier != 1 and abs(pos_wall(val)) > 1e-4:
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
        arr = puck_mallet_collision(pos, vel, velocity(t), t, t_init)
        #x_p, x_m, v_p, v_m, dt_sum
        if len(arr) > 1:
            new_pos = arr[0]
            new_vel = arr[2]
            count_err += 1
            return update_puck(t-arr[4], new_pos, new_vel, t_init+arr[4])
        
        for j in range(2):
            if t > tPZero:
                new_pos[j] = final_pos[j]
            else:
                new_pos[j] = pos[j] + dPt * dir[j]
                new_vel[j] = vt * dir[j]

        return new_pos, new_vel
    

    #Check if it will hit the corner or not
    if wall_idx == 0 and  surface[1]/2 - goal_width/2 < vel[1] * s1 + pos[1] < surface[1]/2 + goal_width/2:
        A = [0,surface[1]/2 - goal_width/2]
        if vel[0] > 0:
            A[0] = surface[0]
        if (y1+y2)/2 > surface[1]/2:
            A[1] = surface[1]/2 + goal_width/2

        return corner_collision(A, pos, vel, t, final_pos, C, D, tPZero, v_norm, dPZero, dir, dPt, vt, t_init)

    #determine if mallet will hit before wall collision
    arr = puck_mallet_collision(pos, vel, velocity(t), thit_min, t_init)
    #x_p, x_m, v_p, v_m, dt_sum
    if len(arr) > 1:
        new_pos = arr[0]
        new_vel = arr[2]
        return update_puck(t-arr[4], new_pos, new_vel, t_init+arr[4])
    
    #Go to where it would hit the wall, set position and vel opposit of what they would be
    dPt_hit = dP(thit_min)
    vt_hit = velocity(thit_min)
    for j in range(2):
        if j == wall_idx:
            new_pos[j] = pos[j] + dPt_hit * dir[j]
            new_vel[j] = -res * vt_hit * dir[j]
        else:
            new_pos[j] = pos[j] + dPt_hit * dir[j]
            new_vel[j] = res * vt_hit * dir[j]

    #print("----")
    #print(pos)
    #print(vel)
    #print(new_pos)

    #s = (new_pos[0] - pos[0])/vel[0]
    #print(pos[1] + s*vel[1])


    return update_puck(t-thit_min, new_pos, new_vel, t_init+thit_min)

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


def ax_error(ax):

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

def ay_error(ay):
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

def solve_a():
    #ax = [x_f[0] / (C1[0]*16.7785) - C2[0]/C1[0]]
    #ay = [x_f[1] / (C1[1]*16.7785) - C2[1]/C1[1]]
    x0=x_f[0] * C7[0]/C1[0] - C2[0]/C1[0]
    ax = 0.5
    if x0 == 0:
        ax = [0]
    else:
        ax, info, ier, msg = fsolve(ax_error, x0, xtol=1e-4, full_output=True)
        if ier != 1 and abs(ax_error(ax)) > 1e-4:
            for n in range(2,11):
                ax, info, ier, msg = fsolve(ax_error, n*x0, xtol=1e-4, full_output=True)
                if ier == 1:
                    break
            if ier != 1 and abs(ax_error(ax)) > 1e-4:
                print("failed to converge ax")
                print(ax_error(ax))
                print(x_0)
                print(x_p)
                print(x_pp)
                print(x_f)
                print(C1)
                print(C2)
                print(C3)
                print(C4)
                print(C5)
                print(C6)
                print(C7)
                print(ax)
                print("--")
                print(ab)
                print(CE)
                print(A2CE)
                print(A3CE)
                print(A4CE)
    
    x0=x_f[1] * C7[1]/C1[1] - C2[1]/C1[1]
    ay = 0.5
    if x0 == 0:
        ay = [0]
    else:
        ay, info, ier, msg = fsolve(ay_error, x0, xtol=1e-4, full_output = True) #2*(abs(x_f[1]-x_0[1]))/math.sqrt(abs(C1[1]*2/R)), xtol=1e-4)
        if ier != 1 and abs(ay_error(ay)) > 1e-4:
            for n in range(2,11):
                ay, info, ier, msg = fsolve(ay_error, n*x0, xtol=1e-4, full_output=True)
                if ier == 1:
                    break
            if ier != 1 and abs(ay_error(ay)) > 1e-4:
                print("failed to converge ay")
                print(ay_error(ay))
                print(x_0)
                print(x_p)
                print(x_pp)
                print(x_f)
                print(C1)
                print(C2)
                print(C3)
                print(C4)
                print(C5)
                print(C6)
                print(C7)
                print(ay)
                print("--")
                print(ab)
                print(CE)
                print(A2CE)
                print(A3CE)
                print(A4CE)
    ax = np.float32(ax)
    ay = np.float32(ay)
    return [ax, ay]

def solve_C1p(x):
    if x <= 0:
        return 1 - x
    #return -x*16.7785 * (np.log((-x*0.836393)/(-x*0.836531-C2[k]*16.9975+C3[k]*345.371-C4[k]*7017.58))\
    #                     /(-20.319)-C2[k]/x) - bounds[1]
    return C2[k]/C7[k] - (x[0]/(ab[k][1]*C7[k])*\
                math.log((x[0]*CE[k][3])/(-x[0]*CE[k][1]+C2[k]*A2CE[k][1]+C3[k]*A3CE[k][1]+C4[k]*A4CE[k][1]))) - bounds[k][1]

def solve_C1n(x):
    if x[0] >= 0:
        return 1 + x[0]
    #return -x*16.7785 * (np.log((-x*0.836393)/(-x*0.836531-C2[k]*16.9975+C3[k]*345.371-C4[k]*7017.58))\
    #                     /(-20.319)-C2[k]/x) - bounds[0]
    return C2[k]/C7[k] - (x[0]/(ab[k][1]*C7[k])*\
                math.log((x[0]*CE[k][3])/(-x[0]*CE[k][1]+C2[k]*A2CE[k][1]+C3[k]*A3CE[k][1]+C4[k]*A4CE[k][1]))) - bounds[k][0]
    

def get_pos(t):
    #A2CE: A e^(at) + Be^(bt) + C
    #A2 = (16.7785+0.218934*math.exp(-1577.51*t)-16.9975*math.exp(-20.319*t))
    #A3 = (345.371*math.exp(-20.319*t)-345.371*math.exp(-1577.51*t))
    #A4 = (544825*math.exp(-1577.51*t)-7017.58*math.exp(-20.319*t))
    pos_ret = [0,0]
    for i in range(2):
        eat = math.exp(ab[i][0]*t)
        ebt = math.exp(ab[i][1]*t)
        A2 = A2CE[i][0] * eat + A2CE[i][1]*ebt + A2CE[i][2]
        A3 = A3CE[i][0] * eat + A3CE[i][1]*ebt
        A4 = A4CE[i][0] * eat + A4CE[i][1]*ebt
        f_t = f(t, i, eat, ebt)
        g_a1 = g(t-a[i], i)
        g_a2 = g(t-a2[i], i)
        pos_ret[i] = C1[i]*(f_t-2*g_a1+g_a2) + C2[i]*A2+C3[i]*A3+C4[i]*A4
    return pos_ret

def get_xp(t):
    xp = [0,0]

    for i in range(2):
        eat = ab[i][0] * math.exp(ab[i][0]*t)
        ebt = ab[i][1] * math.exp(ab[i][1]*t)

        A2 = A2CE[i][0] * eat + A2CE[i][1]*ebt
        A3 = A3CE[i][0] * eat + A3CE[i][1]*ebt
        A4 = A4CE[i][0] * eat + A4CE[i][1]*ebt

        f_t = CE[i][0] * eat + CE[i][1] * ebt + CE[i][2]
        g_a1 = 0
        g_a2 = 0
        if t > a[i]:
            eat = ab[i][0] * math.exp(ab[i][0] * (t-a[i]))
            ebt = ab[i][1] * math.exp(ab[i][1] * (t-a[i]))
            g_a1 = CE[i][0] * eat + CE[i][1] * ebt + CE[i][2]
            if t > a2[i]:
                eat = ab[i][0] * math.exp(ab[i][0] * (t-a2[i]))
                ebt = ab[i][1] * math.exp(ab[i][1] * (t-a2[i]))
                g_a2 = CE[i][0] * eat + CE[i][1] * ebt + CE[i][2]

        xp[i] = C1[i]*(f_t-2*g_a1+g_a2)+C2[i]*A2+C3[i]*A3+C4[i]*A4

    return xp

def get_xpp(t):
    xpp = [0,0]

    for i in range(2):
        eat = ab[i][0]**2 * math.exp(ab[i][0]*t)
        ebt = ab[i][1]**2 * math.exp(ab[i][1]*t)

        A2 = A2CE[i][0] * eat + A2CE[i][1]*ebt
        A3 = A3CE[i][0] * eat + A3CE[i][1]*ebt
        A4 = A4CE[i][0] * eat + A4CE[i][1]*ebt

        f_t = CE[i][0] * eat + CE[i][1] * ebt
        g_a1 = 0
        g_a2 = 0
        if t > a[i]:
            eat = ab[i][0]**2 * math.exp(ab[i][0] * (t-a[i]))
            ebt = ab[i][1]**2 * math.exp(ab[i][1] * (t-a[i]))
            g_a1 = CE[i][0] * eat + CE[i][1] * ebt
            if t > a2[i]:
                eat = ab[i][0]**2 * math.exp(ab[i][0] * (t-a2[i]))
                ebt = ab[i][1]**2 * math.exp(ab[i][1] * (t-a2[i]))
                g_a2 = CE[i][0] * eat + CE[i][1] * ebt

        xpp[i] = C1[i]*(f_t-2*g_a1+g_a2)+C2[i]*A2+C3[i]*A3+C4[i]*A4

    return xpp

def update_path():
    global marker
    global x_0
    global x_p
    global x_pp
    global x_f
    global C2
    global C3
    global C4
    global C1
    global a
    global a2
    global k
    global mallet_x_bounds
    global mallet_y_bounds

    marker = [pxl_bounds[0][0] + margin + random.random() * (pxl_bounds[0][1] - pxl_bounds[0][0] - 2*margin), pxl_bounds[1][0] + margin + random.random() * (pxl_bounds[1][1] - pxl_bounds[1][0] - 2*margin)]
    #marker = [569.5745894577943, 464.8470050107214]

    #print("NEW")
    x_0 = [mallet_pos[0] / plr, mallet_pos[1] / plr]
    x_p = get_xp(Ts)
    x_pp = get_xpp(Ts)
    x_f = [marker[0] / plr, marker[1] / plr]

    C2 = [C5[0]*x_pp[0]+C6[0]*x_p[0]+C7[0]*x_0[0], C5[1]*x_pp[1]+C6[1]*x_p[1]+C7[1]*x_0[1]]
    C3 = [C5[0]*x_p[0]+C6[0]*x_0[0],C5[1]*x_p[1]+C6[1]*x_0[1]]
    C4 = [C5[0]*x_0[0],C5[1]*x_0[1]]

    Vmin = [0,0]

    for j in range(2):
        k=j
        if x_p[j] > 0 and C2[j]/C7[j] > bounds[j][1]:
            val, info, ier, msg = fsolve(solve_C1p, x0=0.05, xtol=1e-4, full_output=True)
            if ier != 1:
                for n in range(1,11):
                    val, info, ier, msg = fsolve(solve_C1p, x0=0.05/(n*10), xtol=1e-4, full_output=True)
                    if ier == 1:
                        break

                if ier != 1:
                    print("C1p failed corvergence")
                    print(C1[j])
                    print(solve_C1n([C1[j]]))
                    print("---")
                    print(C2[j])
                    print(C3[j])
                    print(C4[j])
                    print(C7[j])
                    print("----")
                    print(CE[j][1])
                    print(CE[j][3])
                    print("---")
                    print(bounds[k][0])
                    print(A2CE[k][1])
                    print(A3CE[k][1])
                    print(A4CE[j][1])
                    print(ab[k][1])

            C1[j] = val[0]
            Vmin[j] = C1[j] * 2/pullyR

            #solve C1 > 0 so that x_over[j] = bounds[1]
        elif x_p[j] < 0 and C2[j]/C7[j] < bounds[j][0]:
            val, info, ier, msg = fsolve(solve_C1n, x0=-0.05, xtol=1e-4, full_output=True)
            if ier != 1:
                for n in range(1,11):
                    val, info, ier, msg = fsolve(solve_C1n, x0=-0.05/(n*10), xtol=1e-4, full_output=True)
                    if ier == 1:
                        break

                if ier != 1:
                    print("C1n failed corvergence")
                    print(C1[j])
                    print(solve_C1n([C1[j]]))
                    print("---")
                    print(C2[j])
                    print(C3[j])
                    print(C4[j])
                    print(C7[j])
                    print("----")
                    print(CE[j][1])
                    print(CE[j][3])
                    print("---")
                    print(bounds[k][0])
                    print(A2CE[k][1])
                    print(A3CE[k][1])
                    print(A4CE[j][1])
                    print(ab[k][1])

            C1[j] = val[0]
            #C2[k]/C7[k] - (x[0]/(ab[k][1]*C7[k])*\
             #   math.log((x[0]*CE[k][3])/(-x[0]*CE[k][1]+C2[k]*A2CE[k][1]+C3[k]*A3CE[k][1]+C4[k]*A4CE[k][1]))) - bounds[k][0]

            Vmin[j] = abs(C1[j]) * 2/pullyR
        # set magntiude of C1
        

    Vo = [14, 14] #[random.random() * 2*Vmax, random.random()*2*Vmax]
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

    mallet_x_bounds = [min(x_0[0], x_f[0], x_over[0])-mallet_radius, max(x_0[0], x_f[0], x_over[0])+mallet_radius]
    mallet_y_bounds = [min(x_0[1], x_f[1], x_over[1])-mallet_radius, max(x_0[1], x_f[1], x_over[1])+mallet_radius]

    a = solve_a()
    a = [a[0][0], a[1][0]]

    a2 = [2*a[0] - x_f[0]*C7[0]/C1[0]+C2[0]/C1[0], 2*a[1]-x_f[1]*C7[1]/C1[1]+C2[1]/C1[1]]


pxl_bounds = [[circle_radius_m, (screen_width - circle_radius_m)],[circle_radius_m, (screen_height - circle_radius_m)]]
marker = [103.77566320838407, 250.11977210934586] #[500, 100]

time = 0.0
time2 = 0.0
x_f = [marker[0] / plr, marker[1] / plr]
x_over = [0,0]
if (x_0[0] > x_f[0]):
    C1[0] *= -1
if (x_0[1] > x_f[1]):
    C1[1] *= -1

x_over = [0,0]
for i in range(2):
    x_over[i] = C2[i]/C7[i] - (C1[i]/(ab[i][1]*C7[i])*\
                math.log((C1[i]*CE[i][3])/(-C1[i]*CE[i][1]+C2[i]*A2CE[i][1]+C3[i]*A3CE[i][1]+C4[i]*A4CE[i][1])))

a = solve_a()
a = [a[0][0], a[1][0]]
a2 = [2*a[0] - x_f[0]*C7[0]/C1[0]+C2[0]/C1[0], 2*a[1]-x_f[1]*C7[1]/C1[1]+C2[1]/C1[1]]

mallet_x_bounds = [min(x_0[0], x_f[0], x_over[0])-mallet_radius, max(x_0[0], x_f[0], x_over[0])+mallet_radius]
mallet_y_bounds = [min(x_0[1], x_f[1], x_over[1])-mallet_radius, max(x_0[1], x_f[1], x_over[1])+mallet_radius]



action = 0
time = 0
N = 500
dts = Ts/N

puck_TP = [0,0]
puck_VTP = [0,0]
puck_T = [0,0]
puck_VT = [0,0]
markerP = [0,0]

x_0P = 0
x_pP = 0
x_ppP = 0


action = 0
timer = clock.tick(60) / 1000.0
while running:
    #timer = clock.tick(60) / 1000.0
    #print(timer)
    screen.fill(white)
    pygame.draw.circle(screen, red, marker, 5)
    pygame.draw.rect(screen, red, pygame.Rect(0, screen_height/2-goal_width*plr/2, 10, goal_width*plr))
    pygame.draw.rect(screen, red, pygame.Rect(screen_width-10, screen_height/2-goal_width*plr/2, 10, goal_width*plr))
    count_err = 0

    
    time += dts
    if time > Ts + dts/2:
        #print(time-dts)
        #print("A")
        #print(mallet_pos)
        #print(puck_pos)
        #print(puck_vel)

        x_0P = x_0
        x_pP = x_p
        x_ppP = x_pp

        markerP = marker

        update_path()
        puck_TP = puck_T
        puck_VTP = puck_VT
        
        puck_T = puck_pos
        puck_VT = puck_vel
        time = dts
        #print("update")

    if puck_pos[0] < 0 or puck_pos[0] > surface[0]: #or (puck_vel[0] == 0 and puck_vel[1] == 0):
        puck_pos = [puck_radius + random.random() * (surface[0] - 2*puck_radius), puck_radius + random.random() * (surface[1] - 2*puck_radius)]
        puck_vel = [(random.random()-0.5) * 4, (random.random() - 0.5) * 4]
        while np.sqrt((puck_pos[0] - mallet_pos[0]/plr)**2 + (puck_pos[1] - mallet_pos[1]/plr)**2) < puck_radius + mallet_radius + 1e-4:
            puck_pos = [puck_radius + random.random() * (surface[0] - 2*puck_radius), puck_radius + random.random() * (surface[1] - 2*puck_radius)]
            puck_vel = [(random.random()-0.5) * 4, (random.random() - 0.5) * 4]
        #print("A")
        #print(puck_pos)
        #print(puck_vel)

    if puck_pos[1] < puck_radius or puck_pos[1] > surface[1] - puck_radius or np.sqrt((puck_pos[1] - mallet_pos[1]/plr)**2 + (puck_pos[0]-mallet_pos[0]/plr)**2) - mallet_radius-puck_radius < -1e-2 \
        or ((puck_pos[0] < puck_radius or puck_pos[0] > surface[0] - puck_radius) and (puck_pos[1] < surface[1]/2-goal_width/2 or puck_pos[1] > surface[1]/2+goal_width/2)):
        print("glitch")
        print(puck_pos[1])
        print(np.sqrt((puck_pos[1] - mallet_pos[1]/plr)**2 + (puck_pos[0]-mallet_pos[0]/plr)**2) - mallet_radius-puck_radius)
        print(puck_TP)
        print(puck_VTP)
        print(x_0P)
        print(x_pP)
        print(x_ppP)
        print(markerP)
        print(marker)
        print("---")
        print(puck_pos)
        print(pos)
        puck_pos = [puck_radius + random.random() * (surface[0] - 2*puck_radius), puck_radius + random.random() * (surface[1] - 2*puck_radius)]
        puck_vel = [(random.random()-0.5) * 4, (random.random() - 0.5) * 4]
        while np.sqrt((puck_pos[0] - mallet_pos[0]/plr)**2 + (puck_pos[1] - mallet_pos[1]/plr)**2) < puck_radius + mallet_radius + 1e-4:
            puck_pos = [puck_radius + random.random() * (surface[0] - 2*puck_radius), puck_radius + random.random() * (surface[1] - 2*puck_radius)]
            puck_vel = [(random.random()-0.5) * 4, (random.random() - 0.5) * 4]
        

    
    
    puck_pos, puck_vel = update_puck(dts, puck_pos, puck_vel, time-dts)
    pos = get_pos(time)
    mallet_pos = [pos[0] * plr, pos[1] * plr]

    circle_pos[0] = puck_pos[0] * plr
    circle_pos[1] = puck_pos[1] * plr

    for j in range(2):
        if mallet_pos[j] < circle_radius_m:
            mallet_pos[j] = circle_radius_m
            print("COLLISION")
    if mallet_pos[0] > screen_width - circle_radius_m:
        mallet_pos[0] = screen_width - circle_radius_m
        print("COLLISION")
    if mallet_pos[1] > screen_height - circle_radius_m:
        mallet_pos[1] = screen_height - circle_radius_m
        print("COLLISION")

    action += 1
    if action % int(3600/Ts) == 0:
        print(action * Ts / 3600)
        timer = clock.tick(60) / 1000.0
        print(timer)

        #screen.fill(white)
        #pygame.draw.circle(screen, red, marker, 5)
        #pygame.draw.rect(screen, red, pygame.Rect(0, screen_height/2-goal_width*plr/2, 10, goal_width*plr))
        #pygame.draw.rect(screen, red, pygame.Rect(screen_width-10, screen_height/2-goal_width*plr/2, 10, goal_width*plr))
        #pygame.draw.circle(screen, black, (int(round(mallet_pos[0])), int(round(mallet_pos[1]))), circle_radius_m)
        #pygame.draw.circle(screen, (0, 0, 255), (round(circle_pos[0]), round(circle_pos[1])), circle_radius)
        #pygame.display.flip()


    # Draw the circle
    #print(mallet_pos)
    pygame.draw.circle(screen, black, (int(round(mallet_pos[0])), int(round(mallet_pos[1]))), circle_radius_m)
    pygame.draw.circle(screen, (0, 0, 255), (round(circle_pos[0]), round(circle_pos[1])), circle_radius)

    # Update the display
    pygame.display.flip()

    #for j in range(10000000):
    #    pass


# Quit pygame
pygame.quit()