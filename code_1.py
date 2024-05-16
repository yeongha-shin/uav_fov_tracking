#!/usr/bin/env python3

import glob
import pandas as pd
import os
import numpy as np
from scipy.optimize import minimize
from autograd import grad
import autograd.numpy as anp
import matplotlib.pyplot as plt
from tool.utils import CurveFactory
from tool.utils import Grid, Map, SearchFactory

# Parameters
od_min = 2 
od_max = 15
rho = 0.8
m = 2
order = 3
nc = m + order

dt = 1       
vm = 1.0 
am = 0.5 
vmp = 0.5 
amp = 0.5 

order = 3 
d_thr = 3

fov_angle = 100
fov_depth = 3.5

# Load ESDF
esdf_path = "esdf.csv"
esdf_pd = pd.read_csv(esdf_path)
esdf = esdf_pd.to_numpy().T
esdf = anp.array(esdf, dtype=anp.float64)

def extract_value(v):
    if isinstance(v, np.float64) and hasattr(v, 'item'):
        return v.item()
    elif hasattr(v, '_value'):
        return float(v._value)
    else:
        return float(v)

# ESDF 함수 정의
def esdf_function(x, y):
    
    x_clip = anp.clip(x, 0, esdf.shape[1] - 1)
    y_clip = anp.clip(y, 0, esdf.shape[0] - 1)

    if isinstance(x_clip, anp.ndarray):
        x_clip = anp.array([extract_value(v) for v in x_clip])
    else:
        x_clip = extract_value(x_clip)

    if isinstance(y_clip, anp.ndarray):
        y_clip = anp.array([extract_value(v) for v in y_clip])
    else:
        y_clip = extract_value(y_clip)



    x_floor = anp.floor(x).astype(int)
    y_floor = anp.floor(y).astype(int)
    x_ceil = anp.ceil(x).astype(int)
    y_ceil = anp.ceil(y).astype(int)

    x_floor = np.clip(x_floor, 0, esdf.shape[1] - 1)
    y_floor = np.clip(y_floor, 0, esdf.shape[0] - 1)
    x_ceil = np.clip(x_ceil, 0, esdf.shape[1] - 1)
    y_ceil = np.clip(y_ceil, 0, esdf.shape[0] - 1)

    E_floor = esdf[x_floor, y_floor]
    E_ceil = esdf[x_ceil, y_ceil]

    if E_floor != E_ceil:
        E = E_floor + (E_ceil-E_floor)*np.linalg.norm((x_clip-x_floor,y_clip-y_floor))/(np.linalg.norm((x_ceil-x_floor,y_ceil-y_floor)))

    else: 
        E = E_floor

    return  E

def penalty(x):
    return anp.maximum(0, x**3)

def target_pose_predict(c):         # Output : 

    p_0 = c[0][0:2]
    p_1 = c[1][0:2]
    # psi_0 = c[0][2]/180*anp.pi           # angle in [rad]
    # psi_1 = c[1][2]/180*anp.pi           # angle in [rad]
    dist = anp.linalg.norm(p_0-p_1)      
    r = (c[1][2] - c[0][2])    # indicates yaw rate (for each time step)
    
    predicted_pose = np.zeros((nc-2,3))
    predicted_pose[0] = c[1]
    for i in range(nc-3):
        predicted_pose[i+1][0] = predicted_pose[i][0] + dist * np.cos(predicted_pose[i][2]/180*np.pi)
        predicted_pose[i+1][1] = predicted_pose[i][1] + dist * np.sin(predicted_pose[i][2]/180*np.pi)
        predicted_angle = predicted_pose[i][2] 
        
        if predicted_angle > 180: 
            predicted_angle -= 360
        elif predicted_angle <-180:
            predicted_angle += 360

        predicted_pose[i+1][2] = (predicted_angle)

    return predicted_pose

def current_pose(Q):
    p = []
    for i in range(nc-2):
        temp1 = Q[i]
        temp2 = Q[i+1]
        temp3 = Q[i+2]
        combined = temp1 + 4*temp2 + temp3
        p.append(combined / 6)
    return anp.array(p)

def fov(p, c):
    cent = []
    radi = []
    for i in range(nc-2):
        drn_pose = p[i][0:2]
        tar_pose = c[i][0:2]
        for j in range(m):
            lamb = (j+1) / m
            center = lamb * (tar_pose - drn_pose) + drn_pose
            radius = anp.linalg.norm(drn_pose - tar_pose) * lamb * rho
            cent.append(center)
            radi.append(radius)
    return anp.array(cent), anp.array(radi)

def obstacle_distance(x, y):
    E = esdf_function(x, y)

    return E

def dist_cost(p, c):
    dist_cost = 0
    for i in range(nc-2):
        d = anp.linalg.norm(p[i][0:2] - c[i][0:2])
        dist_cost += penalty(od_min**2 - d**2) + penalty(d**2 - od_max**2)
    return dist_cost

def ang_cost(p, c):
    ang_cost = 0
    for i in range(nc-2):
        best_ang = anp.arctan2((c[i][1] - p[i][1]), (c[i][0] - p[i][0]))
        ang_cost += ((p[i][2])*3.14/180 - best_ang)**2
    return ang_cost

def obs_cost(p, c):
    obs_cost = 0
    cent, radi = fov(p, c)
    for i in range((nc-2) * m):
        cent_x = cent[i, 0]
        cent_y = cent[i, 1]
        E = obstacle_distance(cent_x, cent_y)
        obs_cost += penalty(radi[i]**2 - E**2)
    return obs_cost

def dyn_cost(Q):
    v = []
    a = []
    j = []
    
    for i in range(nc-1):
        v.append((Q[i+1] - Q[i]) / dt)
             
    for i in range(nc-2):
        a.append((v[i+1] - v[i]) / dt)

    for i in range(nc-3):
        j.append((a[i+1] - a[i]) / dt)
    
    v = anp.array(v)
    a = anp.array(a)
    j = anp.array(j)
    
    Jf = 0
    Jfp = 0
    Js = 0
    Jsp = 0
    
    for i in range(nc-1):
        Jf += penalty(v[i][0]**2 + v[i][1]**2 - vm**2)
        Jfp += penalty(v[i][2]**2 - vmp**2)
    
    for i in range(nc-2):
        Jf += penalty(a[i][0]**2 + a[i][1]**2 - am**2)
        Jfp += penalty(a[i][2]**2 - amp**2)
    
    for i in range(nc-3):
        Js += j[i][0]**2 + j[i][1]**2
        Jsp += j[i][2]**2
    
    return Jf + Jfp + Js + Jsp

def collision_cost(Q):
    Jc = 0
    for i in range(nc-2):
        E = obstacle_distance(Q[i, 0], Q[i, 1])
        Jc += penalty(d_thr**2 - E**2) * E
    return Jc

def total_cost(Q_flat, c):
    Q = anp.reshape(Q_flat, (nc, 3))
    p = current_pose(Q)
    dist = dist_cost(p, c)
    ang = ang_cost(p, c)
    obs = obs_cost(p, c)
    dyn = dyn_cost(Q)
    col = collision_cost(Q)
    cost = 0.5*dist + 0.5*ang + 0.5*dyn + 0.5*obs + 0.5*col
    # cost = 0.0 * dist + 0.0 * ang + 0.0 * dyn + 0.0 * obs + 0.0 * col
    return cost


def point_in_triangle(pt, v1, v2, v3):

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def draw_fov(x_coords, y_coords, psi_coords, c_segment, iteration, length=1.0, color='r', fov_angle=80, fov_depth=10):
    fail_cases = []
    success_cases = []

    for idx, (x, y, psi) in enumerate(zip(x_coords, y_coords, psi_coords)):

        psi_radian = psi * (np.pi / 180)
        dx = length * np.cos(psi_radian)
        dy = length * np.sin(psi_radian)

        plt.arrow(x, y, dx, dy, head_width=0.5, head_length=0.5, fc=color, ec=color)

        left_angle = psi_radian + np.radians(fov_angle / 2)
        right_angle = psi_radian - np.radians(fov_angle / 2)

        left_dx = fov_depth * np.cos(left_angle)
        left_dy = fov_depth * np.sin(left_angle)

        right_dx = fov_depth * np.cos(right_angle)
        right_dy = fov_depth * np.sin(right_angle)

        fov_x = [x, x + left_dx, x + right_dx, x]
        fov_y = [y, y + left_dy, y + right_dy, y]

        # plt.fill(fov_x, fov_y, color='orange', alpha=0.3)

        v1 = (x, y)
        v2 = (x + left_dx, y + left_dy)
        v3 = (x + right_dx, y + right_dy)

        # for cx, cy in zip(c_segment[:, 0], c_segment[:, 1]):

        if idx < len(c_segment):
            cx, cy = c_segment[idx, 0], c_segment[idx, 1]
            if point_in_triangle((cx, cy), v1, v2, v3):
                # plt.scatter(cx, cy, color='gray', marker='o')
                plt.fill(fov_x, fov_y, color='blue', alpha=0.3)
                success_cases.append(idx)
            else:
                # plt.scatter(cx, cy, color='red', marker='o')
                plt.fill(fov_x, fov_y, color='orange', alpha=0.3)
                fail_cases.append(idx)

    with open('./output/test_result/log.txt', 'a') as log_file:
        log_file.write(f"Iteration: {iteration}\n")
        log_file.write(f"Fail case (red): {fail_cases}\n")
        log_file.write(f"Success case (blue): {success_cases}\n")
        log_file.write("\n")

def main():
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, 'local')
    target_path = os.path.join(current_directory, 'global', 'global_path.csv')
    #
    # folder_path = os.path.join(current_directory, 'output', 'local')
    # target_path = os.path.join(current_directory, 'output', 'global', 'global_path.csv')

    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
    file_paths.sort()
    
    start = (5, 25)
    goal = (45, 5)
    env = Grid(51, 31)
    current_position = np.array(start, dtype=float)
    goal_position = np.array(goal, dtype=float)
    data_target = pd.read_csv(target_path, skiprows=1)
    c = data_target.to_numpy()
    c = anp.array(c, dtype=anp.float64)
    file_path_to_index = {path: i for i, path in enumerate(file_paths)}
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    search_factory = SearchFactory()
    planner = search_factory("a_star", start=tuple(current_position), goal=tuple(goal_position), env=env)
    
    
    for i in range(len(c)):
        data = c[i:i+nc]
        p_prev = current_pose(data)

        initial_guess = anp.array(data.flatten(), dtype=anp.float64)

        c_segment = target_pose_predict(c[i+nc-1:i+1+nc])  # Adjust the slicing as per your specific needs

        grad_total_cost = grad(lambda Q_flat: total_cost(Q_flat, c_segment))
        result = minimize(lambda Q_flat: total_cost(Q_flat, c_segment), initial_guess, method='L-BFGS-B', jac=grad_total_cost)
        q_optimized = result.x.reshape(-1, 3)
        p_next = current_pose(q_optimized)

        print(f"Optimization result for {i}: {result}")
        x_coords = q_optimized[:, 0]
        y_coords = q_optimized[:, 1]
        psi_coords = q_optimized[:, 2]

        planner = search_factory("a_star", start=tuple((x_coords[0], y_coords[0])), goal=tuple((c_segment[0,0], c_segment[0,1])), env=env)
        # planner.run()
        #

        # print("start point", x_coords[0], y_coords[0])
        # print("goal point", c_segment[0,0], c_segment[0,1])
        
        # update_path = planner.run()  # path 계획 메서드로 가정
        # print("path", update_path)

        # Plotting
        # plt.scatter(x_coords, y_coords, color=colors[1], label=f'optimized {i}')
        # plt.scatter(data[:, 0], data[:, 1], color=colors[0], marker='x', label=f'Initial control point {i}')
        plt.scatter(c_segment[:, 0], c_segment[:, 1], color=colors[2], marker='o', label=f'Target {i}')
        plt.scatter(p_prev[:, 0], p_prev[:, 1], color=colors[3], marker='x', label=f'P_prev {i}')
        plt.scatter(p_next[:, 0], p_next[:, 1], color=colors[4], marker='o', label=f'P_next {i}')

        draw_fov(x_coords, y_coords, psi_coords, c_segment, i)

        plt.xlabel('X coord')
        plt.ylabel('Y coord')
        fig = planner.plot.plotEnv("Map")
        plt.title('X-Y graph')
        plt.grid(True)
        plt.legend()
        plt.show()

        # path update
        

if __name__ == '__main__':
    main()