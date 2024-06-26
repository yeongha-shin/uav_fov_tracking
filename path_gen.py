
# #!/usr/bin/env python3
# import glob
# import pandas as pd
# import os
# import numpy as np
# from scipy.optimize import minimize
# from autograd import grad
# import autograd.numpy as anp
# import matplotlib.pyplot as plt
# from tool.utils import CurveFactory
# from tool.utils import Grid, Map, SearchFactory
# from scipy.interpolate import RectBivariateSpline
# from autograd import elementwise_grad as egrad
# from scipy.ndimage import map_coordinates

# # Parameters
# od_min = 5 
# od_max = 8
# rho = 0.8
# m = 2
# order = 3
# nc = m + order

# dt = 1       
# vm = 1.0 
# am = 0.5 
# vmp = 0.5 
# amp = 0.5 

# order = 3 
# d_thr = 1

# fov_angle = 100
# fov_depth = 3.5

# # Load ESDF
# esdf_path = "esdf.csv"
# esdf_pd = pd.read_csv(esdf_path)
# esdf = esdf_pd.to_numpy()
# esdf = anp.array(esdf, dtype=anp.float64)
# x = anp.arange(0, esdf.shape[1], 1, dtype=anp.float64)
# y = anp.arange(0, esdf.shape[0], 1, dtype=anp.float64)
# X, Y = anp.meshgrid(x, y)
# # ESDF 보간 함수 생성
# def esdf_function(x, y):
#     coords = anp.array([[y], [x]], dtype=anp.float64)
#     return map_coordinates(esdf, coords, order=1, mode='nearest')[0]


# def penalty(x):
#     return anp.maximum(0, x**3)

# def current_pose(Q):
#     p = []
#     for i in range(nc-2):
#         temp1 = Q[i]
#         temp2 = Q[i+1]
#         temp3 = Q[i+2]
#         combined = temp1 + 4*temp2 + temp3
#         p.append(combined / 6)
#     return anp.array(p)

# def fov(p, c):
#     cent = []
#     radi = []
#     for i in range(nc-2):
#         drn_pose = p[i][0:2]
#         tar_pose = c[i][0:2]
#         for j in range(m):
#             lamb = (j+1) / m
#             center = lamb * (tar_pose - drn_pose) + drn_pose
#             radius = anp.linalg.norm(drn_pose - tar_pose) * lamb * rho
#             cent.append(center)
#             radi.append(radius)
#     return anp.array(cent), anp.array(radi)

# def obstacle_distance(x, y):

#     E = esdf_function(x, y)
#     return E

# def dist_cost(p, c):
#     dist_cost = 0
#     for i in range(nc-2):
#         d = anp.linalg.norm(p[i][0:2] - c[i][0:2])
#         dist_cost += penalty(od_min**2 - d**2) + penalty(d**2 - od_max**2)
#     return dist_cost

# def ang_cost(p, c):
#     ang_cost = 0
#     for i in range(nc-2):
#         best_ang = anp.arctan2((c[i][1] - p[i][1]), (c[i][0] - p[i][0]))
#         ang_cost += (p[i][2] - best_ang)**2
#     return ang_cost

# def obs_cost(p, c):
#     obs_cost = 0
#     cent, radi = fov(p, c)
#     for i in range((nc-2) * m):
#         cent_x = cent[i, 0]
#         cent_y = cent[i, 1]
#         E = obstacle_distance(cent_x, cent_y)
#         obs_cost += penalty(radi[i]**2 - E**2)
#     return obs_cost

# def dyn_cost(Q):
#     v = []
#     a = []
#     j = []
    
#     for i in range(nc-1):
#         v.append((Q[i+1] - Q[i]) / dt)
             
#     for i in range(nc-2):
#         a.append((v[i+1] - v[i]) / dt)

#     for i in range(nc-3):
#         j.append((a[i+1] - a[i]) / dt)
    
#     v = anp.array(v)
#     a = anp.array(a)
#     j = anp.array(j)
    
#     Jf = 0
#     Jfp = 0
#     Js = 0
#     Jsp = 0
    
#     for i in range(nc-1):
#         Jf += penalty(v[i][0]**2 + v[i][1]**2 - vm**2)
#         Jfp += penalty(v[i][2]**2 - vmp**2)
    
#     for i in range(nc-2):
#         Jf += penalty(a[i][0]**2 + a[i][1]**2 - am**2)
#         Jfp += penalty(a[i][2]**2 - amp**2)
    
#     for i in range(nc-3):
#         Js += j[i][0]**2 + j[i][1]**2
#         Jsp += j[i][2]**2
    
#     return Jf + Jfp + Js + Jsp

# def collision_cost(Q):
#     Jc = 0
#     for i in range(nc-2):
#         E = obstacle_distance(Q[i, 0], Q[i, 1])
#         Jc += penalty(d_thr**2 - E**2) * E
#     return Jc

# def total_cost(Q_flat, c):
#     Q = anp.reshape(Q_flat, (nc, 3))
#     p = current_pose(Q)
#     dist = dist_cost(p, c)
#     ang = ang_cost(p, c)
#     obs = obs_cost(p, c)
#     dyn = dyn_cost(Q)
#     col = collision_cost(Q)
#     cost = dist + ang + dyn + obs + col
#     return cost

# def main():
#     current_directory = os.getcwd()
#     folder_path = os.path.join(current_directory, 'local')
#     target_path = os.path.join(current_directory, 'global', 'global_path.csv')
#     file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
#     file_paths.sort()
    
#     start = (5, 25)
#     goal = (45, 5)
#     env = Grid(51, 31)
#     current_position = np.array(start, dtype=float)
#     goal_position = np.array(goal, dtype=float)
#     data_target = pd.read_csv(target_path, skiprows=1)
#     c = data_target.to_numpy()
#     c = anp.array(c, dtype=anp.float64)
#     file_path_to_index = {path: i for i, path in enumerate(file_paths)}
#     colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#     search_factory = SearchFactory()
#     planner = search_factory("a_star", start=tuple(current_position), goal=tuple(goal_position), env=env)
#     for i in range(len(c)):
#         data = c[i:i+nc]
#         initial_guess = anp.array(data.flatten(), dtype=anp.float64)

#         c_segment = c[i+nc+5:i+nc+8]  # Adjust the slicing as per your specific needs

#         grad_total_cost = grad(lambda Q_flat: total_cost(Q_flat, c_segment))
#         result = minimize(lambda Q_flat: total_cost(Q_flat, c_segment), initial_guess, method='L-BFGS-B', jac=grad_total_cost)
#         q_optimized = result.x.reshape(-1, 3)
#         print(f"Optimization result for {i}: {result}")
#         x_coords = q_optimized[:, 0]
#         y_coords = q_optimized[:, 1]

#         # Plotting
#         plt.scatter(x_coords, y_coords, color=colors[1], label=f'optimized {i}')
#         plt.scatter(data[:, 0], data[:, 1], color=colors[0], marker='x', label=f'Initial control point {i}')
#         plt.scatter(c_segment[:, 0], c_segment[:, 1], color=colors[2], marker='o', label=f'Target {i}')
#         plt.xlabel('X coord')
#         plt.ylabel('Y coord')
#         fig = planner.plot.plotEnv("Map")
#         plt.title('X-Y graph')
#         plt.grid(True)
#         plt.legend()
#         plt.show()

# if __name__ == '__main__':
#     main()

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
od_min = 5 
od_max = 8
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
d_thr = 1

fov_angle = 100
fov_depth = 3.5

# Load ESDF
esdf_path = "esdf.csv"
esdf_pd = pd.read_csv(esdf_path)
esdf = esdf_pd.to_numpy()
esdf = anp.array(esdf, dtype=anp.float64)

# ESDF 함수 정의
def esdf_function(x, y):
    x = anp.clip(x, 0, esdf.shape[1] - 1)
    y = anp.clip(y, 0, esdf.shape[0] - 1)
    x = anp.floor(x).astype(int)
    y = anp.floor(y).astype(int)
    return esdf[y, x]

def penalty(x):
    return anp.maximum(0, x**3)

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
        ang_cost += (p[i][2] - best_ang)**2
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
    # cost = 50*dist + 0.5*ang + 0.5*dyn + 50*obs + 50*col
    cost = 5*dist + 0*ang + 0*dyn + 0*obs + 0*col
    return cost

def main():
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, 'output', 'local')
    target_path = os.path.join(current_directory, 'output', 'global', 'global_path.csv')
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
    first_path, first_fig = planner.run()

    # for i in range(len(c)):
    for i in range(len(c) - nc - 6):
        data = c[i:i+nc]
        initial_guess = anp.array(data.flatten(), dtype=anp.float64)

        c_segment = c[i+nc+1:i+nc+4]  # Adjust the slicing as per your specific needs

        grad_total_cost = grad(lambda Q_flat: total_cost(Q_flat, c_segment))
        result = minimize(lambda Q_flat: total_cost(Q_flat, c_segment), initial_guess, method='L-BFGS-B', jac=grad_total_cost)
        q_optimized = result.x.reshape(-1, 3)
        print(f"Optimization result for {i}: {result}")
        x_coords = q_optimized[:, 0]
        y_coords = q_optimized[:, 1]

        # Plotting

        # FOV plot

        # path update
        new_start = (x_coords[0], y_coords[0])
        new_goal = (c_segment[0, 0], c_segment[0, 1])

        print("new start", new_start)
        print("new goal", new_goal)

        planner = search_factory("a_star", start=tuple(new_start), goal=tuple(new_goal), env=env)
        global_path, global_fig = planner.run()  # path 계획 메서드로 가정

        plt.scatter(x_coords, y_coords, color=colors[1], label=f'optimized {i}')
        plt.scatter(data[:, 0], data[:, 1], color=colors[0], marker='x', label=f'Initial control point {i}')
        plt.scatter(c_segment[:, 0], c_segment[:, 1], color=colors[2], marker='o', label=f'Target {i}')

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

