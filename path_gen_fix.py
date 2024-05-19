
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
order = 3


dt = 1       
vm = 1.0 
am = 0.5 
vmp = 0.5 
amp = 0.5 
global nc
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



    x_floor = anp.floor(x_clip).astype(int)
    y_floor = anp.floor(y_clip).astype(int)
    x_ceil = anp.ceil(x_clip).astype(int)
    y_ceil = anp.ceil(y_clip).astype(int)

    E_floor = esdf[x_floor, y_floor]
    E_ceil = esdf[x_ceil, y_ceil]

    if E_floor != E_ceil:
        E = E_floor + (E_ceil-E_floor)*np.linalg.norm((x_clip-x_floor,y_clip-y_floor))/(np.linalg.norm((x_ceil-x_floor,y_ceil-y_floor)))

    else: 
        E = E_floor

    return  E

def penalty(x):
    return anp.maximum(0, x**3)

# def target_pose_predict(c):         # Output : 

#     p_0 = c[0][0:2]
#     p_1 = c[1][0:2]
#     # psi_0 = c[0][2]/180*anp.pi           # angle in [rad]
#     # psi_1 = c[1][2]/180*anp.pi           # angle in [rad]
#     dist = anp.linalg.norm(p_0-p_1)      
#     r = (c[1][2] - c[0][2])    # indicates yaw rate (for each time step)
    
#     predicted_pose = np.zeros((nc-2,3))
#     predicted_pose[0] = c[1]
#     for i in range(nc-3):
#         predicted_pose[i+1][0] = predicted_pose[i][0] + dist * np.cos(predicted_pose[i][2]/180*np.pi)
#         predicted_pose[i+1][1] = predicted_pose[i][1] + dist * np.sin(predicted_pose[i][2]/180*np.pi)
#         predicted_angle = predicted_pose[i][2] 
        
#         if predicted_angle > 180: 
#             predicted_angle -= 360
#         elif predicted_angle <-180:
#             predicted_angle += 360

#         predicted_pose[i+1][2] = (predicted_angle)

#     return predicted_pose

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
    return cost



def main():
    current_directory = os.getcwd()
    # folder_path = os.path.join(current_directory, 'local')
    target_path = os.path.join(current_directory, 'global', 'global_path.csv')
    # file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
    # file_paths.sort()
    
    start = (5, 25)

    env = Grid(51, 31)

    target_data = pd.read_csv(target_path, skiprows=1)
    target_path = target_data.to_numpy()
    target_path = anp.array(target_path, dtype=anp.float64)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    search_factory = SearchFactory()
    planner = search_factory("a_star", start=tuple(start), goal=tuple((target_path[0][0],target_path[0][1])), env=env)
    
    # first turn
    follower_path, first_fig = planner.run()

    follower_path = anp.array(follower_path)
    follower_path = np.hstack((follower_path, np.zeros((len(follower_path), 1))))


    for i in range(len(follower_path) - 1):
        x_diff = follower_path[i + 1, 0] - follower_path[i, 0]
        y_diff = follower_path[i + 1, 1] - follower_path[i, 1]
        follower_path[i, 2] = np.arctan2(y_diff, x_diff)

    follower_path = anp.array(follower_path)
    global nc

    nc=len(follower_path)-2

    global m
    m = nc- order
    
    
    for i in range(len(target_path)-3):

        p_prev = current_pose(follower_path)
        target_path_segment = target_path[i:i+nc]  # Adjust the slicing as per your specific needs
        
        initial_guess = anp.array(follower_path[1:-1].flatten(), dtype=anp.float64)
        grad_total_cost = grad(lambda Q_flat: total_cost(Q_flat, target_path_segment))
        result = minimize(lambda Q_flat: total_cost(Q_flat, target_path_segment), initial_guess, method='L-BFGS-B', jac=grad_total_cost)

        q_optimized = result.x.reshape(-1, 3)
        updated_follower_path = anp.vstack((follower_path[0,:].reshape(1,3),q_optimized,follower_path[-1,:].reshape(1,3)))

        print(len(follower_path),len(q_optimized),len(updated_follower_path))
        p_next = current_pose(updated_follower_path)

        print(f"Optimization result for {i}: {result}")
        x_coords = updated_follower_path[:, 0]
        y_coords = updated_follower_path[:, 1]

        new_start = (int(x_coords[1]), int(y_coords[1]))
        new_goal = (int(target_path_segment[1, 0]), int(target_path_segment[1, 1]))

        print("new start", new_start)
        print("new goal", new_goal)

        planner = search_factory("a_star", start=tuple(new_start), goal=tuple(new_goal), env=env)
        # follower_path, update_fig = planner.run()  # path 계획 메서드로 가정


        # for i in range(len(follower_path) - 1):
        #     x_diff = follower_path[i + 1, 0] - follower_path[i, 0]
        #     y_diff = follower_path[i + 1, 1] - follower_path[i, 1]
        #     follower_path[i, 2] = np.arctan2(y_diff, x_diff)


        # Plotting
        # plt.scatter(x_coords, y_coords, color=colors[1], label=f'optimized {i}')
        # plt.scatter(data[:, 0], data[:, 1], color=colors[0], marker='x', label=f'Initial control point {i}')
        
        plt.scatter(target_path_segment[:, 0], target_path_segment[:, 1], color=colors[2], marker='o', label=f'Target {i}')
        plt.scatter(p_prev[:, 0], p_prev[:, 1], color=colors[3], marker='x', label=f'P_prev {i}')
        plt.scatter(p_next[:, 0], p_next[:, 1], color=colors[4], marker='o', label=f'P_next {i}')
        
        plt.xlabel('X coord')
        plt.ylabel('Y coord')
        fig = planner.plot.plotEnv("Map")
        plt.title('X-Y graph')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()

# #  할일: 1. control point에 현재점 추가, p계산할때는 사용, 그러나 최적화에서는 제외
# # 2. global 새로 뽑기(csv)
# # 3. 매번 A*적용 가능하게 바꾸기


#!/usr/bin/env python3
# import os
# import glob
# import numpy as np
# import pandas as pd
# from scipy.optimize import minimize
# from autograd import grad
# import autograd.numpy as anp
# import matplotlib.pyplot as plt
# from tool.utils import CurveFactory, Grid, Map, SearchFactory

# class PathGenerator:
#     def __init__(self, esdf_path, target_path):
#         self.esdf_path = esdf_path
#         self.target_path = target_path
#         self.nc = None
#         self.esdf = self.load_esdf()
#         self.init_parameters()
    
#     def load_esdf(self):
#         esdf_pd = pd.read_csv(self.esdf_path)
#         esdf = esdf_pd.to_numpy().T
#         return anp.array(esdf, dtype=anp.float64)
    
#     def init_parameters(self):
#         self.od_min = 2
#         self.od_max = 15
#         self.rho = 0.8
#         self.order = 3
#         self.dt = 1
#         self.vm = 1.0
#         self.am = 0.5
#         self.vmp = 0.5
#         self.amp = 0.5
#         self.d_thr = 3
#         self.fov_angle = 100
#         self.fov_depth = 3.5
#         self.m = self.nc - self.order if self.nc else None
    
#     def extract_value(self, v):
#         if isinstance(v, np.float64) and hasattr(v, 'item'):
#             return v.item()
#         elif hasattr(v, '_value'):
#             return float(v._value)
#         else:
#             return float(v)

#     def esdf_function(self, x, y):
#         x_clip = anp.clip(x, 0, self.esdf.shape[1] - 1)
#         y_clip = anp.clip(y, 0, self.esdf.shape[0] - 1)

#         x_clip = anp.array([self.extract_value(v) for v in x_clip]) if isinstance(x_clip, anp.ndarray) else self.extract_value(x_clip)
#         y_clip = anp.array([self.extract_value(v) for v in y_clip]) if isinstance(y_clip, anp.ndarray) else self.extract_value(y_clip)

#         x_floor = anp.floor(x_clip).astype(int)
#         y_floor = anp.floor(y_clip).astype(int)
#         x_ceil = anp.ceil(x_clip).astype(int)
#         y_ceil = anp.ceil(y_clip).astype(int)

#         E_floor = self.esdf[x_floor, y_floor]
#         E_ceil = self.esdf[x_ceil, y_ceil]

#         if E_floor != E_ceil:
#             E = E_floor + (E_ceil - E_floor) * np.linalg.norm((x_clip - x_floor, y_clip - y_floor)) / np.linalg.norm((x_ceil - x_floor, y_ceil - y_floor))
#         else:
#             E = E_floor

#         return E
    

#     def penalty(self, x):
#         return anp.maximum(0, x**3)
    
#     def current_pose(self, Q):
#         p = []
#         for i in range(self.nc - 2):
#             temp1 = Q[i]
#             temp2 = Q[i + 1]
#             temp3 = Q[i + 2]
#             combined = temp1 + 4 * temp2 + temp3
#             p.append(combined / 6)
#         return anp.array(p)

#     def fov(self, p, c):
#         cent = []
#         radi = []
#         for i in range(self.nc - 2):
#             drn_pose = p[i][0:2]
#             tar_pose = c[i][0:2]
#             for j in range(self.m):
#                 lamb = (j + 1) / self.m
#                 center = lamb * (tar_pose - drn_pose) + drn_pose
#                 radius = anp.linalg.norm(drn_pose - tar_pose) * lamb * self.rho
#                 cent.append(center)
#                 radi.append(radius)
#         return anp.array(cent), anp.array(radi)
    
#     def obstacle_distance(self, x, y):
#         return self.esdf_function(x, y)

#     def dist_cost(self, p, c):
#         dist_cost = 0

#         for i in range(self.nc - 2):
#             d = anp.linalg.norm(p[i][0:2] - c[i][0:2])
#             dist_cost += self.penalty(self.od_min**2 - d**2) + self.penalty(d**2 - self.od_max**2)
#         return dist_cost

#     def ang_cost(self, p, c):
#         ang_cost = 0
#         for i in range(self.nc - 2):
#             best_ang = anp.arctan2((c[i][1] - p[i][1]), (c[i][0] - p[i][0]))
#             ang_cost += ((p[i][2]) * 3.14 / 180 - best_ang)**2
#         return ang_cost

#     def obs_cost(self, p, c):
#         obs_cost = 0
#         cent, radi = self.fov(p, c)
#         for i in range((self.nc - 2) * self.m):
#             cent_x = cent[i, 0]
#             cent_y = cent[i, 1]
#             E = self.obstacle_distance(cent_x, cent_y)
#             obs_cost += self.penalty(radi[i]**2 - E**2)
#         return obs_cost

#     def dyn_cost(self, Q):
#         v = []
#         a = []
#         j = []

#         for i in range(self.nc - 1):
#             v.append((Q[i + 1] - Q[i]) / self.dt)

#         for i in range(self.nc - 2):
#             a.append((v[i + 1] - v[i]) / self.dt)

#         for i in range(self.nc - 3):
#             j.append((a[i + 1] - a[i]) / self.dt)

#         v = anp.array(v)
#         a = anp.array(a)
#         j = anp.array(j)

#         Jf = 0
#         Jfp = 0
#         Js = 0
#         Jsp = 0

#         for i in range(self.nc - 1):
#             Jf += self.penalty(v[i][0]**2 + v[i][1]**2 - self.vm**2)
#             Jfp += self.penalty(v[i][2]**2 - self.vmp**2)

#         for i in range(self.nc - 2):
#             Jf += self.penalty(a[i][0]**2 + a[i][1]**2 - self.am**2)
#             Jfp += self.penalty(a[i][2]**2 - self.amp**2)

#         for i in range(self.nc - 3):
#             Js += j[i][0]**2 + j[i][1]**2
#             Jsp += j[i][2]**2

#         return Jf + Jfp + Js + Jsp

#     def collision_cost(self, Q):
#         Jc = 0
#         for i in range(self.nc - 2):
#             E = self.obstacle_distance(Q[i, 0], Q[i, 1])
#             Jc += self.penalty(self.d_thr**2 - E**2) * E
#         return Jc

#     def total_cost(self, Q_flat, c):
#         Q = anp.reshape(Q_flat, (self.nc, 3))
#         p = self.current_pose(Q)
#         dist = self.dist_cost(p, c)
#         ang = self.ang_cost(p, c)
#         obs = self.obs_cost(p, c)
#         dyn = self.dyn_cost(Q)
#         col = self.collision_cost(Q)
#         cost = 0.5 * dist + 0.5 * ang + 0.5 * dyn + 0.5 * obs + 0.5 * col
#         return cost

#     def optimize_path(self, follower_path, target_segment):
#         initial_guess = anp.array(follower_path[1:-1].flatten(), dtype=anp.float64)
#         grad_total_cost = grad(lambda Q_flat: self.total_cost(Q_flat, target_segment))
#         result = minimize(lambda Q_flat: self.total_cost(Q_flat, target_segment), initial_guess, method='L-BFGS-B', jac=grad_total_cost)
#         return result.x.reshape(-1, 3), result

#     def plot_paths(self, updated_follower_path, target_path_segment, p_prev, p_next, planner, i):
#         x_coords = updated_follower_path[:, 0]
#         y_coords = updated_follower_path[:, 1]
#         colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#         plt.figure()
#         plt.scatter(target_path_segment[:, 0], target_path_segment[:, 1], color=colors[2], marker='o', label=f'Target {i}')
#         plt.scatter(p_prev[:, 0], p_prev[:, 1], color=colors[3], marker='x', label=f'P_prev {i}')
#         plt.scatter(p_next[:, 0], p_next[:, 1], color=colors[4], marker='o', label=f'P_next {i}')
        
#         plt.xlabel('X coord')
#         plt.ylabel('Y coord')
#         fig = planner.plot.plotEnv("Map")
#         plt.title('X-Y graph')
#         plt.grid(True)
#         plt.legend()
#         plt.show()

#     def generate(self):
#         current_directory = os.getcwd()
#         target_data = pd.read_csv(self.target_path, skiprows=1)
#         target_path = target_data.to_numpy()
#         target_path = anp.array(target_path, dtype=anp.float64)

#         start = (5, 25)
#         env = Grid(51, 31)
#         search_factory = SearchFactory()
        
#         planner = search_factory("a_star", start=start, goal=(target_path[0][0], target_path[0][1]), env=env)
#         follower_path = planner.run()



        
#         for i in range(len(target_path) - 3):

#             follower_path = np.hstack((follower_path, np.zeros((len(follower_path), 1))))
#             follower_path = anp.array(follower_path)

#             for i in range(len(follower_path) - 1):
#                 x_diff = follower_path[i + 1, 0] - follower_path[i, 0]
#                 y_diff = follower_path[i + 1, 1] - follower_path[i, 1]
#                 follower_path[i, 2] = np.arctan2(y_diff, x_diff)

            
#             self.nc = len(follower_path)-2
#             self.m = self.nc - self.order
#             p_prev = self.current_pose(follower_path)
#             target_path_segment = target_path[i:i + self.nc]

#             updated_follower_path, result = self.optimize_path(follower_path, target_path_segment)
#             p_next = self.current_pose(updated_follower_path)

#             print(f"Optimization result for {i}: {result}")
            
#             new_start = (updated_follower_path[1, 0], updated_follower_path[1, 1])
#             new_goal = (target_path_segment[1, 0], target_path_segment[1, 1])
#             planner = search_factory("a_star", start=new_start, goal=new_goal, env=env)
#             follower_path = planner.run()
#             follower_path = anp.array(follower_path)

#             self.plot_paths(updated_follower_path, target_path_segment, p_prev, p_next, planner, i)


# if __name__ == '__main__':
#     esdf_path = "esdf.csv"
#     target_path = os.path.join(os.getcwd(), 'global', 'global_path.csv')
#     generator = PathGenerator(esdf_path, target_path)
#     generator.generate()
