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

# !/usr/bin/env python3
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
od_min = 1
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
    return anp.maximum(0, x ** 3)


def current_pose(Q):
    p = []
    for i in range(nc - 2):
        temp1 = Q[i]
        temp2 = Q[i + 1]
        temp3 = Q[i + 2]
        combined = temp1 + 4 * temp2 + temp3
        p.append(combined / 6)
    return anp.array(p)


def fov(p, c):
    cent = []
    radi = []
    for i in range(nc - 2):
        drn_pose = p[i][0:2]
        tar_pose = c[i][0:2]
        for j in range(m):
            lamb = (j + 1) / m
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
    for i in range(nc - 2):
        d = anp.linalg.norm(p[i][0:2] - c[i][0:2])
        dist_cost += penalty(od_min ** 2 - d ** 2) + penalty(d ** 2 - od_max ** 2)
    return dist_cost


def ang_cost(p, c):
    ang_cost = 0
    for i in range(nc - 2):
        best_ang = anp.arctan2((c[i][1] - p[i][1]), (c[i][0] - p[i][0]))
        ang_cost += (p[i][2] - best_ang) ** 2
    return ang_cost


def obs_cost(p, c):
    obs_cost = 0
    cent, radi = fov(p, c)
    for i in range((nc - 2) * m):
        cent_x = cent[i, 0]
        cent_y = cent[i, 1]
        E = obstacle_distance(cent_x, cent_y)
        obs_cost += penalty(radi[i] ** 2 - E ** 2)
    return obs_cost


def dyn_cost(Q):
    v = []
    a = []
    j = []

    for i in range(nc - 1):
        v.append((Q[i + 1] - Q[i]) / dt)

    for i in range(nc - 2):
        a.append((v[i + 1] - v[i]) / dt)

    for i in range(nc - 3):
        j.append((a[i + 1] - a[i]) / dt)

    v = anp.array(v)
    a = anp.array(a)
    j = anp.array(j)

    Jf = 0
    Jfp = 0
    Js = 0
    Jsp = 0

    for i in range(nc - 1):
        Jf += penalty(v[i][0] ** 2 + v[i][1] ** 2 - vm ** 2)
        Jfp += penalty(v[i][2] ** 2 - vmp ** 2)

    for i in range(nc - 2):
        Jf += penalty(a[i][0] ** 2 + a[i][1] ** 2 - am ** 2)
        Jfp += penalty(a[i][2] ** 2 - amp ** 2)

    for i in range(nc - 3):
        Js += j[i][0] ** 2 + j[i][1] ** 2
        Jsp += j[i][2] ** 2

    return Jf + Jfp + Js + Jsp


def collision_cost(Q):
    Jc = 0
    for i in range(nc - 2):
        E = obstacle_distance(Q[i, 0], Q[i, 1])
        Jc += penalty(d_thr ** 2 - E ** 2) * E
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
    cost = 0 * dist + 0 * ang + 0 * dyn + 0 * obs + 0 * col
    return cost


# def pso_optimize(cost_function, bounds, num_particles=30, maxiter=100, w=0.5, c1=0.8, c2=0.9):
#     num_dimensions = len(bounds)
#     swarm = [np.random.uniform(bounds[:, 0], bounds[:, 1], num_dimensions) for _ in range(num_particles)]
#     velocity = [np.random.uniform(-1, 1, num_dimensions) for _ in range(num_particles)]
#     personal_best_positions = swarm[:]
#     personal_best_scores = [cost_function(p) for p in swarm]
#     global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
#     global_best_score = min(personal_best_scores)
#
#     for iter in range(maxiter):
#         for i in range(num_particles):
#             r1, r2 = np.random.rand(), np.random.rand()
#             velocity[i] = (w * velocity[i] +
#                            c1 * r1 * (personal_best_positions[i] - swarm[i]) +
#                            c2 * r2 * (global_best_position - swarm[i]))
#             swarm[i] = swarm[i] + velocity[i]
#             swarm[i] = np.clip(swarm[i], bounds[:, 0], bounds[:, 1])
#             score = cost_function(swarm[i])
#
#             if score < personal_best_scores[i]:
#                 personal_best_positions[i] = swarm[i]
#                 personal_best_scores[i] = score
#
#             if score < global_best_score:
#                 global_best_position = swarm[i]
#                 global_best_score = score
#
#     return global_best_position, global_best_score
#
#
# def optimize_fov(fail_cases, x, y, psi, fov_angle, fov_depth, c_segment):
#     def cost_function(params):
#         new_psi, new_x, new_y = params
#         new_psi_radian = new_psi * (np.pi / 180)
#         left_angle = new_psi_radian + np.radians(fov_angle / 2)
#         right_angle = new_psi_radian - np.radians(fov_angle / 2)
#
#         left_dx = fov_depth * np.cos(left_angle)
#         left_dy = fov_depth * np.sin(left_angle)
#
#         right_dx = fov_depth * np.cos(right_angle)
#         right_dy = fov_depth * np.sin(right_angle)
#
#         v1 = (new_x, new_y)
#         v2 = (new_x + left_dx, new_y + left_dy)
#         v3 = (new_x + right_dx, new_y + right_dy)
#
#         cost = 0
#         for idx in fail_cases:
#             cx, cy = c_segment[idx, 0], c_segment[idx, 1]
#             if not point_in_triangle((cx, cy), v1, v2, v3):
#                 cost += 1  # Penalty for each point outside the FOV
#         return cost
#
#     bounds = np.array([(psi - 10, psi + 10), (x - 0, x + 0), (y - 0, y + 2)])  # Adjust the bounds as necessary
#     best_params, best_cost = pso_optimize(cost_function, bounds)
#     return best_params


def pso_optimize(cost_function, bounds, num_particles=30, maxiter=100, w=0.5, c1=0.8, c2=0.9):
    num_dimensions = len(bounds)
    swarm = [np.random.uniform(bounds[:, 0], bounds[:, 1], num_dimensions) for _ in range(num_particles)]
    velocity = [np.random.uniform(-1, 1, num_dimensions) for _ in range(num_particles)]
    personal_best_positions = swarm[:]
    personal_best_scores = [cost_function(p) for p in swarm]
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    global_best_score = min(personal_best_scores)

    for iter in range(maxiter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocity[i] = (w * velocity[i] +
                           c1 * r1 * (personal_best_positions[i] - swarm[i]) +
                           c2 * r2 * (global_best_position - swarm[i]))
            swarm[i] = swarm[i] + velocity[i]
            swarm[i] = np.clip(swarm[i], bounds[:, 0], bounds[:, 1])
            score = cost_function(swarm[i])

            if score < personal_best_scores[i]:
                personal_best_positions[i] = swarm[i]
                personal_best_scores[i] = score

            if score < global_best_score:
                global_best_position = swarm[i]
                global_best_score = score

    return global_best_position, global_best_score

def optimize_fov(fail_idx, x, y, psi, fov_angle, fov_depth, cx, cy):
    def cost_function(params):
        new_psi, new_x, new_y = params
        new_psi_radian = new_psi * (np.pi / 180)
        left_angle = new_psi_radian + np.radians(fov_angle / 2)
        right_angle = new_psi_radian - np.radians(fov_angle / 2)

        left_dx = fov_depth * np.cos(left_angle)
        left_dy = fov_depth * np.sin(left_angle)

        right_dx = fov_depth * np.cos(right_angle)
        right_dy = fov_depth * np.sin(right_angle)

        v1 = (new_x, new_y)
        v2 = (new_x + left_dx, new_y + left_dy)
        v3 = (new_x + right_dx, new_y + right_dy)

        cost = 0
        if not point_in_triangle((cx, cy), v1, v2, v3):
            cost += 1  # Penalty for each point outside the FOV
        return cost

    bounds = np.array([(psi - 10, psi + 10), (x - 0, x + 0), (y - 0, y + 0)])  # Adjust the bounds as necessary
    best_params, best_cost = pso_optimize(cost_function, bounds)
    return best_params

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

        plt.fill(fov_x, fov_y, color='orange', alpha=0.3)

        v1 = (x, y)
        v2 = (x + left_dx, y + left_dy)
        v3 = (x + right_dx, y + right_dy)

        cx, cy = c_segment[idx, 0], c_segment[idx, 1]
        if point_in_triangle((cx, cy), v1, v2, v3):
            plt.scatter(cx, cy, color='blue', marker='o')
            success_cases.append(idx)
        else:
            plt.scatter(cx, cy, color='red', marker='o')
            fail_cases.append(idx)

            # Perform optimization for the fail case
            best_params = optimize_fov(idx, x, y, psi, fov_angle, fov_depth, cx, cy)
            new_psi, new_x, new_y = best_params

            # Update the coordinates and psi for the optimized fail case
            x_coords[idx] = new_x
            y_coords[idx] = new_y
            psi_coords[idx] = new_psi

            plt.figure()
            plt.scatter(x_coords, y_coords, color='g', marker='s', label='optimized positions')

            # Draw optimized FOV
            new_psi_radian = new_psi * (np.pi / 180)
            new_dx = length * np.cos(new_psi_radian)
            new_dy = length * np.sin(new_psi_radian)

            plt.arrow(new_x, new_y, new_dx, new_dy, head_width=0.5, head_length=0.5, fc='purple', ec='purple')

            new_left_angle = new_psi_radian + np.radians(fov_angle / 2)
            new_right_angle = new_psi_radian - np.radians(fov_angle / 2)

            new_left_dx = fov_depth * np.cos(new_left_angle)
            new_left_dy = fov_depth * np.sin(new_left_angle)

            new_right_dx = fov_depth * np.cos(new_right_angle)
            new_right_dy = fov_depth * np.sin(new_right_angle)

            new_fov_x = [new_x, new_x + new_left_dx, new_x + new_right_dx, new_x]
            new_fov_y = [new_y, new_y + new_left_dy, new_y + new_right_dy, new_y]

            plt.fill(new_fov_x, new_fov_y, color='blue', alpha=0.3)
            plt.scatter(c_segment[:, 0], c_segment[:, 1], color='orange', marker='o', label='c_segment')
            plt.xlabel('X coord')
            plt.ylabel('Y coord')
            plt.title('Optimized FOV with Particle Swarm Optimization')
            plt.legend()
            plt.grid(True)
            # plt.show()

    with open('./output/test_result/log.txt', 'a') as log_file:
        log_file.write(f"Iteration: {iteration}\n")
        log_file.write(f"Fail case (red): {fail_cases}\n")
        log_file.write(f"Success case (blue): {success_cases}\n")
        log_file.write("\n")

def point_in_triangle(pt, v1, v2, v3):

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


# def main():
#     current_directory = os.getcwd()
#     folder_path = os.path.join(current_directory, 'output', 'local')
#     target_path = os.path.join(current_directory, 'output', 'global', 'global_path.csv')
#     file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
#     file_paths.sort()
#
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
#         data = c[i:i + nc]
#         initial_guess = anp.array(data.flatten(), dtype=anp.float64)
#
#         # c_segment = c[i + nc + 1:i + nc + 4]  # Adjust the slicing as per your specific needs
#         c_segment = c[i + nc + 1:i + nc + 6]  # Adjust the slicing as per your specific needs
#
#         grad_total_cost = grad(lambda Q_flat: total_cost(Q_flat, c_segment))
#         result = minimize(lambda Q_flat: total_cost(Q_flat, c_segment), initial_guess, method='L-BFGS-B',
#                           jac=grad_total_cost)
#         q_optimized = result.x.reshape(-1, 3)
#         print(f"Optimization result for {i}: {result}")
#         x_coords = q_optimized[:, 0]
#         y_coords = q_optimized[:, 1]
#         psi_coords = q_optimized[:, 2]
#
#         # Plotting
#         plt.scatter(x_coords, y_coords, color=colors[1], label=f'optimized {i}')
#         plt.scatter(data[:, 0], data[:, 1], color=colors[0], marker='x', label=f'Initial control point {i}')
#         plt.scatter(c_segment[:, 0], c_segment[:, 1], color=colors[2], marker='o', label=f'Target {i}')
#
#         # Draw direction
#         draw_fov(x_coords, y_coords, psi_coords, c_segment, i)
#         print("psi result", psi_coords)
#
#
#         plt.xlabel('X coord')
#         plt.ylabel('Y coord')
#         fig = planner.plot.plotEnv("Map")
#         plt.title('X-Y graph')
#         plt.grid(True)
#         plt.legend()
#         plt.show()

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
    for i in range(len(c)):
        data = c[i:i + nc]
        initial_guess = anp.array(data.flatten(), dtype=anp.float64)

        c_segment = c[i + nc + 1:i + nc + 6]  # Adjust the slicing as per your specific needs

        grad_total_cost = grad(lambda Q_flat: total_cost(Q_flat, c_segment))
        result = minimize(lambda Q_flat: total_cost(Q_flat, c_segment), initial_guess, method='L-BFGS-B',
                          jac=grad_total_cost)
        q_optimized = result.x.reshape(-1, 3)
        print(f"Optimization result for {i}: {result}")
        x_coords = q_optimized[:, 0]
        y_coords = q_optimized[:, 1]
        psi_coords = q_optimized[:, 2]

        # Plotting
        plt.scatter(x_coords, y_coords, color=colors[1], label=f'optimized {i}')
        plt.scatter(data[:, 0], data[:, 1], color=colors[0], marker='x', label=f'Initial control point {i}')
        plt.scatter(c_segment[:, 0], c_segment[:, 1], color=colors[2], marker='o', label=f'Target {i}')

        # Draw direction and optimize FOV if necessary
        draw_fov(x_coords, y_coords, psi_coords, c_segment, i)
        print("psi result", psi_coords)

        plt.xlabel('X coord')
        plt.ylabel('Y coord')
        fig = planner.plot.plotEnv("Map")
        plt.title('X-Y graph')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()

