import numpy as np
import pandas as pd

import sys
sys.path.insert(0, './')

from tool.utils import Grid, Map, SearchFactory
from tool.utils import CurveFactory

import matplotlib.pyplot as plt

def calculate_angle(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def run_simulation(start, goal, env, search_factory, move_step=1.0, min_distance=1.0):
    current_position = np.array(start, dtype=float)
    goal_position = np.array(goal, dtype=float)
    mid_position = np.array(start, dtype=float)

    planner = search_factory("a_star", start=tuple(current_position), goal=tuple(goal_position), env=env)
    global_path, global_fig = planner.run()  # path 계획 메서드로 가정

    global_fig.show()

    # write csv file
    angles = [0]
    for i in range(1, len(global_path)):
        angle = calculate_angle(global_path[i-1], global_path[i])
        angles.append(angle)

    data = {
        'x': [p[0] for p in global_path],
        'y': [p[1] for p in global_path],
        'angle': angles
    }

    df = pd.DataFrame(data)
    df.to_csv('./output2/global/global_path_100.csv', index=False)
    print("CSV file has been created with path and angles.")


    index = 0
    dis_thresh = 5

    # while np.linalg.norm(goal_position - mid_position) > min_distance:
    #     print("running... DTG = ", np.linalg.norm(goal_position - current_position))
    #
    #     if index + dis_thresh > len(global_path):
    #         mid_position = global_path[len(global_path)]
    #     else:
    #         mid_position = global_path[index+dis_thresh]
    #
    #     planner = search_factory("a_star", start=tuple(current_position), goal=tuple(mid_position), env=env)
    #     path, fig = planner.run()  # path 계획 메서드로 가정
    #
    #     local_angles = [0]
    #     for i in range(1, len(path)):
    #         angle = calculate_angle(path[i - 1], path[i])
    #         local_angles.append(angle)
    #
    #     local_data = {
    #         'x': [p[0] for p in path],
    #         'y': [p[1] for p in path],
    #         'angle': local_angles
    #     }
    #
    #     df = pd.DataFrame(local_data)
    #     df.to_csv(f'./output2/local/{index}.csv', index=False)
    #     plt.savefig(f'./output2/local/{index}.png')
    #     # curve added
    #
    #     curve_factory = CurveFactory()
    #     generator = curve_factory("bspline", step=0.01, k=3)
    #     generator.run(path)
    #
    #     current_position = np.array(path[1], dtype=float)
    #
    #     index += 1
    #
    # print("Reached the proximity of the goal.")


def plot_distance_transform(env, name, width, height):
    # Grid 환경 크기
    # width = env.width
    # height = env.height

    # 거리 계산을 위한 배열 생성
    distance_map = np.zeros((height, width), dtype=float)

    # 장애물 위치
    if isinstance(env, Grid):
        obstacle_positions = np.array(list(env.obstacles))
    elif isinstance(env, Map):
        # Map에서 모든 장애물 수집
        obstacle_positions = []
        for (ox, oy, w, h) in env.boundary:
            obstacle_positions.extend([(x, y) for x in range(ox, ox + w) for y in range(oy, oy + h)])
        for (ox, oy, w, h) in env.obs_rect:
            obstacle_positions.extend([(x, y) for x in range(ox, ox + w) for y in range(oy, oy + h)])
        for (ox, oy, r) in env.obs_circ:
            for x in range(-r, r + 1):
                for y in range(-r, r + 1):
                    if x**2 + y**2 <= r**2:
                        obstacle_positions.append((ox + x, oy + y))

        obstacle_positions = np.array(obstacle_positions)

    # 각 픽셀에 대해 가장 가까운 장애물까지의 거리 계산
    for x in range(width):
        for y in range(height):
            # 각 픽셀에 대해 가장 가까운 장애물과의 거리 계산
            distances = np.linalg.norm(obstacle_positions - [x, y], axis=1)
            distance_map[y, x] = np.min(distances)

    # 거리 맵을 시각화
    plt.imshow(distance_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Distance to Nearest Obstacle')
    plt.title(name)
    plt.show()

# ------------------------------------------------------------------
#                            Map generation
# ------------------------------------------------------------------

# 환경 및 검색 팩토리 설정
# env = Grid(51, 31)
env = Grid(100, 100)
# env = Grid(500, 500)
# env = Map(51, 31)

# ------------------------------------------------------------------
#                            Map generation (Outside) for
# ------------------------------------------------------------------

# 51 * 31
# start = (13, 25)
# goal = (45, 7)

# 100 * 100
start = (45, 71)
goal = (45, 29)

# 500 * 500
# start = (40 * 5, 75 * 5)
# goal = (40 * 5, 25 * 5)


current_position = np.array(start, dtype=float)
goal_position = np.array(goal, dtype=float)

search_factory = SearchFactory()
planner = search_factory("a_star", start=tuple(current_position), goal=tuple(goal_position), env=env)

fig = planner.plot.plotEnv("Map")
plt.show()

plot_distance_transform(env, "a_star", 51, 31)

# 시뮬레이션 실행
run_simulation(start, goal, env, search_factory)
