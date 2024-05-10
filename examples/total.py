import numpy as np
import pandas as pd

import sys
sys.path.insert(0, './')

from tool.utils import Grid, Map, SearchFactory
from tool.utils import CurveFactory

def calculate_angle(p1, p2):
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

def run_simulation(start, goal, env, search_factory, move_step=1.0, min_distance=1.0):
    current_position = np.array(start, dtype=float)
    goal_position = np.array(goal, dtype=float)
    mid_position = np.array(start, dtype=float)

    planner = search_factory("a_star", start=tuple(current_position), goal=tuple(goal_position), env=env)
    global_path = planner.run()  # path 계획 메서드로 가정

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
    df.to_csv('./output/global/global_path.csv', index=False)
    print("CSV file has been created with path and angles.")


    index = 0
    dis_thresh = 10

    while np.linalg.norm(goal_position - mid_position) > min_distance:
        print("running... DTG = ", np.linalg.norm(goal_position - current_position))

        if index + dis_thresh > len(global_path):
            mid_position = global_path[len(global_path)]
        else:
            mid_position = global_path[index+dis_thresh]

        planner = search_factory("a_star", start=tuple(current_position), goal=tuple(mid_position), env=env)
        path = planner.run()  # path 계획 메서드로 가정

        local_angles = [0]
        for i in range(1, len(path)):
            angle = calculate_angle(path[i - 1], path[i])
            local_angles.append(angle)

        local_data = {
            'x': [p[0] for p in path],
            'y': [p[1] for p in path],
            'angle': local_angles
        }

        df = pd.DataFrame(local_data)
        df.to_csv(f'./output/local/{index}.csv', index=False)

        # curve added

        curve_factory = CurveFactory()
        generator = curve_factory("bspline", step=0.01, k=3)
        generator.run(path)

        current_position = np.array(path[1], dtype=float)

        index += 1

    print("Reached the proximity of the goal.")

# 환경 및 검색 팩토리 설정
env = Grid(51, 31)
# env = Map(51, 31)
search_factory = SearchFactory()
start = (5, 25)
# goal = (45, 25)
goal = (45, 5)

# 시뮬레이션 실행
run_simulation(start, goal, env, search_factory)
