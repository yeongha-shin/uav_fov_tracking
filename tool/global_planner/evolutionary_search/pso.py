"""
@file: pso.py
@breif: Particle Swarm Optimization (PSO) motion planning
@author: Winter
@update: 2024.2.8
"""
import random, math

from .evolutionary_search import EvolutionarySearcher
from tool.utils import Env
from tool.curve_generation import BSpline

GEN_MODE_CIRCLE = 0
GEN_MODE_RANDOM = 1

class PSO(EvolutionarySearcher):
    """
    Class for Particle Swarm Optimization (PSO) motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type
        n_particles (int): number of particles
        w_inertial (float): inertial weight
        w_social (float): social weight
        w_cognitive (float): cognitive weight
        point_num (int): number of position points contained in each particle
        max_speed (int): The maximum velocity of particle motion
        max_iter (int): maximum iterations
        init_mode (int): Set the generation mode for the initial position points of the particle swarm

    Examples:
        >>> from utils import Grid
        >>> from evolutionary_search import PSO
        >>> start = (5, 5)
        >>> goal = (45, 25)
        >>> env = Grid(51, 31)
        >>> planner = PSO(start, goal, env)
        >>> planner.run()
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean", 
        n_particles: int = 30, point_num: int = 6, w_inertial: float = 0.5,
        w_social: float = 1.0, w_cognitive: float = 0.6, max_speed: int = 8,
        max_iter: int = 100, init_mode: int = GEN_MODE_RANDOM) -> None:
        super().__init__(start, goal, env, heuristic_type)
        self.max_iter = max_iter
        self.n_particles = n_particles
        self.w_inertial = w_inertial
        self.w_social = w_social
        self.w_cognitive = w_cognitive
        self.point_num = point_num
        self.init_mode = init_mode
        self.max_speed = max_speed

        self.particles = []
        self.inherited_particles = []
        self.best_particle = self.Particle()
        self.b_spline_gen = BSpline(step=0.01, k=4)

    def __str__(self) -> str:
        return "Particle Swarm Optimization (PSO)"

    class Particle:
        def __init__(self) -> None:
            self.reset()
        
        def reset(self):
            self.position = []
            self.velocity = []
            self.fitness = -1
            self.best_pos = []
            self.best_fitness = -1
        
    def plan(self):
        """
        Particle Swarm Optimization (PSO) motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
        """
        # Generate initial position of particle swarm
        init_positions = self.initializePositions()

        # Particle initialization
        for i in range(self.n_particles):
            # Calculate fitness
            init_fitness = self.calFitnessValue(init_positions[i])

            if i == 0 or init_fitness > self.best_particle.fitness:
                self.best_particle.fitness = init_fitness
                self.best_particle.position = init_positions[i]
            
            # Create and add particle objects to containers
            p = self.Particle()
            p.position = init_positions[i]
            p.velocity = [(0, 0) for _ in range(self.point_num)]
            p.best_pos = init_positions[i]
            p.fitness = init_fitness
            p.best_fitness = init_fitness
            self.particles.append(p)

        # Iterative optimization
        for _ in range(self.max_iter):
            for p in self.particles:
                self.optimizeParticle(p)

        # Generating Paths from Optimal Particles
        points = [self.start.current] + self.best_particle.position + [self.goal.current]
        points = sorted(set(points), key=points.index)
        path = self.b_spline_gen.run(points, display=False)
        return self.b_spline_gen.length(path), path


    def initializePositions(self) -> list:
        """
        Generate n particles with pointNum_ positions each within the map range.

        Returns:
            init_positions (list): initial position sequence of particle swarm
        """
        init_positions = []

        # Calculate sequence direction
        x_order = self.goal.x > self.start.x
        y_order = self.goal.y > self.start.y

        # circle generation
        center_x, center_y, radius = None, None, None
        if self.init_mode == GEN_MODE_CIRCLE:
            # Calculate the center and the radius of the circle (midpoint between start and goal)
            center_x = (self.start.x + self.goal.x) / 2
            center_y = (self.start.y + self.goal.y) / 2
            radius = 5 if self.dist(self.start, self.goal) / 2.0 < 5 else self.dist(self.start, self.goal) / 2.0

        # initialize n_particles positions
        for _ in range(self.n_particles):
            point_id, visited = 0, []
            pts_x, pts_y = [], []
            # Generate point_num_ unique coordinates
            while point_id < self.point_num:
                if self.init_mode == GEN_MODE_RANDOM:
                    pt_x = random.randint(self.start.x, self.goal.x)
                    pt_y = random.randint(self.start.y, self.goal.y)
                    pos_id = pt_x + self.env.x_range * pt_y
                else:
                    # Generate random angle in radians
                    angle = random.random() * 2 * math.pi
                    # Generate random distance from the center within the circle
                    r = math.sqrt(random.random()) * radius
                    # Convert polar coordinates to Cartesian coordinates
                    pt_x = int(center_x + r * math.cos(angle))
                    pt_y = int(center_y + r * math.sin(angle))
                    # Check if the coordinates are within the map range
                    if 0 <= pt_x < self.env.x_range and 0 <= pt_y < self.env.y_range:
                        pos_id = pt_x + self.env.x_range * pt_y
                    else:
                        continue

                # Check if the coordinates have already been used
                if not pos_id in visited:
                    point_id = point_id + 1
                    visited.append(pos_id)
                    pts_x.append(pt_x)
                    pts_y.append(pt_y)

            # sort
            pts_x = sorted(pts_x, reverse=False) if x_order else sorted(pts_x, reverse=True)
            pts_y = sorted(pts_y, reverse=False) if y_order else sorted(pts_y, reverse=True)

            # Store elements from x and y in particle_positions
            init_positions.append([(ix, iy) for (ix, iy) in zip(pts_x, pts_y)])
        
        return init_positions

    def calFitnessValue(self, position: list) -> float:
        """
        Calculate the value of fitness function.

        Parameters:
            position (list): control points calculated by PSO

        Returns:
            fitness (float): the value of fitness function
        """
        points = [self.start.current] + position + [self.goal.current]
        points = sorted(set(points), key=points.index)
        try:
            points = self.b_spline_gen.run(points, display=False)
        except:
            return float("inf")

        # collision detection
        obs_cost = 0
        for p in points:
            px, py = int(p[0]), int(p[1])
            if (px, py) in self.env.obstacles or px < 0 or px >= self.env.x_range \
                or py < 0 or py >= self.env.y_range:
                obs_cost = obs_cost + 1

        # Calculate particle fitness
        return 100000.0 / (self.b_spline_gen.length(points) + 50000 * obs_cost)

    def updateParticleVelocity(self, particle):
        """
        A function to update the particle velocity

        Parameters:
            particle (Particle): the particle
        """
        # update Velocity
        for i in range(self.point_num):
            rand1, rand2 = random.random(), random.random()
            vx, vy = particle.velocity[i]
            px, py = particle.position[i]
            vx_new = self.w_inertial * vx + self.w_social * rand1 * (particle.best_pos[i][0] - px) \
                + self.w_cognitive * rand2 * (self.best_particle.position[i][0] - px)

            vy_new = self.w_inertial * vy + self.w_social * rand1 * (particle.best_pos[i][1] - py) \
                + self.w_cognitive * rand2 * (self.best_particle.position[i][1] - py)
            
            vy_new *= 1.3
 
            # Velocity limit
            vx_new = vx_new if vx_new < self.max_speed else self.max_speed
            vx_new = vx_new if vx_new > -self.max_speed else -self.max_speed
            vy_new = vy_new if vy_new < self.max_speed else self.max_speed
            vy_new = vy_new if vy_new > -self.max_speed else -self.max_speed
            particle.velocity[i] = (vx_new, vy_new)

    def updateParticlePosition(self, particle):
        """
        A function to update the particle position

        Parameters:
            particle (Particle): the particle
        """
        # update Position
        for i in range(self.point_num):
            px = particle.position[i][0] + int(particle.velocity[i][0])
            py = particle.position[i][1] + int(particle.velocity[i][1])

            # Position limit
            px = px if px < self.goal.x else self.goal.x
            px = px if px > self.start.x else self.start.x
            py = py if py < self.goal.y else self.goal.y
            py = py if py > self.start.y else self.start.y

            particle.position[i] = (px, py)
        particle.position.sort(key=lambda p: p[0])

    def optimizeParticle(self, particle):
        """
        Particle update optimization iteration

        Parameters:
            particle (Particle): the particle
        """
        # update speed
        self.updateParticleVelocity(particle)
        # update position
        self.updateParticlePosition(particle)

        # Calculate fitness
        particle.fitness = self.calFitnessValue(particle.position)

        # Update individual optima
        if particle.fitness > particle.best_fitness:
            particle.best_fitness = particle.fitness
            particle.best_pos = particle.position

        # Update global optimal particles
        if particle.best_fitness > self.best_particle.fitness:
            self.best_particle.fitness = particle.best_fitness
            self.best_particle.position = particle.position

    def run(self):
        """
        Running both plannig and animation.
        """
        cost, path = self.plan()
        self.plot.animation(path, str(self), cost)