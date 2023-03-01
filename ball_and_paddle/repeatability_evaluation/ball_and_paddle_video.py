import sys
sys.path.append("../")
sys.path.append("../pympc/")
sys.path.append("../pympc/control/hscc/")
import subprocess
import numpy as np

N = 25

solution = dict()
solution_file = "./solutions/N25_h0.05_seed11.mps.solution"
with open(solution_file, 'r') as in_file:
    for line in in_file.readlines():
        variable, value = line.split()
        solution[variable] = float(value)

N = 30 if "N30" in solution_file else 25
x = np.array([np.array([solution[f'x@{t}[{k}]'] for k in range(10)]) for t in range(N+1)])
assert(len(x) == N + 1)

import meshcat
import meshcat.transformations as tf
from meshcat.geometry import Box, Sphere, Cylinder, MeshLambertMaterial
from meshcat.animation import Animation

# initialize visualizer
vis = meshcat.Visualizer()
vis.open()

# geometric parameters of the ball and the paddle
import numeric_parameters as params
tickness = .01
depth = .3

# colors of the scene
red = 0xff5555
blue = 0x5555ff
green = 0x55ff55
grey = 0xffffff

vis['ball_right'].set_object(
    Sphere(params.r),
    MeshLambertMaterial(color=blue)
)
vis['ball_left'].set_object(
    Sphere(params.r),
    MeshLambertMaterial(color=green)
)
vis['floor'].set_object(
    Box([depth, params.l*2., tickness]),
    MeshLambertMaterial(color=red)
)
vis['ceiling'].set_object(
    Box([depth, params.l*2., tickness]),
    MeshLambertMaterial(color=grey)
)

x = list(x)
x = np.array([x[0]] * 5 + x)

# initialize animation
anim = Animation()

# animate, e.g., the solution with infinity norm (and convex-hull method -- irrelevant)

for t, xt in enumerate(x):
    with anim.at_frame(vis, t*params.h*80) as frame: # 30 frames per second to get real time
        frame['ball_right'].set_transform(
            tf.translation_matrix([0, xt[0], xt[1]+params.r]).dot(
                tf.rotation_matrix(xt[2], [1.,0.,0.]).dot(
                    tf.translation_matrix([0, -0.001, 0])
                )
            )
        )
        frame['ball_left'].set_transform(
            tf.translation_matrix([0, xt[0], xt[1]+params.r]).dot(
                tf.rotation_matrix(xt[2], [1.,0.,0.]).dot(
                    tf.translation_matrix([0, 0.001, 0])
                )
            )
        )
        frame['floor'].set_transform(
            tf.translation_matrix([0, xt[3], xt[4]-tickness/2.])
        )
        frame['ceiling'].set_transform(
            tf.translation_matrix([0, 0, params.d+tickness/2.])
        )

# visualize result
vis.set_animation(anim)

