# external imports
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
sys.path.append("../pympc/control/hscc/")
# internal imports
from pympc.control.hscc.controllers import HybridModelPredictiveController

# internal imports
from pympc.geometry.polyhedron import Polyhedron


import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', default=0, type=int, help='random seed')
parser.add_argument('-N', '--step', default=30, type=int, help='steps')
parser.add_argument('-H', '--step-size', default=0.05, type=float, help='step size')
parser.add_argument('-f', '--folder', default="./", type=str, help='folder to store the mps file')
args = parser.parse_args()

print(args)

seed = args.seed
random.seed(seed)

import numeric_parameters as params

params.N = args.step
params.h = args.step_size

from pwa_dynamics import S

# mixed-integer formulations
method = 'ch'

# initial condition
x0 = [0., 0., np.pi,
      0., 0.,
      0., 0., 0.,
      0., 0.]

if params.N * params.h <= 25 * 0.05:
    lb, ub = params.l/2, params.l
    x0[0] = random.uniform(lb, ub)
else:
    for j in [0,2]:
        lb, ub = -params.x_max[j], params.x_max[j]
        x0[j] = random.uniform(lb, ub)

x0 = np.array(x0)
print(x0)

gurobi_options = {'OutputFlag': 1, 'Threads': 1} # set OutputFlag to 0 to turn off gurobi log

for method in ['ch']:
    norm = 'none'
    # build the copntroller
    controller = HybridModelPredictiveController(
        S,
        params.N,
        params.Q,
        params.R,
        params.P,
        params.X_N,
        method,
        norm,
        noTerminal=False
    )

    # immediately kill solution
    controller.prog.setParam('TimeLimit', 0)

    modelName = "{}/N{}_h{}_seed{}.mps".format(args.folder, params.N, params.h, seed)

    # solve and store result
    controller.feedforward(x0, gurobi_options, modelName=modelName)
