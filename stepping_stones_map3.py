import numpy as np
import matplotlib.pyplot as plt
from spp.convex_sets import Singleton, Polyhedron, CartesianProduct
from spp.convex_functions import SquaredTwoNorm
from spp.pwa_systems import PieceWiseAffineSystem, ShortestPathRegulator
from pydrake.solvers import CommonSolverOption

from pydrake.all import MathematicalProgram, MosekSolver, le, ge, eq, SolverOptions

import random
import sys

s = int(sys.argv[1])
random.seed(s)
# time horizon
K = 25

filename = f"map3_step25_seed{s}.mps"


def plot_terrain(q=None, u=None):
    plt.rc('axes', axisbelow=True)
    plt.gca().set_aspect('equal')

    for i, Dqi in enumerate(Dq):
        # color = 'lightcoral' if i in [1, 5] else 'lightcyan'
        # color = 'lightcoral' if i in [0, 1] else 'lightcyan'
        color = 'lightcyan'
        Dqi.plot(facecolor=color, alpha=min(1, Bs[i] + 0.4))

    plt.scatter(*q1, s=300, c='g', marker='+', zorder=2)
    plt.scatter(*qK, s=300, c='g', marker='x', zorder=2)

    if q is not None:
        plt.plot(*q.T, c='k', marker='o', markeredgecolor='k', markerfacecolor='w')

        if u is not None:
            for t, ut in enumerate(u):
                plt.arrow(*q[t], *ut, color='b', head_starts_at_zero=0, head_width=.15, head_length=.3)

    plt.xlabel(r'$q_1, w_1$')
    plt.ylabel(r'$q_2, w_2$')
    plt.grid(1)


# initial state
#z1 = np.array([-3.5, .5, 0, 0])
z1 = np.array([0.5, 0.5, 0, 0])
q1 = z1[:2]

# target set
#zK = np.array([3.5, 6.5, 0, 0])
zK = np.array([6.5, 6.5, 0, 0])
qK = zK[:2]
Z = Singleton(zK)



# cost matrices
q_dot_cost = .2 ** .5
Q = np.diag([0, 0, q_dot_cost, q_dot_cost])
R = np.eye(2)
S = Q # ininfluential
cost_matrices = (Q, R, S)


B1 = [0, 0]
# configuration bounds
Dq = []
for i in range(5):
    for j in range(5):
        B = [B1[0] + i * 1.5, B1[1] + j * 1.5]
        Dq.append(Polyhedron.from_bounds(B, [B[0] + 1, B[1] + 1]))

# velocity bounds
qdot_max = np.ones(2) * 1
qdot_min = - qdot_max
Dqdot = Polyhedron.from_bounds(qdot_min, qdot_max)

# control bounds
u_max = np.ones(2) * 0.5
u_min = - u_max
Du = Polyhedron.from_bounds(u_min, u_max)

# pwa domains
domains = [CartesianProduct((Dqi, Dqdot, Du)) for Dqi in Dq]

# dynamics
A = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
B = np.vstack((np.zeros((2, 2)), np.eye(2)))

Bs = [0.25] * 5 + [1] * 20
random.shuffle(Bs)

#Bred = B / 10
# Bred = B / 1000
# Bred = B / 1
c = np.zeros(4)
dynamics = [(A, Bs[i] * B, c) for i in range(len(domains))]

# pieceiwse affine system
pwa = PieceWiseAffineSystem(dynamics, domains)

var = 0
varToName = dict()

# initialize program
prog = MathematicalProgram()

# continuous decision variables
z = []
for r in range(K):
    lst = []
    for c in range(4):
        lst.append(prog.NewContinuousVariables(1, f"z{c}@{r}")[0])
        var += 1
        varToName[f"X{var}"] = f"z{c}@{r}"
    z.append(lst)
z = np.array(z)
#z = prog.NewContinuousVariables(K, 4)
u = []
for r in range(K - 1):
    lst = []
    for c in range(2):
        lst.append(prog.NewContinuousVariables(1, f"u{c}@{r}")[0])
        var += 1
        varToName[f"X{var}"] = f"u{c}@{r}"
    u.append(lst)
u = np.array(u)
#u = prog.NewContinuousVariables(K - 1, 2)

q = z[:, :2]
qdot = z[:, 2:]

# indicator variables
b = []
for r in range(K - 1):
    lst = []
    for c in range(len(Dq)):
        lst.append(prog.NewBinaryVariables(1, f"b{c}@{r}")[0])
        var += 1
        varToName[f"X{var}"] = f"b{c}@{r}"
    b.append(lst)
b = np.array(b)
#b = prog.NewBinaryVariables(K - 1, len(Dq))

# slack variables for the cost function
#s = prog.NewContinuousVariables(K - 1, len(Dq))
#prog.AddLinearConstraint(ge(s.flatten(), 0))
prog.AddLinearCost(0)

# initial and terminal conditions
prog.AddLinearConstraint(eq(z[0], z1))
prog.AddLinearConstraint(eq(z[-1], zK))

# containers for copies of state and the controls
z_aux = []
u_aux = []

# loop over time
for k in range(K - 1):
    # auxiliary copies of state and the controls
    Z = []
    for r in range(len(Dq)):
        lst = []
        for c in range(4):
            lst.append(prog.NewContinuousVariables(1, f"Z{r}_{c}@{k}")[0])
            var += 1
            varToName[f"X{var}"] = f"Z{r}_{c}@{k}"

        Z.append(lst)
    Z = np.array(Z)
    #Z = prog.NewContinuousVariables(len(Dq), 4)

    U = []
    for r in range(len(Dq)):
        lst = []
        for c in range(2):
            lst.append(prog.NewContinuousVariables(1, f"U{r}_{c}@{k}")[0])
            var += 1
            varToName[f"X{var}"] = f"U{r}_{c}@{k}"
        U.append(lst)
    U = np.array(U)
    #U = prog.NewContinuousVariables(len(Dq), 2)

    Znext = []
    for r in range(len(Dq)):
        lst = []
        for c in range(4):
            lst.append(prog.NewContinuousVariables(1, f"Znext{r}_{c}@{k}")[0])
            var += 1
            varToName[f"X{var}"] = f"Znext{r}_{c}@{k}"
        Znext.append(lst)
    Znext = np.array(Znext)

    #Znext = prog.NewContinuousVariables(len(Dq), 4)

    Q = Z[:, :2]
    Qdot = Z[:, 2:]
    z_aux.append(Z)
    u_aux.append(U)

    # loop over modes of the pwa system
    for i, Dqi in enumerate(Dq):

        # state and input bounds
        prog.AddLinearConstraint(le(Dqi.C.dot(Q[i]), Dqi.d * b[k, i]))
        prog.AddLinearConstraint(le(U[i], u_max * b[k, i]))
        prog.AddLinearConstraint(ge(U[i], - u_max * b[k, i]))
        prog.AddLinearConstraint(le(Qdot[i], qdot_max * b[k, i]))
        prog.AddLinearConstraint(ge(Qdot[i], - qdot_max * b[k, i]))

        # pwa dynamics
        Ai, Bi, ci = pwa.dynamics[i]
        prog.AddLinearConstraint(eq(Ai.dot(Z[i]) + Bi.dot(U[i]) + ci, Znext[i]))

        # reconstruct auxiliary variables
        prog.AddLinearConstraint(eq(sum(Z), z[k]))
        prog.AddLinearConstraint(eq(sum(U), u[k]))
        prog.AddLinearConstraint(eq(sum(Znext), z[k + 1]))
        prog.AddLinearConstraint(sum(b[k]) == 1)

# solve optimization
solver_options = SolverOptions()
solver_options.SetOption(MosekSolver().solver_id(), "MSK_IPAR_MIO_MAX_NUM_SOLUTIONS", 1)
solver_options.SetOption(MosekSolver().solver_id(), "MSK_IPAR_MIO_MAX_NUM_BRANCHES", 0)
solver_options.SetOption(MosekSolver().solver_id(), "MSK_IPAR_NUM_THREADS", 1)
solver_options.SetOption(MosekSolver().solver_id(), "writedata", filename)
solver_options.SetOption(MosekSolver().solver_id(), "MSK_DPAR_MIO_MAX_TIME", 10.0)
solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
sol = MosekSolver().Solve(prog, solver_options=solver_options)

z = sol.GetSolution(z)
u = sol.GetSolution(u)
q = z[:, :2]
u = u

mps = open(filename, 'r').read().strip().split("\n")
lines = []
for line in mps:
    for tok in line.split():
        if "X" in tok and tok in varToName:
            line = line.replace(tok, varToName[tok])
    lines.append(line)
with open(filename, 'w') as out_file:
    for line in lines:
        out_file.write(line + "\n")
# plot solution
# plt.figure(figsize=(4, 3))
plt.figure(figsize=(4, 4))
plot_terrain(q, u)
plt.xticks(range(-1, 17))
plt.yticks(range(-1, 13))

figure_name = f'{filename}.png'
plt.savefig(figure_name, bbox_inches='tight')
plt.close()
