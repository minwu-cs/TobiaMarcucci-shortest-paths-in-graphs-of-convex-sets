import numpy as np
import matplotlib.pyplot as plt
from spp.convex_sets import Singleton, Polyhedron, CartesianProduct
from spp.convex_functions import SquaredTwoNorm
from spp.pwa_systems import PieceWiseAffineSystem, ShortestPathRegulator

# initial state
# z1 = np.array([-3.5, .5, 0, 0])
z1 = np.array([8.1, 0.1, 0, 0])
q1 = z1[:2]

# target set
# zK = np.array([3.5, 6.5, 0, 0])
zK = np.array([19, 0.1, 0, 0])
qK = zK[:2]
Z = Singleton(zK)

# time horizon
K = 5

# cost matrices
q_dot_cost = .2 ** .5
Q = np.diag([0, 0, q_dot_cost, q_dot_cost])
R = np.eye(2)
S = Q  # ininfluential
cost_matrices = (Q, R, S)

# configuration bounds
Dq = [
    Polyhedron.from_bounds([0, 0], [8, 0.2]),
    Polyhedron.from_bounds([9, 0], [19.2, 0.2]),
    Polyhedron.from_bounds([1, 1], [7, 1.2]),
    Polyhedron.from_bounds([10, 1], [18, 1.2]),
    Polyhedron.from_bounds([2, 2], [7, 2.2]),
    Polyhedron.from_bounds([10, 2], [17, 2.2]),
    Polyhedron.from_bounds([3, 3], [8, 3.2]),
    Polyhedron.from_bounds([11, 3], [16, 3.2]),
    Polyhedron.from_bounds([4, 4], [9, 4.2]),
    Polyhedron.from_bounds([12, 4], [15, 4.2]),
    Polyhedron.from_bounds([5, 5], [8, 5.2]),
    Polyhedron.from_bounds([11, 5], [14, 5.2]),
    Polyhedron.from_bounds([6, 6], [7, 6.2]),
    Polyhedron.from_bounds([11, 6], [13, 6.2]),
    Polyhedron.from_bounds([7, 7], [8, 7.2]),
    Polyhedron.from_bounds([10, 7], [12, 7.2]),
    Polyhedron.from_bounds([2, 8], [5, 8.2]),
    Polyhedron.from_bounds([8, 8], [9, 8.2]),
    Polyhedron.from_bounds([10, 8], [11, 8.2]),
    Polyhedron.from_bounds([13, 9], [15, 9.2]),

    Polyhedron.from_bounds([8, 11], [11, 11.2]),
    Polyhedron.from_bounds([7, 12], [12, 12.2]),
    Polyhedron.from_bounds([6, 13], [13, 13.2]),
    Polyhedron.from_bounds([16, 13], [19, 13.2]),
    Polyhedron.from_bounds([5, 14], [14, 14.2]),
    Polyhedron.from_bounds([4, 15], [15, 15.2]),
    Polyhedron.from_bounds([3, 16], [9, 16.2]),
    Polyhedron.from_bounds([11, 16], [16, 16.2]),
    Polyhedron.from_bounds([2, 17], [9, 17.2]),
    Polyhedron.from_bounds([11, 17], [17, 17.2]),
    Polyhedron.from_bounds([1, 18], [18, 18.2]),
    Polyhedron.from_bounds([0, 19], [19, 19.2]),

    Polyhedron.from_bounds([0, 0], [0.2, 19]),
    Polyhedron.from_bounds([1, 1], [1.2, 18]),
    Polyhedron.from_bounds([2, 2], [2.2, 7]),
    Polyhedron.from_bounds([2, 8], [2.2, 17]),
    Polyhedron.from_bounds([3, 3], [3.2, 8]),
    Polyhedron.from_bounds([3, 9], [3.2, 16]),
    Polyhedron.from_bounds([4, 4], [4.2, 7]),
    Polyhedron.from_bounds([4, 9], [4.2, 15]),
    Polyhedron.from_bounds([5, 5], [5.2, 8]),
    Polyhedron.from_bounds([5, 9], [5.2, 14]),
    Polyhedron.from_bounds([6, 6], [6.2, 13]),
    Polyhedron.from_bounds([7, 7], [7.2, 9]),
    Polyhedron.from_bounds([7, 10], [7.2, 12]),
    Polyhedron.from_bounds([8, 0], [8.2, 3]),
    Polyhedron.from_bounds([8, 5], [8.2, 7]),
    Polyhedron.from_bounds([8, 8], [8.2, 11]),
    Polyhedron.from_bounds([9, 0], [9.2, 8]),

    Polyhedron.from_bounds([10, 2], [10.2, 8]),
    Polyhedron.from_bounds([10, 15], [10.2, 18]),
    Polyhedron.from_bounds([11, 3], [11.2, 5]),
    Polyhedron.from_bounds([11, 8], [11.2, 11]),
    Polyhedron.from_bounds([12, 7], [12.2, 12]),
    Polyhedron.from_bounds([13, 6], [13.2, 13]),
    Polyhedron.from_bounds([14, 5], [14.2, 8]),
    Polyhedron.from_bounds([14, 10], [14.2, 14]),
    Polyhedron.from_bounds([15, 4], [15.2, 9]),
    Polyhedron.from_bounds([15, 10], [15.2, 15]),
    Polyhedron.from_bounds([16, 3], [16.2, 12]),
    Polyhedron.from_bounds([16, 13], [16.2, 16]),
    Polyhedron.from_bounds([17, 2], [17.2, 12]),
    Polyhedron.from_bounds([17, 14], [17.2, 17]),
    Polyhedron.from_bounds([18, 1], [18.2, 12]),
    Polyhedron.from_bounds([18, 14], [18.2, 18]),
    Polyhedron.from_bounds([19, 1], [19.2, 19])
]

# velocity bounds
qdot_max = np.ones(2) * 6
qdot_min = - qdot_max
Dqdot = Polyhedron.from_bounds(qdot_min, qdot_max)

# control bounds
u_max = np.ones(2) * 6
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
Bred = B / 10
c = np.zeros(4)
# dynamics = [(A, Bred, c) if i in [1, 5] else (A, B, c) for i in range(len(domains))]
# dynamics = [(A, Bred, c) if i in [5, 6] else (A, B, c) for i in range(len(domains))]
dynamics = [(A, B, c) for i in range(len(domains))]

# pieceiwse affine system
pwa = PieceWiseAffineSystem(dynamics, domains)

# solve optimal control problem
relaxation = 0
reg = ShortestPathRegulator(pwa, K, z1, Z, cost_matrices, relaxation=relaxation)
sol = reg.solve()
print('Cost:', sol.spp.cost)
print('Solve time:', sol.spp.time)

# unpack result
q = sol.z[:, :2]
u = sol.u


def plot_terrain(q=None, u=None):
    plt.rc('axes', axisbelow=True)
    plt.gca().set_aspect('equal')

    for i, Dqi in enumerate(Dq):
        # color = 'lightcoral' if i in [1, 5] else 'lightcyan'
        # color = 'lightcoral' if i in [5, 6] else 'lightcyan'
        color = 'lightcyan'
        Dqi.plot(facecolor=color)

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


# plot solution
# plt.figure(figsize=(4, 3))
plt.figure(figsize=(10, 10))
plot_terrain(q, u)
# plt.xticks(range(-6, 7))
plt.xticks(range(0, 20))
plt.yticks(range(0, 20))

# plot transparent triangles
if relaxation:
    for v in reg.spp.graph.vertices:
        E_out = reg.spp.graph.outgoing_edges(v)[1]
        flow = sum(sol.spp.primal.phi[E_out])
        if not np.isclose(flow, 0):
            qv = sum(sol.spp.primal.y[E_out])[:2] / flow
            plt.scatter(*qv, alpha=flow,
                        marker='^', edgecolor='k', facecolor='w', zorder=2)
plt.savefig('footstep_spp.png', bbox_inches='tight')
