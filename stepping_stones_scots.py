import numpy as np
import matplotlib.pyplot as plt
from spp.convex_sets import Singleton, Polyhedron, CartesianProduct
from spp.convex_functions import SquaredTwoNorm
from spp.pwa_systems import PieceWiseAffineSystem, ShortestPathRegulator


def plot_terrain(q=None, u=None):
    plt.rc('axes', axisbelow=True)
    plt.gca().set_aspect('equal')

    for i, Dqi in enumerate(Dq):
        # color = 'lightcoral' if i in [1, 5] else 'lightcyan'
        # color = 'lightcoral' if i in [0, 1] else 'lightcyan'
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


# initial state
# z1 = np.array([-3.5, .5, 0, 0])
z1 = np.array([0.5, 0.5, 0, 0])
q1 = z1[:2]

# target set
# zK = np.array([3.5, 6.5, 0, 0])
zK = np.array([50.5, 0.5, 0, 0])
qK = zK[:2]
Z = Singleton(zK)

# time horizon
K = 60

# cost matrices
q_dot_cost = .2 ** .5
Q = np.diag([0, 0, q_dot_cost, q_dot_cost])
R = np.eye(2)
S = Q  # ininfluential
cost_matrices = (Q, R, S)

import os
if not os.path.exists('2_base_maps/'):
   os.makedirs('2_base_maps/')

for alpha in np.arange(49, 39, -2):
    for bravo in np.arange(0, 10, 2):
        for charlie in np.arange(49, 39, -2):
            for delta in np.arange(0, 10, 2):
                for echo in np.arange(49, 39, -2):
                    # print(alpha, bravo, charlie, delta, echo)

                    # configuration bounds
                    Dq = [
                        Polyhedron.from_bounds([0, alpha], [10, alpha + 1]),
                        Polyhedron.from_bounds([10, bravo], [20, bravo + 1]),
                        Polyhedron.from_bounds([20, charlie], [30, charlie + 1]),
                        Polyhedron.from_bounds([30, delta], [40, delta + 1]),
                        Polyhedron.from_bounds([40, echo], [50, echo + 1]),

                        # Polyhedron.from_bounds([0, 49], [10, 50]),
                        # Polyhedron.from_bounds([10, 0], [20, 1]),
                        # Polyhedron.from_bounds([20, 49], [30, 50]),
                        # Polyhedron.from_bounds([30, 0], [40, 1]),
                        # Polyhedron.from_bounds([40, 49], [50, 50]),

                        Polyhedron.from_bounds([0, 17], [3, 18]),
                        Polyhedron.from_bounds([10, 32], [13, 33]),
                        Polyhedron.from_bounds([20, 4], [22, 5]),
                        Polyhedron.from_bounds([30, 44], [33, 45]),
                        Polyhedron.from_bounds([40, 7], [42, 8]),

                        Polyhedron.from_bounds([0, 0], [1, 50]),
                        Polyhedron.from_bounds([10, 0], [11, 50]),
                        Polyhedron.from_bounds([20, 0], [21, 50]),
                        Polyhedron.from_bounds([30, 0], [31, 50]),
                        Polyhedron.from_bounds([40, 0], [41, 50]),
                        Polyhedron.from_bounds([50, 0], [51, 50]),
                    ]
                    print("alpha:", [0, alpha], [10, alpha + 1])
                    print("bravo:", [10, bravo], [20, bravo + 1])
                    print("charlie:", [20, charlie], [30, charlie + 1])
                    print("delta:", [30, delta], [40, delta + 1])
                    print("echo:", [40, echo], [50, echo + 1])

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
                    # Bred = B / 10
                    # Bred = B / 1000
                    # Bred = B / 1
                    c = np.zeros(4)
                    # dynamics = [(A, Bred, c) if i in [1, 5] else (A, B, c) for i in range(len(domains))]
                    # dynamics = [(A, Bred, c) if i in [0, 1] else (A, B, c) for i in range(len(domains))]
                    dynamics = [(A, B, c) for i in range(len(domains))]

                    # pieceiwse affine system
                    pwa = PieceWiseAffineSystem(dynamics, domains)

                    # solve optimal control problem
                    relaxation = 0
                    reg = ShortestPathRegulator(pwa, K, z1, Z, cost_matrices, relaxation=relaxation)
                    sol = reg.solve()
                    # print('Cost:', sol.spp.cost)
                    # print('Solve time:', sol.spp.time)

                    # unpack result
                    q = sol.z[:, :2]
                    u = sol.u

                    # plot solution
                    # plt.figure(figsize=(4, 3))
                    plt.figure(figsize=(20, 20))
                    plot_terrain(q, u)
                    # plt.xticks(range(-6, 7))
                    plt.xticks(range(-2, 54))
                    plt.yticks(range(-2, 54))

                    # plot transparent triangles
                    if relaxation:
                        for v in reg.spp.graph.vertices:
                            E_out = reg.spp.graph.outgoing_edges(v)[1]
                            flow = sum(sol.spp.primal.phi[E_out])
                            if not np.isclose(flow, 0):
                                qv = sum(sol.spp.primal.y[E_out])[:2] / flow
                                plt.scatter(*qv, alpha=flow,
                                            marker='^', edgecolor='k', facecolor='w', zorder=2)
                    figure_name = '2_base_maps/alpha-[%d]-bravo-[%d]-charlie-[%d]-delta-[%d]-echo-[%d]-cost-[%g]-solve-[%g].png' % (
                        alpha, bravo, charlie, delta, echo, sol.spp.cost, sol.spp.time)
                    plt.savefig(figure_name, bbox_inches='tight')
                    plt.close()
                    print("\n"*3)
