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
        color = 'lightcoral' if i in [5, 6] else 'lightcyan'
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


# configuration bounds
Dq = [
    # Polyhedron.from_bounds([-4, 0], [3, 1]),
    # Polyhedron.from_bounds([-6, 1], [-5, 3]),
    # Polyhedron.from_bounds([4, 1], [5, 2]),
    # Polyhedron.from_bounds([-4, 3], [4, 4]),
    # Polyhedron.from_bounds([-5, 5], [-4, 6]),
    # Polyhedron.from_bounds([5, 4], [6, 6]),
    # Polyhedron.from_bounds([-3, 6], [4, 7])

    Polyhedron.from_bounds([2, 6], [4, 18]),
    Polyhedron.from_bounds([6, 6], [8, 18]),
    Polyhedron.from_bounds([11, 6], [13, 18]),
    Polyhedron.from_bounds([16, 6], [18, 18]),
    Polyhedron.from_bounds([20, 6], [22, 18]),
    Polyhedron.from_bounds([9, 10], [10, 15]),
    Polyhedron.from_bounds([14, 10], [15, 15]),
    Polyhedron.from_bounds([7, 3], [17, 5]),
    Polyhedron.from_bounds([2, 19], [4, 21]),
    Polyhedron.from_bounds([20, 19], [22, 21])
]

B = np.vstack((np.zeros((2, 2)), np.eye(2)))
# Bred = B / 10

for init_x in np.arange(7.5, 17.5, 1):
    for init_y in np.arange(3.5, 5.5, 1):
        # print(init_x, init_y)
        z1 = np.array([init_x, init_y, 0, 0])
        for target_x in np.concatenate((np.arange(2.5, 4.5, 1), np.arange(20.5, 22.5, 1)), axis=None):
            for target_y in np.arange(19.5, 21.5, 1):
                # print(target_x, target_y)
                zK = np.array([target_x, target_y, 0, 0])
                for disparity in [1, 10, 100, 1000]:
                    Bred = B / disparity
                    # print(Bred)

                    # original codes start here...
                    # initial state
                    # z1 = np.array([-3.5, .5, 0, 0])
                    # z1 = np.array([7.5, 4.5, 0, 0])
                    q1 = z1[:2]

                    # target set
                    # zK = np.array([3.5, 6.5, 0, 0])
                    # zK = np.array([20.5, 19.5, 0, 0])
                    qK = zK[:2]
                    Z = Singleton(zK)

                    # time horizon
                    K = 30

                    # cost matrices
                    q_dot_cost = .2 ** .5
                    Q = np.diag([0, 0, q_dot_cost, q_dot_cost])
                    R = np.eye(2)
                    S = Q  # ininfluential
                    cost_matrices = (Q, R, S)

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
                    # B = np.vstack((np.zeros((2, 2)), np.eye(2)))
                    # Bred = B / 10
                    c = np.zeros(4)
                    # dynamics = [(A, Bred, c) if i in [1, 5] else (A, B, c) for i in range(len(domains))]
                    dynamics = [(A, Bred, c) if i in [5, 6] else (A, B, c) for i in range(len(domains))]

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

                    # plot solution
                    # plt.figure(figsize=(4, 3))
                    plt.figure(figsize=(8, 8))
                    plot_terrain(q, u)
                    # plt.xticks(range(-6, 7))
                    plt.xticks(range(0, 25))
                    plt.yticks(range(1, 24))

                    # plot transparent triangles
                    if relaxation:
                        for v in reg.spp.graph.vertices:
                            E_out = reg.spp.graph.outgoing_edges(v)[1]
                            flow = sum(sol.spp.primal.phi[E_out])
                            if not np.isclose(flow, 0):
                                qv = sum(sol.spp.primal.y[E_out])[:2] / flow
                                plt.scatter(*qv, alpha=flow,
                                            marker='^', edgecolor='k', facecolor='w', zorder=2)

                    figure_name = 'base_maps/initial-[%g,%g]-target-[%g,%g]-disparity-[%d]-cost-[%g]-solve-[%g].png' % (
                        init_x, init_y, target_x, target_y, disparity, sol.spp.cost, sol.spp.time)
                    plt.savefig(figure_name, bbox_inches='tight')
                    plt.close()
