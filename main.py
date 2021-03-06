from mpi4py import MPI
import numpy as np
from time import time
import sys
from random_graph import generate_graph

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
np.random.seed(1234)

if __name__ == '__main__':
    if rank == 0:
        graph = generate_graph(1000)
        start = time()
        # graph = np.array(
        #     [
        #         [np.inf, 2, np.inf, 4, np.inf],
        #         [np.inf, np.inf, 3, 3, np.inf],
        #         [np.inf, np.inf, np.inf, np.inf, 2],
        #         [np.inf, np.inf, 3, np.inf, 4],
        #         [np.inf, np.inf, np.inf, np.inf, np.inf],
        #     ]
        # )
        n = graph.shape[0]
        p = int(n / size)

        for i in range(1, size):
            if i != size - 1:
                p_graph = graph[:, i * p : (i + 1) * p]
                vp = np.arange(i * p, (i + 1) * p)
            else:
                p_graph = graph[:, i * p :]
                vp = np.arange(i * p, n)
            comm.send((p_graph, vp, n), i)

        p_graph = graph[:, :p]
        vp = np.arange(p)

    else:
        p_graph, vp, n = comm.recv(source=0)

    visited = np.zeros(n)
    d = np.zeros(len(vp))
    for i in range(len(vp)):
        d[i] = p_graph[0, i]

    u = 0
    u_dist = 0
    visited[0] = 1

    while visited.sum() != n:
        best_dist = np.inf
        best_v = np.inf

        for i in range(len(vp)):
            d[i] = np.min([d[i], u_dist + p_graph[u, i]])
            if not visited[vp[i]] and d[i] < best_dist:
                best_dist = d[i]
                best_v = vp[i]

        if best_v != np.inf:
            print(f"Proces {rank}: wybieram wierzcholek {best_v} o dystansie {best_dist}")
        else:
            print(f"Proces {rank}: moje wszystkie wierzcholki zostaly odwiedzone")
        sys.stdout.flush()

        best_dist, best_v = comm.allreduce((best_dist, best_v), op=MPI.MINLOC)
        visited[best_v] = 1
        u = best_v
        u_dist = best_dist

        if rank == 0:
            print(f"Globalnie najlepszy wierzcholek to {best_v} o dystansie {best_dist}\n")
    if rank==0:
        print(time() - start)
        sys.stdout.flush()
