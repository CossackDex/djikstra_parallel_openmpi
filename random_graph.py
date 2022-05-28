import numpy as np


def generate_graph(n):
    adj = np.random.randint(0, 2, (n, n))
    adj = adj.astype(np.float32)

    for i in range(n - 1):
        for j in range(n):
            if i == j or adj[i, j] == 0:
                adj[i, j] = np.inf
            else:
                adj[i, j] = float(np.random.randint(1, 11))

    adj[-1] = np.ones(n) * np.inf

    return adj


np.random.seed(1234)
print(generate_graph(5))
