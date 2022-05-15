import numpy as np

def simulateLDS(N, A, Q, C, R, m0, V0):
    M = A.shape[0]
    P = C.shape[0]
    # state noise
    w = np.random.multivariate_normal(np.zeros(M), Q, N).T
    # measurement noise
    v = np.random.multivariate_normal(np.zeros(P), R, N).T
    # initial state noise
    x = np.empty(shape=(M, N))
    y = np.empty(shape=(P, N))
    x0 = np.random.multivariate_normal(m0, V0, 1).flatten()
    x[:, 0] = A @ x0 + w[:, 0]
    for n in range(1, N):
        x[:, n] = A @ x[:, n-1] + w[:, n]
    y = C @ x + v
    return x0, x, y
