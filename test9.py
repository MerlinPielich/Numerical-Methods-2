
import numpy as np
import numba

@numba.njit
def compute_diffusion(phi, i, j, nx, ny, h, K):
    # Compute horizontal difference
    if j == 0:
        gx = (phi[i, j+1] - phi[i, j]) / h
    elif j == ny - 1:
        gx = (phi[i, j] - phi[i, j-1]) / h
    else:
        gx = (phi[i, j+1] - phi[i, j-1]) / (2 * h)
    # Compute vertical difference
    if i == 0:
        gy = (phi[i+1, j] - phi[i, j]) / h
    elif i == nx - 1:
        gy = (phi[i, j] - phi[i-1, j]) / h
    else:
        gy = (phi[i+1, j] - phi[i-1, j]) / (2 * h)
    G = np.sqrt(gx ** 2 + gy ** 2)
    return 1.0 / (1 + (G / K) ** 2)

@numba.njit
def assemble_coefficients(phi, phi0, h, dt, lambda_val, K):
    nx, ny = phi.shape
    N = nx * ny
    # Max entries: up to 5 per cell (1 diag + 4 neighbors)
    max_entries = 5 * N
    rows = np.empty(max_entries, dtype=np.int32)
    cols = np.empty(max_entries, dtype=np.int32)
    data = np.empty(max_entries, dtype=np.float64)
    b = np.empty(N, dtype=np.float64)
    count = 0
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            c_center = compute_diffusion(phi, i, j, nx, ny, h, K)
            if i < nx - 1:
                c_south = (c_center + compute_diffusion(phi, i + 1, j, nx, ny, h, K)) / 2
            else:
                c_south = c_center
            if i > 0:
                c_north = (c_center + compute_diffusion(phi, i - 1, j, nx, ny, h, K)) / 2
            else:
                c_north = c_center
            if j < ny - 1:
                c_east = (c_center + compute_diffusion(phi, i, j + 1, nx, ny, h, K)) / 2
            else:
                c_east = c_center
            if j > 0:
                c_west = (c_center + compute_diffusion(phi, i, j - 1, nx, ny, h, K)) / 2
            else:
                c_west = c_center

            diag = (1 + dt * lambda_val) + dt * (c_east + c_west + c_north + c_south) / (h ** 2)
            rows[count] = k
            cols[count] = k
            data[count] = diag
            count += 1
            if i < nx - 1:
                rows[count] = k
                cols[count] = k + ny
                data[count] = -dt * c_south / (h ** 2)
                count += 1
            if i > 0:
                rows[count] = k
                cols[count] = k - ny
                data[count] = -dt * c_north / (h ** 2)
                count += 1
            if j < ny - 1:
                rows[count] = k
                cols[count] = k + 1
                data[count] = -dt * c_east / (h ** 2)
                count += 1
            if j > 0:
                rows[count] = k
                cols[count] = k - 1
                data[count] = -dt * c_west / (h ** 2)
                count += 1

            b[k] = phi[i, j] + dt * lambda_val * phi0[i, j]
    return rows[:count], cols[:count], data[:count], b


def construct_matrix_A(phi, phi0, c_in, lambda_val, h, nx, ny, dt, K):
    N = nx * ny
    # Use the numba‑accelerated routine to build the entries and right-hand side
    rows, cols, data, b = assemble_coefficients(phi, phi0, h, dt, lambda_val, K)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A, b