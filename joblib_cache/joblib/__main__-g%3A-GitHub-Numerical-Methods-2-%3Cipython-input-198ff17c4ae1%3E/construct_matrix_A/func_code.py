# first line: 1
@memory.cache
def construct_matrix_A(phi, phi0, c_in, lambda_val, h, nx, ny, dt, K):
    N = nx * ny
    b = np.zeros(N)
    A = sparse.lil_matrix((N, N))

    # Helper function to compute diffusion coefficient at a point (i,j)
    def compute_c(i, j):
        if c_in is not None:
            return c_in[i, j]
        # Compute horizontal gradient
        if j == 0:
            gx = (phi[i, j + 1] - phi[i, j]) / h
        elif j == ny - 1:
            gx = (phi[i, j] - phi[i, j - 1]) / h
        else:
            gx = (phi[i, j + 1] - phi[i, j - 1]) / (2 * h)
        # Compute vertical gradient
        if i == 0:
            gy = (phi[i + 1, j] - phi[i, j]) / h
        elif i == nx - 1:
            gy = (phi[i, j] - phi[i - 1, j]) / h
        else:
            gy = (phi[i + 1, j] - phi[i - 1, j]) / (2 * h)
        G = np.sqrt(gx**2 + gy**2)
        return 1 / (1 + (G / K) ** 2)

    # Loop over each grid cell
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j  # 1D index (row-major order)
            c_center = compute_c(i, j)
            # Compute neighbor coefficients by averaging with current pixel
            if i < nx - 1:
                c_south = (c_center + compute_c(i + 1, j)) / 2
            else:
                c_south = c_center
            if i > 0:
                c_north = (c_center + compute_c(i - 1, j)) / 2
            else:
                c_north = c_center
            if j < ny - 1:
                c_east = (c_center + compute_c(i, j + 1)) / 2
            else:
                c_east = c_center
            if j > 0:
                c_west = (c_center + compute_c(i, j - 1)) / 2
            else:
                c_west = c_center

            # Diagonal element
            A[k, k] = (1 + dt * lambda_val) + dt * (c_east + c_west + c_north + c_south) / h**2
            # Off-diagonal elements
            if i < nx - 1:  # South neighbor
                A[k, k + ny] = -dt * c_south / h**2
            if i > 0:  # North neighbor
                A[k, k - ny] = -dt * c_north / h**2
            if j < ny - 1:  # East neighbor
                A[k, k + 1] = -dt * c_east / h**2
            if j > 0:  # West neighbor
                A[k, k - 1] = -dt * c_west / h**2

            b[k] = phi[i, j] + dt * lambda_val * phi0[i, j]  # Correct RHS

    if A is None or b is None:
        raise ValueError("Matrix A or vector b is not properly constructed.")
    return A.tocsc(), b  # Convert to CSC format
