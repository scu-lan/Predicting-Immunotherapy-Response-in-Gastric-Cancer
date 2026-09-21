def compute_mmd(X, Y, gamma=1.0):
    """
    Compute Maximum Mean Discrepancy (MMD) between two matrices X and Y using RBF kernel.

    Args:
        X (numpy.ndarray): First matrix with shape (n_samples_X, n_features).
        Y (numpy.ndarray): Second matrix with shape (n_samples_Y, n_features).
        gamma (float): Parameter for the RBF kernel.

    Returns:
        float: MMD value.
    """

    def rbf_kernel(x1, x2, gamma):
        """Compute RBF kernel between two matrices."""
        dist = np.sum(x1**2, axis=1, keepdims=True) + np.sum(x2**2, axis=1, keepdims=True).T - 2 * np.dot(x1, x2.T)
        return np.exp(-gamma * dist)

    # Compute RBF kernels
    K_XX = rbf_kernel(X, X, gamma)
    K_YY = rbf_kernel(Y, Y, gamma)
    K_XY = rbf_kernel(X, Y, gamma)

    m = X.shape[0]  # Number of samples in X
    n = Y.shape[0]  # Number of samples in Y

    # Compute MMD
    mmd = np.sum(K_XX) / (m * m)  # Include diagonal terms
    mmd += np.sum(K_YY) / (n * n)  # Include diagonal terms
    mmd -= 2 * np.sum(K_XY) / (m * n)

    return mmd
