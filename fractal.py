import torch
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Barnsley Fern — PyTorch IFS
# -------------------------------

def generate_barnsley_fern(
    n_points=300_000,
    burn_in=20,
    batch_size=8192,
    device=None,
    dtype=torch.float32,
    return_all=True,
):
    """
    Generate Barnsley Fern points using PyTorch and the classic 4 affine maps.

    Args:
        n_points   : total number of output points (after burn-in)
        burn_in    : number of initial steps to discard
        batch_size : number of parallel walkers (bigger -> faster on GPU)
        device     : "cuda" if available else "cpu"
        dtype      : torch dtype
        return_all : if True, returns all points (N x 2) tensor

    Returns:
        points (N x 2) torch.FloatTensor on CPU
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Affine maps (classic Barnsley parameters)
    # Each transform: (a,b,c,d,e,f) corresponds to:
    # x' = a*x + b*y + e
    # y' = c*x + d*y + f
    A = torch.tensor([
        [0.0,   0.0,   0.0,  0.16,  0.0,  0.0],   # f1
        [0.85,  0.04, -0.04, 0.85,  0.0,  1.6],   # f2
        [0.20, -0.26,  0.23, 0.22,  0.0,  1.6],   # f3
        [-0.15, 0.28,  0.26, 0.24,  0.0,  0.44],  # f4
    ], device=device, dtype=dtype)

    # Probabilities for each map
    probs = torch.tensor([0.01, 0.85, 0.07, 0.07], device=device, dtype=dtype)
    probs = probs / probs.sum()

    # Categorical sampler
    cat = torch.distributions.Categorical(probs=probs)

    # Initialize batch of walkers at origin
    x = torch.zeros(batch_size, device=device, dtype=dtype)
    y = torch.zeros(batch_size, device=device, dtype=dtype)

    # Burn-in (to get onto the attractor)
    for _ in range(burn_in):
        idx = cat.sample((batch_size,))  # (batch_size,)
        # Gather parameters for selected transforms
        a, b, c, d, e, f = (A[idx, 0], A[idx, 1], A[idx, 2], A[idx, 3], A[idx, 4], A[idx, 5])
        x, y = a * x + b * y + e, c * x + d * y + f

    # Allocate output buffer
    points = torch.empty((n_points, 2), device=device, dtype=dtype)

    written = 0
    while written < n_points:
        # Single IFS step for all walkers
        idx = cat.sample((batch_size,))
        a, b, c, d, e, f = (A[idx, 0], A[idx, 1], A[idx, 2], A[idx, 3], A[idx, 4], A[idx, 5])
        x, y = a * x + b * y + e, c * x + d * y + f

        # Write as many as we can from this step
        take = min(batch_size, n_points - written)
        points[written:written + take, 0] = x[:take]
        points[written:written + take, 1] = y[:take]
        written += take

    # Return on CPU for plotting/analysis compatibility
    return points.detach().cpu()


def plot_scatter(points, s=0.05):
    """Simple scatter plot (fast, clean)."""
    plt.figure(figsize=(6, 9))
    plt.scatter(points[:, 0].numpy(), points[:, 1].numpy(), s=s, marker=".", linewidths=0)
    plt.axis("equal")
    plt.axis("off")
    plt.title("Barnsley Fern — Scatter")
    plt.show()


def plot_density(points, bins=1000):
    """
    Density/heatmap using 2D histogram.
    Produces a crisp fern without scatter overdraw.
    """
    x = points[:, 0].numpy()
    y = points[:, 1].numpy()

    # Tight bounding box for classic fern
    xlim = (x.min(), x.max())
    ylim = (y.min(), y.max())

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xlim, ylim])
    H = H.T  # for imshow (row = y)

    plt.figure(figsize=(6, 9))
    plt.imshow(
        np.log1p(H),  # log scale for nicer contrast
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation="nearest",
        aspect="equal",
    )
    plt.axis("off")
    plt.title("Barnsley Fern — Density")
    plt.colorbar(shrink=0.7, label="log(1 + count)")
    plt.show()


# -------------------------------
# (Optional) Box-counting dimension
# -------------------------------

def estimate_box_counting_dimension(points, eps_list=None):
    """
    Estimate fractal (box-counting) dimension by counting occupied boxes
    for multiple box sizes eps, then linear fit on log-log plot.

    Returns:
        (eps_list, counts, slope)
    """
    if eps_list is None:
        eps_list = np.geomspace(0.01, 0.3, num=10)

    xy = points.numpy()
    x, y = xy[:, 0], xy[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    counts = []
    for eps in eps_list:
        nx = int(np.ceil((x_max - x_min) / eps))
        ny = int(np.ceil((y_max - y_min) / eps))
        # Map points to grid indices
        ix = np.floor((x - x_min) / eps).astype(int)
        iy = np.floor((y - y_min) / eps).astype(int)
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        # Count unique occupied boxes
        occupied = np.unique(ix * ny + iy).size
        counts.append(occupied)

    # Linear regression on log-log
    log_eps_inv = np.log(1 / np.array(eps_list))
    log_counts  = np.log(np.array(counts))
    slope, intercept = np.polyfit(log_eps_inv, log_counts, 1)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(log_eps_inv, log_counts, "o-", label=f"fit slope ≈ {slope:.3f}")
    plt.xlabel("log(1/ε)")
    plt.ylabel("log(N(ε))")
    plt.title("Box-Counting Dimension — Barnsley Fern")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return eps_list, counts, slope


# -------------------------------
# Demo
# -------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Generate points (increase n_points for crisper density plots)
    pts = generate_barnsley_fern(
        n_points=400_000,
        burn_in=20,
        batch_size=16384,
        device=device
    )

    # Visualisations
    plot_scatter(pts, s=0.03)
    plot_density(pts, bins=900)

    # Optional: estimate fractal dimension
    _, _, dim = estimate_box_counting_dimension(pts)
    print(f"Estimated box-counting dimension ≈ {dim:.3f}")
