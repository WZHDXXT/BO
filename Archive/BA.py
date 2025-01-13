
import math
from dataclasses import dataclass
import torch
from sklearn.metrics.pairwise import rbf_kernel
from pyDOE import lhs
import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
@dataclass
class BaxusState:
    # dimensionality of input variables
    dim: int 
    # evaluatoin budget
    eval_budget: int 
    # new bins each dimension split
    new_bins_on_split: int = 2
    # initial dimensionality used by the algorithm
    d_init: int = float("nan")  
    # target dimensionality
    target_dim: int = float("nan")  
    # number of times that has been split
    n_splits: int = float("nan") 

    # trust region
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    
    failure_counter: int = 0
    success_counter: int = 0

    success_tolerance: int = 3
    
    # value of new best
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        n_splits = round(math.log(self.dim, self.new_bins_on_split + 1))
        self.d_init = 1 + np.argmin(
            np.abs(
                (1 + np.arange(self.new_bins_on_split))
                * (1 + self.new_bins_on_split) ** n_splits
                - self.dim
            )
        )
        self.target_dim = self.d_init
        self.n_splits = n_splits

    @property
    def split_budget(self) -> int:
        split_budget = round(
            -1
            * (self.new_bins_on_split * self.eval_budget * self.target_dim)
            / (self.d_init * (1 - (self.new_bins_on_split + 1) ** (self.n_splits + 1)))
        )
        return split_budget
    @property
    def failure_tolerance(self) -> int:
        if self.target_dim == self.dim:
            return self.target_dim
        k = math.floor(math.log(self.length_min / self.length_init, 0.5))
        split_budget = self.split_budget
        failure_tolerance = min(self.target_dim, max(1, math.floor(split_budget / k)))
        
        return failure_tolerance

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region by half
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state    

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length_init: float = 0.8
    length: float = 0.8
    length_min: float = 0.5 ** 4
    length_max: float = 1.6
    
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 3  # Note: The original paper uses 3
    
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


'''def get_initial_points(
    dim: int, 
    n_pts: int, 
    seed: int,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    dtype=torch.float64
):
    sobol = SobolEngine(dimension=dim, scramble=False, seed=seed)
    X_init = (
        10 * sobol.draw(n=n_pts).to(dtype=dtype, device=device) - 5
    )  # points have to be in [-5, 5]^d
    return X_init'''

def get_initial_points(
    dim: int, 
    n_pts: int, 
    seed: int,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    dtype=torch.float64
):
    np.random.seed(seed)
    X_init = lhs(dim, samples=n_pts, criterion='center')
    # X_init = 10 * X_init - 5
    X_init = torch.tensor(X_init, dtype=dtype, device=device)
    
    return X_init

def get_initial_points_high(
    dim: int, 
    n_pts: int, 
    seed: int,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    dtype=torch.float64
):
    np.random.seed(seed)
    X_init = lhs(dim, samples=n_pts, criterion='center')
    # X_init = 10 * X_init - 5
    X_init = torch.tensor(X_init, dtype=dtype, device=device)
    return X_init


'''def get_initial_points_high(
    dim: int, 
    n_pts: int, 
    seed: int,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
    dtype=torch.float64
):
    np.random.seed(seed)
    X_init = lhs(dim, samples=n_pts, criterion='center')
    mean = np.mean(X_init, axis=0)
    std = np.std(X_init, axis=0)
    X_init = (X_init - mean) / std
    X_init = torch.tensor(X_init, dtype=dtype, device=device)
    return X_init
'''
def create_candidate(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [-5, 5]^d
    Y,  # Function values
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float64
,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei")
    # assert X.min() >= -5.0 and X.max() <= 5.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.detach().view(-1)
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length, -1.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length, -1.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=1)

    elif acqf == "ei":
        ei = LogExpectedImprovement(model, Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next


def create_candidate_turbo(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float64
,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei")
    assert X.min() >= -5.0 and X.max() <= 5.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=dtype, device=device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = LogExpectedImprovement(model, Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next


def embedding_matrix(input_dim: int, target_dim: int, seed: int, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float64
) -> torch.Tensor:
    torch.manual_seed(seed)
    if target_dim >= input_dim:
        return torch.eye(input_dim, device=device, dtype=dtype)
    input_dims_perm = (
        torch.randperm(input_dim, device=device) + 1
    )  
    bins = torch.tensor_split(
        input_dims_perm, target_dim
    ) 
    bins = torch.nn.utils.rnn.pad_sequence(
        bins, batch_first=True
    )
    mtrx = torch.zeros(
        (target_dim, input_dim + 1), dtype=dtype, device=device
    )  
    mtrx = mtrx.scatter_(
        1,
        bins,
        2 * torch.randint(2, (target_dim, input_dim), dtype=dtype, device=device) - 1,
    ) 
    return mtrx[:, 1:] 

def embedding_matrix_morenonzero(
    input_dim: int,
    target_dim: int,
    non_zero_per_column: int = 2,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float64

) -> torch.Tensor:
    if target_dim >= input_dim:  
        return torch.eye(input_dim, device=device, dtype=dtype)
    mtrx = torch.zeros((target_dim, input_dim), dtype=dtype, device=device)
    if non_zero_per_column>target_dim:
        non_zero_per_column = target_dim
    
    for col in range(input_dim):
        target_indices = torch.randperm(target_dim, device=device)[:non_zero_per_column]
        random_signs = 2 * torch.randint(2, (non_zero_per_column,), dtype=dtype, device=device) - 1
        for i, idx in enumerate(target_indices):
            mtrx[idx, col] = random_signs[i]

    return mtrx



'''def embedding_matrix_random(input_dim: int, target_dim: int, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float) -> torch.Tensor:
    if (
        target_dim >= input_dim
    ):  # return identity matrix if target size greater than input size
        return torch.eye(input_dim, device=device, dtype=dtype)

    input_dims_perm = (
        torch.randperm(input_dim, device=device) + 1
    )  # add 1 to indices for padding column in matrix

    bins = torch.tensor_split(
        input_dims_perm, target_dim
    )  # split dims into almost equally-sized bins
    bins = torch.nn.utils.rnn.pad_sequence(
        bins, batch_first=True
    )  # zero pad bins, the index 0 will be cut off later

    mtrx = torch.zeros(
        (target_dim, input_dim + 1), dtype=dtype, device=device
    )  
    mtrx = mtrx.scatter_(
        1,
        bins,
        2 * torch.randint(2, (target_dim, input_dim), dtype=dtype, device=device) - 1,
    )  # fill mask with random +/- 1 at indices

    return mtrx[:, 1:]  # cut off index zero as this corresponds to zero padding'''

def embedding_matrix_random(input_dim: int, target_dim: int, seed: int, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float64) -> torch.Tensor:
    torch.manual_seed(seed)
    mtrx = torch.randn(target_dim, input_dim, dtype=dtype, device=device)
    return mtrx



def embedding_matrix_pca(
    input_dim: int,
    target_dim: int,
    data: torch.Tensor,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float64
,
) -> torch.Tensor:
    if target_dim >= input_dim:
        return torch.eye(input_dim, device=device, dtype=dtype)
    data = data.to(device=device, dtype=dtype)
    n_samples = data.shape[0]  # n_samples: Number of data points

    data_mean = data.mean(dim=0, keepdim=True)  # Shape: (1, input_dim)
    data_centered = data - data_mean  # Shape: (n_samples, input_dim)
    covariance_matrix = torch.matmul(data_centered.T, data_centered) / n_samples  # Shape: (input_dim, input_dim)
    
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)  # eigenvectors: (input_dim, input_dim)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_eigenvectors = eigenvectors[:, sorted_indices[:target_dim]]  # Shape: (input_dim, target_dim)
    embedding_matrix = top_eigenvectors.T
    
    return embedding_matrix  



def embedding_matrix_kpca(
    input_dim: int,
    target_dim: int,
    data: torch.Tensor,
    gamma: float = 1.0,  # RBF kernel parameter
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float64
,
) -> torch.Tensor:
    data = data.to(device=device, dtype=dtype)
    n_samples = data.shape[0]

    kernel_matrix = torch.tensor(rbf_kernel(data.cpu().numpy(), gamma=gamma), device=device, dtype=dtype)
    one_n = torch.ones((n_samples, n_samples), device=device, dtype=dtype) / n_samples
    K_centered = kernel_matrix - one_n @ kernel_matrix - kernel_matrix @ one_n + one_n @ kernel_matrix @ one_n
    
    eigenvalues, eigenvectors = torch.linalg.eigh(K_centered)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_eigenvectors = eigenvectors[:, sorted_indices[:target_dim]]  # Shape: (n_samples, target_dim)
    
    embedding_matrix = torch.zeros((target_dim, input_dim), dtype=dtype, device=device)
    for i in range(target_dim):
        eigenvector_column = top_eigenvectors[:, i].unsqueeze(1)  # Shape: (n_samples, 1)        
        projection = torch.matmul(data.T, eigenvector_column)  # Shape: (input_dim, 1)
        embedding_matrix[i, :] = projection.flatten() / torch.norm(projection.flatten())
    return embedding_matrix

'''def embedding_matrix_kpca(
    input_dim: int,
    target_dim: int,
    data: torch.Tensor,
    gamma: float = 1.0,  # RBF kernel parameter
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.float,
) -> torch.Tensor:
    """
    Generate an embedding matrix using Kernel PCA.
    Args:
        input_dim (int): Input dimensionality.
        target_dim (int): Target dimensionality after embedding.
        data (torch.Tensor): Input data, shape (n_samples, target_dim).
        gamma (float): Kernel parameter for RBF kernel.
        device (torch.device): Device to perform computation on (cpu or cuda).
        dtype (torch.dtype): Data type (default is float32).
    
    Returns:
        torch.Tensor: The embedding matrix of shape (target_dim, input_dim).
    """
    if target_dim >= input_dim:
        return torch.eye(input_dim, device=device, dtype=dtype)

    # Ensure the data is on the correct device
    data = data.to(device=device, dtype=dtype)
    n_samples = data.shape[0]
    
    # Compute the RBF kernel matrix
    kernel_matrix = torch.tensor(rbf_kernel(data.cpu().numpy(), gamma=gamma), device=device, dtype=dtype)
    
    # Center the kernel matrix
    one_n = torch.ones((n_samples, n_samples), device=device, dtype=dtype) / n_samples
    K_centered = kernel_matrix - one_n @ kernel_matrix - kernel_matrix @ one_n + one_n @ kernel_matrix @ one_n
    
    # Eigen-decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(K_centered)
    
    # Sort eigenvalues and select top eigenvectors
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    top_eigenvectors = eigenvectors[:, sorted_indices[:target_dim]]  # Shape: (n_samples, target_dim)
    
    # Initialize embedding matrix
    embedding_matrix = torch.zeros((target_dim, input_dim), dtype=dtype, device=device)
    
    # Project eigenvectors to input space
    for i in range(target_dim):
        # Compute the projection of eigenvectors
        projection = torch.matmul(top_eigenvectors[:, i].unsqueeze(1).T, data)  # Shape: (1, input_dim)
        
        # Normalize the projection to fit input dimension
        embedding_matrix[i, :] = projection.flatten() / torch.norm(projection.flatten())
    
    return embedding_matrix'''


def increase_embedding_and_observations(S, X, n_new_bins, seed, device=None, dtype=torch.float64):
    torch.manual_seed(seed)
    S_update = S.clone()
    X_update = X.clone()
    for row_idx in range(len(S)):
        row = S[row_idx]
        idxs_non_zero = torch.nonzero(row)
        idxs_non_zero = idxs_non_zero[torch.randperm(len(idxs_non_zero))].reshape(-1)
        if len(idxs_non_zero) <= 1:
            continue
        non_zero_elements = row[idxs_non_zero].reshape(-1)
        n_row_bins = min(n_new_bins, len(idxs_non_zero))
        new_bins = torch.tensor_split(idxs_non_zero, n_row_bins)[1:]
        elements_to_move = torch.tensor_split(non_zero_elements, n_row_bins)[1:]
        new_bins_padded = torch.nn.utils.rnn.pad_sequence(new_bins, batch_first=True)
        els_to_move_padded = torch.nn.utils.rnn.pad_sequence(elements_to_move, batch_first=True)
        S_stack = torch.zeros((n_row_bins - 1, len(row) + 1), device=device, dtype=dtype)
        S_stack = S_stack.scatter_(1, new_bins_padded + 1, els_to_move_padded)
        S_update[row_idx, torch.hstack(new_bins)] = 0
        X_update = torch.hstack((X_update, X[:, row_idx].reshape(-1, 1).repeat(1, len(new_bins))))
        S_update = torch.vstack((S_update, S_stack[:, 1:]))
    return S_update, X_update


def multi_scale_embedding(input_dim: int, target_dim: int, seed: int, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype=torch.float64) -> torch.Tensor:
    torch.manual_seed(seed)
    scales = target_dim
    step = input_dim // scales
    embedding_matrix = torch.zeros(target_dim, input_dim, device=device, dtype=dtype)
    for i in range(target_dim):
        scale = i % scales + 1
        embedding_matrix[i, (scale - 1) * step : scale * step] = torch.randn(step, device=device)
    return embedding_matrix


