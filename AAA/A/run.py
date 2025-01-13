import matplotlib.pyplot as plt
import numpy as np
import torch
from ioh import get_problem, ProblemClass
import argparse


def run_all_optimizations(
    problem,
    dim,
    n_init,
    EVALUATION_BUDGET,
    max_cholesky_size,
    N_CANDIDATES,
    NUM_RESTARTS,
    RAW_SAMPLES,
    device,
    dtype,
    seed
):
    from optimization import (
        optimize_with_baxus,
        optimize_with_random_embedding,
        optimize_with_multi_scale_embedding,
        optimize_with_morenonzero_embedding,
        optimize_with_pca_lowdimension_project,
        optimize_with_kpca_lowdimension_project,
    )
    
    Y_baxus = optimize_with_baxus(
        problem, dim, n_init, EVALUATION_BUDGET, max_cholesky_size, N_CANDIDATES, NUM_RESTARTS, RAW_SAMPLES, device, dtype, seed
    )
    Y_random_embedding = optimize_with_random_embedding(
        problem, dim, n_init, EVALUATION_BUDGET, max_cholesky_size, N_CANDIDATES, NUM_RESTARTS, RAW_SAMPLES, device, dtype, seed
    )
    Y_multi_scaling = optimize_with_multi_scale_embedding(
        problem, dim, n_init, EVALUATION_BUDGET, max_cholesky_size, N_CANDIDATES, NUM_RESTARTS, RAW_SAMPLES, device, dtype, seed
    )
    Y_non_zero = optimize_with_morenonzero_embedding(
        problem, dim, n_init, EVALUATION_BUDGET, max_cholesky_size, N_CANDIDATES, NUM_RESTARTS, RAW_SAMPLES, device, dtype, seed
    )

    if dim >= 3:
        Y_pca_project = optimize_with_pca_lowdimension_project(
            problem, dim, n_init, EVALUATION_BUDGET, max_cholesky_size, N_CANDIDATES, NUM_RESTARTS, RAW_SAMPLES, device, dtype, seed
        )
        Y_kpca_project = optimize_with_kpca_lowdimension_project(
            problem, dim, n_init, EVALUATION_BUDGET, max_cholesky_size, N_CANDIDATES, NUM_RESTARTS, RAW_SAMPLES, device, dtype, seed
        )
        return [Y_baxus, Y_random_embedding, Y_multi_scaling, Y_non_zero,
                Y_pca_project, Y_kpca_project]
    else:
        return [Y_baxus, Y_random_embedding, Y_multi_scaling, Y_non_zero]


def plot_optimization_comparison(names, runs):
    fig, ax = plt.subplots(figsize=(5, 4))

    for name, run in zip(names, runs): 
        fx = np.minimum.accumulate(run.cpu())
        plt.plot(fx, label=name)

    plt.xlabel("Number of evaluations", fontsize=8) 
    plt.ylabel("Objective value", fontsize=8)
    plt.xlim([0, len(runs[0])-1])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1) 
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.3), 
        ncol=3, 
        fontsize=8, 
    )
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)  
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dim", type=int, default=10
    )
    parser.add_argument(
        "--problem_num",
        type=int,
        default="21",
    )
    # parser.add_argument("--instance_num", type=int, default=2)
    # parser.add_argument("--evaluation_budget", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    Y_baxus_alls = []
    args = parse_args()
    dim = args.dim
    problem_num = args.problem_num
    # instance_num = args.instance_num
    # evaluation_budget = args.evaluation_budget
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    dtype = torch.float64
    # initial sample points
    n_init = dim 
    max_cholesky_size = float("inf")
    
    # EVALUATION_BUDGET = 10*dim + evaluation_budget 
    EVALUATION_BUDGET = 10*dim
    NUM_RESTARTS = 3 
    RAW_SAMPLES = 51 
    N_CANDIDATES = min(dim, max(200, 20 * dim)) 
    for i in range(3):
        result_for_this_instance = []
        problem = get_problem(problem_num, dimension=dim, instance=i, problem_class=ProblemClass.BBOB)
        for j in range(5):
            Y_baxus_alls.append(run_all_optimizations(
                    problem,
                    dim,
                    n_init,
                    EVALUATION_BUDGET,
                    max_cholesky_size,
                    N_CANDIDATES,
                    NUM_RESTARTS,
                    RAW_SAMPLES,
                    device,
                    dtype,
                    j
                ))
    Y_baxus_alls = list(zip(*Y_baxus_alls)) 
    Y_baxus_all = [torch.stack(strategy).mean(dim=0) for strategy in Y_baxus_alls]
    
    
    all_names = ['BAxUs', 'random_embedding', 'multi-scale_embedding', 'multi_nonzero_embedding', 
             'pca_based_embedding', 'kpca_based_embedding']

    if dim < 3:
        names = all_names[:4]
        runs = [-Y_baxus_all[0], -Y_baxus_all[1], -Y_baxus_all[2], -Y_baxus_all[3]]
    else:
        names = all_names
        runs = [-result for result in Y_baxus_all]

    plot_optimization_comparison(names, runs)

