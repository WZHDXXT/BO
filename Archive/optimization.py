from ioh import logger
import numpy as np
import BA

import torch
import botorch
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP

def optimize_with_baxus(
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
    print('optimize_with_baxus')
    state = BA.BaxusState(dim=dim, eval_budget=EVALUATION_BUDGET
)
    S = BA.embedding_matrix(input_dim=state.dim, target_dim=state.d_init, seed=seed)
    X_baxus_target = BA.get_initial_points(state.d_init, n_init, seed)
    
    X_baxus_input = X_baxus_target @ S
    X_baxus_input = torch.clamp(X_baxus_input, min=-5, max=5)
    Y_baxus = torch.tensor(
            [-np.abs(problem(list(x))-problem.optimum.y) for x in X_baxus_input], dtype=dtype, device=device
        ).unsqueeze(-1)
    # print(Y_baxus)
    with botorch.settings.validate_input_scaling(False):
        for _ in range(EVALUATION_BUDGET - n_init):  # Run until evaluation budget depleted
                # Fit a GP model
                train_Y = (Y_baxus - Y_baxus.mean()) / Y_baxus.std()
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                model = SingleTaskGP(
                    X_baxus_target, train_Y, likelihood=likelihood
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                # Fit the model using Cholesky context
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    try:
                        fit_gpytorch_mll(mll)
                    except ModelFittingError:
                        # Use Adam-based optimization if Cholesky decomposition fails
                        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
                        for _ in range(200):
                            optimizer.zero_grad()
                            output = model(X_baxus_target)
                            loss = -mll(output, train_Y.flatten())
                            loss.backward()
                            optimizer.step()

                    # Generate new candidates
                    X_next_target = BA.create_candidate(
                        state=state,
                        model=model,
                        X=X_baxus_target,
                        Y=train_Y,
                        device=device,
                        dtype=dtype,
                        n_candidates=N_CANDIDATES,
                        num_restarts=NUM_RESTARTS,
                        raw_samples=RAW_SAMPLES,
                        acqf="ts",
                    )
                
                # Map new candidates to high-dimensional space
                X_next_input = X_next_target @ S
                X_next_input = torch.clamp(X_next_input, min=-5, max=5)
                Y_next = torch.tensor(
                    [-np.abs(problem(list(x))-problem.optimum.y) for x in X_next_input], dtype=dtype, device=device
                ).unsqueeze(-1)

                # Update state and concatenate new points
                state = BA.update_state(state=state, Y_next=Y_next)
                X_baxus_input = torch.cat((X_baxus_input, X_next_input), dim=0)
                X_baxus_target = torch.cat((X_baxus_target, X_next_target), dim=0)
                Y_baxus = torch.cat((Y_baxus, Y_next), dim=0)
                
                print(
                    f"iteration {len(X_baxus_input)}, d={len(X_baxus_target.T)})  Best value: {state.best_value:.3}, TR length: {state.length:.3}"
                )

                if state.restart_triggered:
                    state.restart_triggered = False
                    print("increasing target space")

                    S, X_baxus_target = BA.increase_embedding_and_observations(
                        S, X_baxus_target, state.new_bins_on_split, seed
                    )
                    print(f"new dimensionality: {len(S)}")
                    
                    state.target_dim = len(S)
                    state.length = state.length_init
                    state.failure_counter = 0
                    state.success_counter = 0
    return Y_baxus
     
def optimize_with_morenonzero_embedding(
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
    print('optimize_with_morenonzero_embedding')
    state = BA.BaxusState(dim=dim, eval_budget=EVALUATION_BUDGET
)
    S = BA.embedding_matrix_morenonzero(input_dim=state.dim, target_dim=state.d_init)
    X_baxus_target = BA.get_initial_points(state.d_init, n_init, seed)
    X_baxus_input = X_baxus_target @ S
    X_baxus_input = torch.clamp(X_baxus_input, min=-5, max=5)
    Y_baxus = torch.tensor(
            [-np.abs(problem(list(x))-problem.optimum.y) for x in X_baxus_input], dtype=dtype, device=device
        ).unsqueeze(-1)

    with botorch.settings.validate_input_scaling(False):
        for _ in range(EVALUATION_BUDGET - n_init):  # Run until evaluation budget depleted
                # Fit a GP model
                train_Y = (Y_baxus - Y_baxus.mean()) / Y_baxus.std()
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                model = SingleTaskGP(
                    X_baxus_target, train_Y, likelihood=likelihood
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                # Fit the model using Cholesky context
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    try:
                        fit_gpytorch_mll(mll)
                    except ModelFittingError:
                        # Use Adam-based optimization if Cholesky decomposition fails
                        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
                        for _ in range(200):
                            optimizer.zero_grad()
                            output = model(X_baxus_target)
                            loss = -mll(output, train_Y.flatten())
                            loss.backward()
                            optimizer.step()

                    # Generate new candidates
                    X_next_target = BA.create_candidate(
                        state=state,
                        model=model,
                        X=X_baxus_target,
                        Y=train_Y,
                        device=device,
                        dtype=dtype,
                        n_candidates=N_CANDIDATES,
                        num_restarts=NUM_RESTARTS,
                        raw_samples=RAW_SAMPLES,
                        acqf="ts",
                    )
                
                # Map new candidates to high-dimensional space
                X_next_input = X_next_target @ S
                Y_next = torch.tensor(
                    [-np.abs(problem(list(x))-problem.optimum.y) for x in X_next_input], dtype=dtype, device=device
                ).unsqueeze(-1)

                # Update state and concatenate new points
                state = BA.update_state(state=state, Y_next=Y_next)
                X_baxus_input = torch.cat((X_baxus_input, X_next_input), dim=0)
                X_baxus_target = torch.cat((X_baxus_target, X_next_target), dim=0)
                Y_baxus = torch.cat((Y_baxus, Y_next), dim=0)
                
                print(
                    f"iteration {len(X_baxus_input)}, d={len(X_baxus_target.T)})  Best value: {state.best_value:.3}, TR length: {state.length:.3}"
                )

                if state.restart_triggered:
                    state.restart_triggered = False
                    print("increasing target space")

                    S, X_baxus_target = BA.increase_embedding_and_observations(
                        S, X_baxus_target, state.new_bins_on_split, seed
                    )
                    
                    
                    print(f"new dimensionality: {len(S)}")
                        
                    state.target_dim = len(S)
                    state.length = state.length_init
                    state.failure_counter = 0
                    state.success_counter = 0
    return Y_baxus

def optimize_with_random_embedding(
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
    print('optimize_with_random_embedding')
    state = BA.BaxusState(dim=dim, eval_budget=EVALUATION_BUDGET
)
    S = BA.embedding_matrix_random(input_dim=state.dim, target_dim=state.d_init, seed=seed)
    X_baxus_target = BA.get_initial_points(state.d_init, n_init, seed)
    X_baxus_input = X_baxus_target @ S
    X_baxus_input = torch.clamp(X_baxus_input, min=-5, max=5)

    Y_baxus = torch.tensor(
            [-np.abs(problem(list(x))-problem.optimum.y) for x in X_baxus_input], dtype=dtype, device=device
        ).unsqueeze(-1)
    with botorch.settings.validate_input_scaling(False):
        for _ in range(EVALUATION_BUDGET - n_init):  # Run until evaluation budget depleted
                # Fit a GP model
                train_Y = (Y_baxus - Y_baxus.mean()) / Y_baxus.std()
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                model = SingleTaskGP(
                    X_baxus_target, train_Y, likelihood=likelihood
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                # Fit the model using Cholesky context
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    try:
                        fit_gpytorch_mll(mll)
                    except ModelFittingError:
                        # Use Adam-based optimization if Cholesky decomposition fails
                        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
                        for _ in range(200):
                            optimizer.zero_grad()
                            output = model(X_baxus_target)
                            loss = -mll(output, train_Y.flatten())
                            loss.backward()
                            optimizer.step()

                    # Generate new candidates
                    X_next_target = BA.create_candidate(
                        state=state,
                        model=model,
                        X=X_baxus_target,
                        Y=train_Y,
                        device=device,
                        dtype=dtype,
                        n_candidates=N_CANDIDATES,
                        num_restarts=NUM_RESTARTS,
                        raw_samples=RAW_SAMPLES,
                        acqf="ts",
                    )
                
                # Map new candidates to high-dimensional space
                X_next_input = X_next_target @ S
                X_next_input = torch.clamp(X_next_input, min=-5, max=5)

                Y_next = torch.tensor(
                    [-np.abs(problem(list(x))-problem.optimum.y) for x in X_next_input], dtype=dtype, device=device
                ).unsqueeze(-1)

                # Update state and concatenate new points
                state = BA.update_state(state=state, Y_next=Y_next)
                X_baxus_input = torch.cat((X_baxus_input, X_next_input), dim=0)
                X_baxus_target = torch.cat((X_baxus_target, X_next_target), dim=0)
                Y_baxus = torch.cat((Y_baxus, Y_next), dim=0)
                
                print(
                    f"iteration {len(X_baxus_input)}, d={len(X_baxus_target.T)})  Best value: {state.best_value:.3}, TR length: {state.length:.3}"
                )

                if state.restart_triggered:
                    state.restart_triggered = False
                    print("increasing target space")

                    S, X_baxus_target = BA.increase_embedding_and_observations(
                        S, X_baxus_target, state.new_bins_on_split, seed
                    )
                    print(f"new dimensionality: {len(S)}")
                    
                    state.target_dim = len(S)
                    state.length = state.length_init
                    state.failure_counter = 0
                    state.success_counter = 0
    return Y_baxus

def optimize_with_multi_scale_embedding(
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
    print('optimize_with_multi_scale_embedding')

    state = BA.BaxusState(dim=dim, eval_budget=EVALUATION_BUDGET
)
    S = BA.multi_scale_embedding(input_dim=state.dim, target_dim=state.d_init, seed=seed)
    X_baxus_target = BA.get_initial_points(state.d_init, n_init, seed)
    X_baxus_input = X_baxus_target @ S
    X_baxus_input = torch.clamp(X_baxus_input, min=-5, max=5)

    Y_baxus = torch.tensor(
            [-np.abs(problem(list(x))-problem.optimum.y) for x in X_baxus_input], dtype=dtype, device=device
        ).unsqueeze(-1)

    with botorch.settings.validate_input_scaling(False):
        for _ in range(EVALUATION_BUDGET - n_init):  # Run until evaluation budget depleted
                # Fit a GP model
                train_Y = (Y_baxus - Y_baxus.mean()) / Y_baxus.std()
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                model = SingleTaskGP(
                    X_baxus_target, train_Y, likelihood=likelihood
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                # Fit the model using Cholesky context
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    try:
                        fit_gpytorch_mll(mll)
                    except ModelFittingError:
                        # Use Adam-based optimization if Cholesky decomposition fails
                        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
                        for _ in range(200):
                            optimizer.zero_grad()
                            output = model(X_baxus_target)
                            loss = -mll(output, train_Y.flatten())
                            loss.backward()
                            optimizer.step()

                    # Generate new candidates
                    X_next_target = BA.create_candidate(
                        state=state,
                        model=model,
                        X=X_baxus_target,
                        Y=train_Y,
                        device=device,
                        dtype=dtype,
                        n_candidates=N_CANDIDATES,
                        num_restarts=NUM_RESTARTS,
                        raw_samples=RAW_SAMPLES,
                        acqf="ts",
                    )
                
                # Map new candidates to high-dimensional space
                X_next_input = X_next_target @ S
                X_next_input = torch.clamp(X_next_input, min=-5, max=5)

                Y_next = torch.tensor(
                    [-np.abs(problem(list(x))-problem.optimum.y) for x in X_next_input], dtype=dtype, device=device
                ).unsqueeze(-1)

                # Update state and concatenate new points
                state = BA.update_state(state=state, Y_next=Y_next)
                X_baxus_input = torch.cat((X_baxus_input, X_next_input), dim=0)
                X_baxus_target = torch.cat((X_baxus_target, X_next_target), dim=0)
                Y_baxus = torch.cat((Y_baxus, Y_next), dim=0)
                
                print(
                    f"iteration {len(X_baxus_input)}, d={len(X_baxus_target.T)})  Best value: {state.best_value:.3}, TR length: {state.length:.3}"
                )

                if state.restart_triggered:
                    state.restart_triggered = False
                    print("increasing target space")

                    S, X_baxus_target = BA.increase_embedding_and_observations(
                        S, X_baxus_target, state.new_bins_on_split, seed
                    )
                    print(f"new dimensionality: {len(S)}")
                    
                    state.target_dim = len(S)
                    state.length = state.length_init
                    state.failure_counter = 0
                    state.success_counter = 0
    return Y_baxus

def optimize_with_pca_lowdimension_project(
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
    print('optimize_with_pca_lowdimension_project')

    state = BA.BaxusState(dim=dim, eval_budget=EVALUATION_BUDGET)
    n_sample = n_init
    X_sample = BA.get_initial_points_high(state.dim, n_sample, seed)
    X_sample = torch.clamp(X_sample, min=-5, max=5)
    S = BA.embedding_matrix_pca(input_dim=state.dim, target_dim=state.d_init, data=X_sample)
    
    X_baxus_target = BA.get_initial_points(state.d_init, n_init, seed)
    # X_baxus_target = X_sample @ S.T
    # X_baxus_input = X_sample
    X_baxus_input = X_baxus_target @ S
    
    Y_baxus = torch.tensor(
            [-np.abs(problem(list(x))-problem.optimum.y) for x in X_baxus_input], dtype=dtype, device=device
        ).unsqueeze(-1)
    with botorch.settings.validate_input_scaling(False):
        for _ in range(EVALUATION_BUDGET - n_sample):  # Run until evaluation budget depleted
                # Fit a GP model
                train_Y = (Y_baxus - Y_baxus.mean()) / Y_baxus.std()
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                model = SingleTaskGP(
                    X_baxus_target, train_Y, likelihood=likelihood
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                # Fit the model using Cholesky context
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    try:
                        fit_gpytorch_mll(mll)
                    except ModelFittingError:
                        # Use Adam-based optimization if Cholesky decomposition fails
                        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
                        for _ in range(200):
                            optimizer.zero_grad()
                            output = model(X_baxus_target)
                            loss = -mll(output, train_Y.flatten())
                            loss.backward()
                            optimizer.step()

                    # Generate new candidates
                    X_next_target = BA.create_candidate(
                        state=state,
                        model=model,
                        X=X_baxus_target,
                        Y=train_Y,
                        device=device,
                        dtype=dtype,
                        n_candidates=N_CANDIDATES,
                        num_restarts=NUM_RESTARTS,
                        raw_samples=RAW_SAMPLES,
                        acqf="ts",
                    )
                
                # Map new candidates to high-dimensional space
                X_next_input = X_next_target @ S
                X_next_input = torch.clamp(X_next_input, min=-5, max=5)

                Y_next = torch.tensor(
                    [-np.abs(problem(list(x))-problem.optimum.y) for x in X_next_input], dtype=dtype, device=device
                ).unsqueeze(-1)

                # Update state and concatenate new points
                state = BA.update_state(state=state, Y_next=Y_next)
                X_baxus_input = torch.cat((X_baxus_input, X_next_input), dim=0)
                X_baxus_target = torch.cat((X_baxus_target, X_next_target), dim=0)
                Y_baxus = torch.cat((Y_baxus, Y_next), dim=0)
                
                print(
                    f"iteration {len(X_baxus_input)}, d={len(X_baxus_target.T)})  Best value: {state.best_value:.3}, TR length: {state.length:.3}"
                )

                if state.restart_triggered:
                    state.restart_triggered = False
                    print("increasing target space")

                    S, X_baxus_target = BA.increase_embedding_and_observations(
                        S, X_baxus_target, state.new_bins_on_split, seed
                    )
                    print(f"new dimensionality: {len(S)}")
                    
                    state.target_dim = len(S)
                    state.length = state.length_init
                    state.failure_counter = 0
                    state.success_counter = 0
    return Y_baxus


def optimize_with_kpca_lowdimension_project(
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
    print('optimize_with_kpca_lowdimension_project')
    state = BA.BaxusState(dim=dim, eval_budget=EVALUATION_BUDGET
)
    n_sample = n_init
    X_sample = BA.get_initial_points_high(state.dim, n_sample, seed)
    X_sample = torch.clamp(X_sample, min=-5, max=5)
    S = BA.embedding_matrix_kpca(input_dim=state.dim, target_dim=state.d_init, data=X_sample)
    X_baxus_target = BA.get_initial_points(state.d_init, n_init, seed)
    # X_baxus_target = X_sample @ S.T
    # X_baxus_input = X_sample
    X_baxus_input = X_baxus_target @ S
    Y_baxus = torch.tensor(
            [-np.abs(problem(list(x))-problem.optimum.y) for x in X_baxus_input], dtype=dtype, device=device
        ).unsqueeze(-1)
    with botorch.settings.validate_input_scaling(False):
        for _ in range(EVALUATION_BUDGET - n_sample):  # Run until evaluation budget depleted
                # Fit a GP model
                train_Y = (Y_baxus - Y_baxus.mean()) / Y_baxus.std()
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
                model = SingleTaskGP(
                    X_baxus_target, train_Y, likelihood=likelihood
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)

                # Fit the model using Cholesky context
                with gpytorch.settings.max_cholesky_size(max_cholesky_size):
                    try:
                        fit_gpytorch_mll(mll)
                    except ModelFittingError:
                        # Use Adam-based optimization if Cholesky decomposition fails
                        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)
                        for _ in range(200):
                            optimizer.zero_grad()
                            output = model(X_baxus_target)
                            loss = -mll(output, train_Y.flatten())
                            loss.backward()
                            optimizer.step()

                    # Generate new candidates
                    X_next_target = BA.create_candidate(
                        state=state,
                        model=model,
                        X=X_baxus_target,
                        Y=train_Y,
                        device=device,
                        dtype=dtype,
                        n_candidates=N_CANDIDATES,
                        num_restarts=NUM_RESTARTS,
                        raw_samples=RAW_SAMPLES,
                        acqf="ts",
                    )
                
                # Map new candidates to high-dimensional space
                X_next_input = X_next_target @ S
                X_next_input = torch.clamp(X_next_input, min=-5, max=5)
                Y_next = torch.tensor(
                    [-np.abs(problem(list(x))-problem.optimum.y) for x in X_next_input], dtype=dtype, device=device
                ).unsqueeze(-1)

                # Update state and concatenate new points
                state = BA.update_state(state=state, Y_next=Y_next)
                X_baxus_input = torch.cat((X_baxus_input, X_next_input), dim=0)
                X_baxus_target = torch.cat((X_baxus_target, X_next_target), dim=0)
                Y_baxus = torch.cat((Y_baxus, Y_next), dim=0)
                
                print(
                    f"iteration {len(X_baxus_input)}, d={len(X_baxus_target.T)})  Best value: {state.best_value:.3}, TR length: {state.length:.3}"
                )

                if state.restart_triggered:
                    state.restart_triggered = False
                    print("increasing target space")

                    S, X_baxus_target = BA.increase_embedding_and_observations(
                        S, X_baxus_target, state.new_bins_on_split, seed
                    )
                    print(f"new dimensionality: {len(S)}")
                    
                    state.target_dim = len(S)
                    state.length = state.length_init
                    state.failure_counter = 0
                    state.success_counter = 0
    return Y_baxus
    