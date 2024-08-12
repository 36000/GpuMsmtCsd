import numpy as np
from scipy.optimize import minimize
from numba_cls import prep_problem, batch_constrained_ls_fit

def constrained_least_squares(R, d, A, b):
    """
    Solves the constrained least squares problem:
    minimize 1/2 ||Rx - d||^2 subject to Ax >= b
    
    Parameters:
        R (np.ndarray): Matrix R
        d (np.ndarray): Vector d
        A (np.ndarray): Matrix A
        b (np.ndarray): Vector b
        
    Returns:
        np.ndarray: Solution vector x
    """
    # Define the objective function
    def objective(x):
        return 0.5 * np.linalg.norm(R @ x - d)**2
    
    # Define the constraint in the form Ax >= b
    constraints = {'type': 'ineq', 'fun': lambda x: A @ x - b}

    # Initial guess for the variables
    x0 = np.zeros(R.shape[1])

    # Solve the problem
    result = minimize(objective, x0, constraints=constraints, method='trust-constr')

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed: " + result.message)

def test_problems():
    # Define a set of test problems
    problems = [
        {
            "R": np.array([[1, 2], [3, 4], [5, 6]]),
            "d": np.array([1, 2, 3]),
            "A": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
            "b": np.array([0, 0, -1, -1])
        },
        {
            "R": np.array([[2, 1], [1, 3], [3, 5]]),
            "d": np.array([2, 1, 3]),
            "A": np.array([[1, 1], [-1, 0], [0, -1]]),
            "b": np.array([1, -0.5, -0.5])
        },
        {
            "R": np.array([[3, 1], [4, 2], [1, 3]]),
            "d": np.array([3, 4, 1]),
            "A": np.array([[0, 1], [1, -1], [-1, 1]]),
            "b": np.array([0, -1, -1])
        },
        {
            "R": np.array([[1, 1], [2, 2], [3, 3]]),
            "d": np.array([1, 1, 1]),
            "A": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
            "b": np.array([0, 0, -1, -1])
        },
        {
            "R": np.array([[1, 0], [0, 1], [1, 1]]),
            "d": np.array([0, 0, 1]),
            "A": np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]]),
            "b": np.array([1, -1, 0, 0])
        }
    ]

    for i, problem in enumerate(problems):
        R, d, A, b = problem["R"].astype(np.float64), problem["d"].astype(np.float64), problem["A"].astype(np.float64), problem["b"].astype(np.float64)

        # Solve the constrained least squares problem
        x = constrained_least_squares(R, d, A, b)

        # Print the results
        print("scipy solution x:", x)
        print("A@x-b:", A@x-b)
        print("ls:", 1/2.*np.linalg.norm(R@x-d))

        coeff = np.zeros((1, 1, 1, R.shape[1]))

        Rt, R_pinv, N_inv = prep_problem(R, A)

        x = batch_constrained_ls_fit[
            (1, 1, 1), 1,
            0, A.shape[1]*8](
                Rt, R_pinv, N_inv, A, b, np.asarray([[[d]]]), coeff)
        x = coeff[0, 0, 0, :]
        print("Our solution x:", x)
        print("A@x-b:", A@x-b)
        print("ls:", 1/2.*np.linalg.norm(R@x-d))

# Run the test
test_problems()
