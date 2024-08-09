import numpy as np
from scipy.optimize import minimize, LinearConstraint

def objective(x, Q, c):
    return 0.5 * np.dot(x, np.dot(Q, x)) + np.dot(c, x)

if False: 
    Q = np.asarray([[1.0, -1.0], [-1.0, 2.0]])
    c = np.asarray([-2.0, -6.0])
    A = np.asarray([
        [-1.0, -1.0],
        [1.0, -2.0],
        [-2.0, -1.0],
        [1.0, 0.0],
        [0.0, 1.0]])
    b = np.asarray([-2.0, -2.0, -3.0, 0.0, 0.0])
else:
    Q = np.array([
        [3.0, -1.0, 0.0, 0.0],
        [-1.0, 4.0, -1.0, 0.0],
        [0.0, -1.0, 5.0, -1.0],
        [0.0, 0.0, -1.0, 6.0]
    ])
    c = np.array([-8.0, -16.0, -4.0, -6.0])

    A = np.array([
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 1.0, -1.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
        [-1.0, 2.0, -1.0, 2.0]
    ])
    b = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 1.0, 1.0, 1.0, 5.0])

x0 = np.zeros(Q.shape[0])

if False: # Ax>=b
    constraints = {'type': 'ineq', 'fun': lambda x: b - np.dot(A, x)}
else: # Ax=b, x>=0
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x},
        {'type': 'ineq', 'fun': lambda x: b - np.dot(A, x)},
        {'type': 'ineq', 'fun': lambda x: np.dot(A, x) - b}]

result = minimize(objective, x0, args=(Q, c), constraints=constraints, method='trust-constr')

print(result.x)
