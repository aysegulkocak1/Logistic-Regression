import numpy as np

#Calculate the first derivative of likelihood function given output (y) , input (x) and p (estimated probability)
def calculate_first_deriv(y, x, p):
    deriv = y * x - p * x
    result = np.sum(deriv)
    return result

#Calculate the likelihood function given output(y) and p
def calculate_likelihood(y, p):
    lf = 1
    for i in range(len(y)):
        if y[i] == 1:
            lf *= p[i]
        else:
            lf *= (1 - p[i])
    return lf

#Calculate the value of p (predictions on each observation) given x_new(input) and estimated ws
def find_p(x_new, w):
    expon = np.sum(x_new * w, axis=1)
    p = np.exp(expon) / (1 + np.exp(expon))
    return p

#Calculate the matrix W with all diagonal values as p 
def find_W(p):
    W = np.zeros((len(p), len(p)))
    np.fill_diagonal(W, p * (1 - p))
    return W

#Calculate hessian matrix 
def calculate_hessian_matrix(x, W):
    hessian = np.zeros((x.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            hessian[i, j] = np.sum(W * x[:, i] * x[:, j])
    return hessian

def logistic_regression(x, y, learning_rate=0.01, tolerance=1e-9, max_iteration=1000):
    iteration = 0
    diff = float('inf')
    
    x_new = np.column_stack((np.ones(len(y)), x))  # Add bias term to x
    w = np.zeros(x_new.shape[1])  # Initialize w vector 

    while diff > tolerance and iteration < max_iteration:
        p = find_p(x_new, w)
        W = find_W(p)
        h = calculate_hessian_matrix(x_new, W)
        
        # Calculate the first derivative
        first_derivative = np.sum(x_new * (y - p)[:, np.newaxis], axis=0)
        
        # Update beta using the inverse of the Hessian matrix
        try:
            hessian_inv = np.linalg.inv(h)
        except np.linalg.LinAlgError:
            print("Hessian matrix is singular and cannot be inverted.")
            break

        w_new = w + learning_rate*( np.dot(hessian_inv, first_derivative))
        
        # Calculate the log-likelihood
        lf = calculate_likelihood(y, p)
        print(f"Iteration {iteration+1}: Likelihood = {lf}")

        # Calculate the difference for convergence criteria
        diff = np.sum(((w_new - w)/learning_rate)**2)
        w = w_new
        iteration += 1
    
    return w

x = np.array(range(1, 11)).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1])
weights = logistic_regression(x, y)
print("Estimated coefficients:", weights)

# Calculate log-likelihood for the final model
x_new = np.column_stack((np.ones(len(y)), x))
final_p = find_p(x_new, weights)
final_ll = calculate_likelihood(y, final_p)
print("Final Likelihood =", final_ll)
