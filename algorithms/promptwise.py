import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel


def update_inverse_lin(V_inv, x):
    x = x.reshape(-1, 1)
    numerator = V_inv @ x @ x.T @ V_inv
    denominator = 1 + x.T @ V_inv @ x
    V_inv_new = V_inv - numerator / denominator
    return V_inv_new


def solve_lin_logistic_regression(X, R, initial_theta=None, method='L-BFGS-B', sample_weight=None):
    n_samples, n_features = X.shape
    weights = np.ones(n_samples) if sample_weight is None else np.asarray(sample_weight)
    if weights.shape != (n_samples,) or np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError('sample_weight must contain one finite nonnegative weight per row')

    # Initialize θ if not provided
    if initial_theta is None:
        initial_theta = np.ones(n_features)

    # Define the objective function (negative log-likelihood)
    def objective(theta):
        scores = X @ theta

        # Log with stability - use logaddexp for better numerical stability
        log_likelihood = np.sum(
            weights * (R * -np.logaddexp(0, -scores) +
            (1 - R) * -np.logaddexp(0, scores))
        )

        # Add L2 regularization to prevent overfitting and improve stability
        reg_penalty = 0.5 * np.sum(theta ** 2)  # Don't regularize bias term too much

        return -log_likelihood + reg_penalty

    def gradient(theta):
        predictions = sigmoid(X @ theta)
        grad = X.T @ (weights * (predictions - R))

        # Add regularization gradient (excluding bias term)
        reg_grad = theta
        grad += reg_grad

        return grad

    # Solve the optimization problem
    result = minimize(objective, initial_theta, method=method, jac=gradient, tol=1e-4, options={'maxiter': 300})

    return result.x


def update_inverse_kernel(K_inv, X, x_new, kernel, gamma, alpha):
    v = np.array([np.exp(-np.linalg.norm(xi - x_new) ** 2 / (2 * gamma ** 2)) for xi in X]).reshape(-1, 1)
    k_nn = kernel(x_new, x_new) + alpha

    u = K_inv @ v
    alpha = v.T @ u
    beta = 1 / (k_nn - alpha) if k_nn - alpha != 0 else 1e-10

    K_inv_new = np.block([
        [K_inv + beta * (u @ u.T), -beta * u],
        [-beta * u.T, beta]
    ])

    return K_inv_new


class KernelLogisticRegression:
    def __init__(self, kernel='rbf', gamma=1.0, degree=3, beta=1.0, lr=0.01):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.beta = beta
        self.lr = lr
        self.alpha_weights = None
        self.X_train = None

    def _kernel_function(self, X1, X2):
        if self.kernel == 'rbf':
            return rbf_kernel(X1, X2, gamma=self.gamma)
        elif self.kernel == 'poly':
            return np.power(X1 @ X2.T + 1, self.degree)
        else:
            raise ValueError("Unsupported kernel. Use 'rbf' or 'poly'.")

    def _objective_function(self, beta_weights, K, y):
        scores = K @ beta_weights
        predictions = sigmoid(scores)

        loss = -np.sum(y * np.log(predictions + 1e-15) +
                       (1 - y) * np.log(1 - predictions + 1e-15))
        loss += self.beta * beta_weights.T @ beta_weights

        gradient = K.T @ (predictions - y) + 2 * self.beta * beta_weights

        return loss, gradient

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X_train = X
        self.alpha_weights = np.zeros(n_samples)

        K = self._kernel_function(X, X)

        res = minimize(
            fun=self._objective_function,
            x0=self.alpha_weights,
            args=(K, y),
            method='L-BFGS-B',
            jac=True,
            tol=1e-6,
        )

        self.alpha_weights = res.x

    def predict_proba(self, X, bonus=0.):
        K = self._kernel_function(X, self.X_train)
        scores = K @ self.alpha_weights
        return sigmoid(scores + bonus)


# Sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class promptwise:
    def __init__(self, G, num_dim, rd_budget, model_cost, cost_para,
                 exp_para=None, delta=0.05, tau_exp=5,
                 reg_method='mle', kernel_method='lin', kernel_para_gamma=1.,
                 per_step_update=False, **kwargs):
        self.G = G
        self.num_dim = num_dim

        self.model_cost = np.array(model_cost, dtype=float)
        self.cost_para = cost_para

        assert reg_method in ['klr', 'mle']
        self.reg_method = reg_method
        self.kernel_method = kernel_method
        self.kernel_gamma = kernel_para_gamma
        self.reg_variable = [np.empty((1, num_dim,)) for _ in range(G)]
        self.reg_target = [np.empty((1,)) for _ in range(G)]
        self.reg_model = [None for _ in range(G)]
        if reg_method == 'klr':
            self.reg_model = [KernelLogisticRegression(kernel='rbf', gamma=kernel_para_gamma,
                                                       beta=1.0) for _ in range(G)]
            self.krr_inverse_mat = [None for _ in range(G)]
        self.inv_Gram_mat = [1e-6 * np.eye(num_dim) for _ in range(G)]

        self.per_step_update = per_step_update
        self.exp_para = exp_para if exp_para is not None else np.sqrt(2. * np.log(2. * G / delta))
        self.tau_exp = tau_exp
        self.visitation = np.zeros((G,), dtype=int)

        self.rd_budget = rd_budget
        self.rd_skip = False    # whether moving on to the next prompt
        self.rd_max_reward = 0.
        self.rd_cum_cost = 0.
        self.rd_used_budget = 0

        self.rd_ucb_q = None
        self.rd_ucb_u = None
        self.rd_action_seq = []

    def select_arm(self, context: np.array):
        assert not self.rd_skip

        if self.rd_used_budget == self.rd_budget or self.rd_max_reward == 1 or \
        (self.rd_used_budget == 1 and not np.all(self.visitation >= self.tau_exp)):
            self.rd_skip = True
            return None

        if not np.all(self.visitation >= self.tau_exp):     # exploration phase
            return np.random.choice(np.where(self.visitation < self.tau_exp)[0])

        assert context.ndim == 2 and context.shape[0] == 1
        if self.rd_ucb_q is None:    # at the first round of the step
            self.rd_ucb_q = np.empty((self.G,))
            self.rd_ucb_u = np.empty((self.G,))
            for g in range(self.G):
                self.rd_ucb_q[g] = np.asarray(self.predict(g=g, context=context)).item()
                self.rd_ucb_u[g] = 1. - self.cost_para * self.model_cost[g] / self.rd_ucb_q[g]

        if np.max(self.rd_ucb_u) < 0:
            self.rd_skip = True
            return None
        else:
            return np.random.choice(np.where(self.rd_ucb_u == np.max(self.rd_ucb_u))[0])

    def kernel_function(self, x1, x2):
        if self.kernel_method == 'rbf':
            return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.kernel_gamma ** 2))
        else:
            raise NotImplementedError

    def update_stats(self, g: int, context: np.array, reward: float):

        if self.rd_skip:  # End of a step
            if self.per_step_update:   # Per-step update
                for g in self.rd_action_seq:
                    self.fit_reg_model(g=g)

            self.reset_rd_stats()

        else:   # Within a step
            self.rd_action_seq.append(g)
            self.rd_max_reward = max(self.rd_max_reward, reward)
            self.rd_cum_cost += self.model_cost[g]
            self.rd_used_budget += 1

            assert context.ndim == 2 and context.shape[0] == 1

            # Update regression dataset
            self.inv_Gram_mat[g] = update_inverse_lin(V_inv=self.inv_Gram_mat[g], x=context)
            if self.visitation[g] == 0:
                if self.reg_method == 'klr':
                    self.krr_inverse_mat[g] = np.array(
                        [1. / self.kernel_function(x1=context[0], x2=context[0]) + self.reg_model[g].beta]).reshape((1, 1))
                self.reg_variable[g] = context
                self.reg_target[g][0] = int(reward)
            else:
                if self.reg_method == 'klr':
                    self.krr_inverse_mat[g] = update_inverse_kernel(K_inv=self.krr_inverse_mat[g],
                                                                    X=self.reg_variable[g],
                                                                    x_new=context[0],
                                                                    kernel=self.kernel_function,
                                                                    alpha=self.reg_model[g].beta,
                                                                    gamma=self.kernel_gamma)
                self.reg_variable[g] = np.concatenate((self.reg_variable[g], context), axis=0)
                self.reg_target[g] = np.concatenate((self.reg_target[g], np.array([reward])),
                                                    axis=0)

            if self.visitation[g] == self.tau_exp - 1:  # first update after exploration phase
                self.fit_reg_model(g=g)
            self.visitation[g] += 1

            if not self.per_step_update and self.visitation[g] > self.tau_exp:     # not in the exploration phase
                self.fit_reg_model(g=g)
                if self.rd_ucb_q is not None:
                    self.rd_ucb_q[g] = np.asarray(self.predict(g=g, context=context)).item()
                    self.rd_ucb_u[g] = 1. - self.cost_para * self.model_cost[g] / self.rd_ucb_q[g]

    def reset_rd_stats(self):
        self.rd_skip = False
        self.rd_max_reward = 0.
        self.rd_cum_cost = 0.
        self.rd_used_budget = 0
        self.rd_ucb_q = None
        self.rd_ucb_u = None
        self.rd_action_seq = []

    def update_model_pool(self, k=1):
        self.G = self.G + k
        self.reg_variable += [np.empty((1, self.num_dim,))] * k
        self.reg_target += [np.empty((1,))] * k
        if self.reg_method == 'klr':
            self.reg_model += [KernelLogisticRegression(kernel='rbf', gamma=self.kernel_gamma,
                                                        beta=1.0)] * k
            self.krr_inverse_mat += [None] * k
        else:
            self.reg_model += [None] * k
        self.inv_Gram_mat += [1e-6 * np.eye(self.num_dim)] * k
        self.visitation = np.concatenate((self.visitation, np.zeros((k,), dtype=int)))

    def predict(self, g, context):
        if self.reg_method == 'mle':
            return sigmoid(self.reg_model[g] @ context.T + self.exp_para * context @ self.inv_Gram_mat[g] @ context.T)
        elif self.reg_method == 'klr':
            kernel_vector_g = np.empty((self.reg_variable[g].shape[0],))
            for n in range(self.visitation[g]):
                kernel_vector_g[n] = self.kernel_function(x1=context[0], x2=self.reg_variable[g][n])
            sig_g = kernel_vector_g.T @ self.krr_inverse_mat[g] @ kernel_vector_g
            sig_g = self.exp_para * (self.reg_model[g].beta ** -0.5) * np.sqrt(max(0., self.kernel_function(x1=context[0],
                                                                                                            x2=context[0]) - sig_g))
            return self.reg_model[g].predict_proba(context, bonus=self.exp_para * sig_g)

    def fit_reg_model(self, g):
        if self.reg_method == 'mle':
            self.reg_model[g] = solve_lin_logistic_regression(X=self.reg_variable[g], R=self.reg_target[g])
        elif self.reg_method == 'klr':
            self.reg_model[g].fit(X=self.reg_variable[g], y=self.reg_target[g])
