import numpy as np


def polynomial_kernel(x1, x2, c=1., d=3., gamma=1.):
    return (gamma * x1.T @ x2 + c) ** d


def rbf_kernel(x1, x2, gamma=1.0):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * gamma ** 2))


def update_inverse(K_inv, X, x_new, kernel, alpha):
    v = np.array([kernel(xi, x_new) for xi in X]).reshape(-1, 1)
    k_nn = kernel(x_new, x_new) + alpha

    u = K_inv @ v
    alpha = v.T @ u
    beta = 1 / (k_nn - alpha) if k_nn - alpha != 0 else 1e-10

    K_inv_new = np.block([
        [K_inv + beta * (u @ u.T), -beta * u],
        [-beta * u.T, beta]
    ])

    return K_inv_new


class cost_aware_pak_ucb_till_succeed:
    def __init__(self, G: int, T: int, num_dim: int, rd_budget: int, delta=.05,
                 kernel_method='poly', reg_alpha=1., kernel_para_c=1., kernel_para_d=3., kernel_para_gamma=1.,
                 exp_eta=None, cost_aware_greedy=False, cost_para: float = 0., model_cost: list = None, **kwargs):
        self.G = G
        self.T = T
        self.num_dim = num_dim
        self.krr_alpha = reg_alpha
        self.rd_budget = rd_budget
        self.exp_eta = np.sqrt(2. * np.log(2 * G / delta)) if exp_eta is None else exp_eta
        self.krr_inverse_mat = [None for _ in range(G)]
        self.krr_variable = [np.empty((1, num_dim,)) for _ in range(G)]
        self.krr_target = [np.empty((1,)) for _ in range(G)]
        self.visitation = np.zeros((G,), dtype=int)
        self.cost_para = cost_para
        self.cost_aware_greedy = cost_aware_greedy
        self.model_cost = np.array(model_cost, dtype=float)

        self.rd_skip = False
        self.rd_max_reward = 0.
        self.rd_used_budget = 0
        self.rd_ucb_values = None

        self.kernel_method = kernel_method
        if kernel_method == 'lin':
            self.c = 0.
            self.d = 1.
        else:
            self.c = kernel_para_c
            self.d = kernel_para_d
        self.gamma = kernel_para_gamma

    def kernel_function(self, x1, x2):
        if self.kernel_method in ['lin', 'poly']:
            return polynomial_kernel(x1=x1, x2=x2, c=self.c, d=self.d, gamma=self.gamma)
        elif self.kernel_method == 'rbf':
            return rbf_kernel(x1=x1, x2=x2, gamma=self.gamma)
        else:
            raise NotImplementedError

    def select_arm(self, context: np.array):
        assert not self.rd_skip

        if self.rd_used_budget == self.rd_budget or self.rd_max_reward == 1:
            self.rd_skip = True
            return None

        assert context.ndim == 2 and context.shape[0] == 1

        if not np.all(self.visitation):
            return np.random.choice(np.where(self.visitation == 0)[0])

        if self.rd_used_budget == 0:
            self.rd_ucb_values = np.empty((self.G,))
            for g in range(self.G):
                kernel_vector_g = np.empty((self.visitation[g],))
                for n in range(self.visitation[g]):
                    kernel_vector_g[n] = self.kernel_function(x1=context[0], x2=self.krr_variable[g][n])

                mu_g = kernel_vector_g.T @ self.krr_inverse_mat[g] @ self.krr_target[g]
                sig_g = kernel_vector_g.T @ self.krr_inverse_mat[g] @ kernel_vector_g
                sig_g = (self.krr_alpha ** -0.5) * np.sqrt(max(0., self.kernel_function(x1=context[0],
                                                                                        x2=context[0]) - sig_g))
                self.rd_ucb_values[g] = mu_g + self.exp_eta * sig_g
            if self.cost_aware_greedy:
                self.rd_ucb_values[:self.G] += - self.cost_para * self.model_cost[:self.G]
        return np.random.choice((np.where(self.rd_ucb_values == np.max(self.rd_ucb_values)))[0])

    def update_stats(self, g: int, context: np.array, reward: float):

        if self.rd_skip:  # if skip this round, reset within-round statistics
            self.reset_rd_stats()
        else:
            assert context.ndim == 2 and context.shape[0] == 1
            self.rd_used_budget += 1
            self.rd_max_reward = max(self.rd_max_reward, reward)

            if self.visitation[g] == 0:
                self.krr_inverse_mat[g] = np.array(
                    [1. / self.kernel_function(x1=context[0], x2=context[0]) + self.krr_alpha]).reshape((1, 1))
                self.krr_variable[g] = context
                self.krr_target[g][0] = reward
            else:
                self.krr_inverse_mat[g] = update_inverse(K_inv=self.krr_inverse_mat[g],
                                                         X=self.krr_variable[g],
                                                         x_new=context[0],
                                                         kernel=self.kernel_function,
                                                         alpha=self.krr_alpha)
                self.krr_variable[g] = np.concatenate((self.krr_variable[g], context), axis=0)
                self.krr_target[g] = np.concatenate((self.krr_target[g], np.array([reward])), axis=0)

            self.visitation[g] += 1

    def update_model_pool(self):
        self.G = self.G + 1
        self.krr_inverse_mat += [None]
        self.krr_variable += [np.empty((1,))]
        self.krr_target += [np.empty((1,))]
        self.visitation = np.concatenate((self.visitation, np.array([0], dtype=int)))

    def reset_rd_stats(self):
        self.rd_skip = False
        self.rd_max_reward = 0.
        self.rd_used_budget = 0
        self.rd_ucb_values = None
