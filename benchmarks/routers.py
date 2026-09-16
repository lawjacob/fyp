"""Linear PromptWise with per-arm observation-clock exponential forgetting.

The original selection rule, quadratic uncertainty bonus, and Gram prior are
preserved. This is a weighted logistic likelihood, not a scalar EWMA estimator.
"""
import numpy as np

from algorithms.promptwise import promptwise, solve_lin_logistic_regression


class DiscountedPromptWise(promptwise):
    def __init__(self, *args, discount=0.98, gram_ridge=1e6, **kwargs):
        if not 0 < discount <= 1 or gram_ridge <= 0:
            raise ValueError('discount must be in (0, 1], gram_ridge must be positive')
        super().__init__(*args, **kwargs)
        if self.reg_method != 'mle' or self.per_step_update:
            raise ValueError('Discounting currently supports per-observation linear MLE only')
        self.discount = discount
        self.gram_ridge = gram_ridge
        self.grams = [gram_ridge * np.eye(self.num_dim) for _ in range(self.G)]
        self.inv_Gram_mat = [np.eye(self.num_dim) / gram_ridge for _ in range(self.G)]

    def fit_reg_model(self, g):
        n = len(self.reg_target[g])
        weights = self.discount ** np.arange(n - 1, -1, -1)
        self.reg_model[g] = solve_lin_logistic_regression(
            self.reg_variable[g], self.reg_target[g], sample_weight=weights)

    def update_stats(self, g, context, reward):
        if self.rd_skip:
            return super().update_stats(g, context, reward)
        # Install the discounted old Gram before the original rank-one update.
        self.grams[g] = (self.discount * self.grams[g]
                         + (1 - self.discount) * self.gram_ridge * np.eye(self.num_dim))
        self.inv_Gram_mat[g] = np.linalg.inv(self.grams[g])
        super().update_stats(g, context, reward)
        self.grams[g] += context.T @ context


def scores(router, context):
    """Separate mean and exploration bonus; do not expose unchosen outcomes."""
    means, bonuses, optimistic = [], [], []
    for g in range(router.G):
        if router.reg_model[g] is None:
            means.append(None)
            bonuses.append(None)
            optimistic.append(None)
        else:
            logit = float((router.reg_model[g] @ context.T).item())
            bonus = float((router.exp_para * context @ router.inv_Gram_mat[g] @ context.T).item())
            means.append(float(1 / (1 + np.exp(-np.clip(logit, -500, 500)))))
            bonuses.append(bonus)
            optimistic.append(float(np.asarray(router.predict(g, context)).item()))
    return {'predicted_success': means, 'logit_bonus': bonuses, 'optimistic_success': optimistic}
