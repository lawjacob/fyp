import numpy as np


class random_selector:
    def __init__(self, G, **kwargs):
        self.G = G
        self.rd_skip = True

    def select_arm(self, **kwargs):
        return np.random.randint(self.G)

    def update_stats(self, **kwargs):
        pass

    def bonus(self, **kwargs):
        pass

    def update_model_pool(self):
        self.G = self.G + 1


class lowest_cost_selector:
    def __init__(self, G, model_cost, **kwargs):

        self.G = G
        self.model_cost = np.array(model_cost, dtype=float)
        self.rd_skip = True

    def select_arm(self, **kwargs):
        return np.random.choice(np.where(self.model_cost[:self.G] == np.min(self.model_cost[:self.G]))[0])

    def update_stats(self, **kwargs):
        pass

    def update_model_pool(self):
        self.G = self.G + 1


class highest_cost_selector:
    def __init__(self, G, model_cost, **kwargs):

        self.G = G
        self.model_cost = np.array(model_cost, dtype=float)
        self.rd_skip = True

    def select_arm(self, **kwargs):
        return np.random.choice(np.where(self.model_cost[:self.G] == np.max(self.model_cost[:self.G]))[0])

    def update_stats(self, **kwargs):
        pass

    def update_model_pool(self):
        self.G = self.G + 1


class random_till_succeed:
    def __init__(self, G, rd_budget, **kwargs):
        self.G = G
        self.rd_budget = rd_budget
        self.rd_max_reward = -np.inf
        self.rd_skip = False
        self.rd_used_budget = 0

    def select_arm(self, **kwargs):
        if self.rd_used_budget == self.rd_budget or self.rd_max_reward == 1:
            self.rd_skip = True
            return None
        return np.random.randint(self.G)

    def update_stats(self, reward, **kwargs):
        if self.rd_skip:
            self.reset_rd_stats()
        else:
            self.rd_max_reward = max(self.rd_max_reward, reward)
            self.rd_used_budget += 1

    def reset_rd_stats(self):
        self.rd_skip = False
        self.rd_used_budget = 0
        self.rd_max_reward = -np.inf

    def update_model_pool(self):
        self.G = self.G + 1


class greedy_selector:
    def __init__(self, G, cost_para, model_cost, **kwargs):
        self.G = G
        self.cost_para = cost_para
        self.model_cost = np.array(model_cost, dtype=float)
        self.expected_reward = + np.inf * np.ones((G,))
        self.visitation = np.zeros((G,), dtype=int)
        self.rd_skip = True

    def select_arm(self, **kwargs):
        v = self.expected_reward - self.cost_para * self.model_cost[:self.G]
        return np.random.choice(np.where(v == np.max(v))[0])

    def update_stats(self, g, reward, **kwargs):
        if g is not None:
            if self.visitation[g] == 0:
                self.expected_reward[g] = reward
                self.visitation[g] = 1
            else:
                self.expected_reward[g] = (self.expected_reward[g] * self.visitation[g] + reward) / \
                                          (self.visitation[g] + 1)
                self.visitation[g] += 1

    def reset_rd_stats(self):
        pass

    def update_model_pool(self):
        self.G = self.G + 1
        self.expected_reward = np.concatenate((self.expected_reward, np.array([+ np.inf], dtype=float)))
        self.visitation = np.concatenate((self.visitation, np.zeros((1,), dtype=int)))


class greedy_till_succeed(greedy_selector):
    def __init__(self, G, cost_para, model_cost, rd_budget, **kwargs):
        super().__init__(G=G, cost_para=cost_para, model_cost=model_cost)
        self.rd_skip = False
        self.rd_budget = rd_budget
        self.rd_used_budget = 0
        self.rd_max_reward = -np.inf

    def select_arm(self, **kwargs):
        if self.rd_used_budget == self.rd_budget or self.rd_max_reward == 1:
            self.rd_skip = True
            return None
        v = self.expected_reward - self.cost_para * self.model_cost[:self.G]
        return np.random.choice(np.where(v == np.max(v))[0])

    def update_stats(self, g, reward, **kwargs):
        if g is not None:
            self.rd_used_budget += 1
            self.rd_max_reward = max(self.rd_max_reward, reward)
            if self.visitation[g] == 0:
                self.expected_reward[g] = reward
                self.visitation[g] = 1
            else:
                self.expected_reward[g] = (self.expected_reward[g] * self.visitation[g] + reward) / \
                                          (self.visitation[g] + 1)
                self.visitation[g] += 1
        else:
            self.reset_rd_stats()

    def reset_rd_stats(self):
        self.rd_used_budget = 0
        self.rd_max_reward = -np.inf
        self.rd_skip = False
