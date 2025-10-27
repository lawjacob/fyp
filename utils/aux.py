import numpy as np


MODEL_COST_PER_1M_TOKENS = {
    'gpt-4o_openai-api': 2.5 + 10.0,
    'gpt-4o_openrouter-api': 2.5 + 10.0,
    'gpt-4o-mini_openai-api': 0.15 + 0.6,
    'gpt-4o-mini_openrouter-api': 0.15 + 0.6,
    'o3-mini_openai-api': 1.1 + 4.4,
    'gpt-4.1_openrouter-api': 2.0 + 8.0,
    'gemini-2.5-flash-preview_openrouter-api': 0.15 + 0.6,
    'gemini-2.0-flash_openrouter-api': 0.1 + 0.4,
    'claude-3.5-haiku_openrouter-api': 0.8 + 4.0,
    'claude-3.7-sonnet_openrouter-api': 3.0 + 15.0,
    'deepseek-chat_deepseek-api': 0.27 + 1.1,
    'qwen-plus_qwen-api': 0.4 + 1.2,
}


class cost_aware_env:
    def __init__(self, G: int, task: str, num_epoch: int, T: int,
                 model_set: list, model_cost: list, rd_budget: int, cost_para: float,
                 model_result_array: np.ndarray, **kwargs):
        self.task = task
        self.model_set = model_set
        self.G = G
        self.model_cost = model_cost
        self.rd_budget = rd_budget
        self.cost_para = cost_para
        self.model_result_array = model_result_array

        # Whole-process stats
        self.num_epoch = num_epoch
        self.T = T
        self.total_cost = np.zeros((T,))
        self.cumulative_reward = np.zeros((T,))
        self.alg_v = np.zeros((T,))
        self.ref_v = np.zeros((T,))
        self.opr = np.zeros((T,))
        self.pick_ratio = np.zeros((T, self.G,))
        self.epoch = 0

        # Within-step stats
        self.max_reward_t = 0.
        self.cumulative_reward_t = 0.
        self.cost_t = 0.
        self.rd_t = 0
        self.ref_v_model_t = None
        self.ref_v_t = None
        self.opr_t = 0
        self.visitation_t = np.zeros((self.G,))
        self.actions_t = []

    def get_reference_value(self, id_t):
        ref_v_model = np.empty((self.G,))
        for g in range(self.G):
            pass_at_1 = np.mean(self.model_result_array[g][id_t])
            if pass_at_1 == 0:
                ref_v_model[g] = 0
            else:
                ref_v_model[g] = 1 - self.cost_para * self.model_cost[g] / pass_at_1

        self.ref_v_model_t = ref_v_model
        self.ref_v_t = max(0., np.max(ref_v_model))
        return self.ref_v_t

    def get_reward(self, selected_model, id_t):
        assert self.rd_t < self.rd_budget or (self.rd_t == self.rd_budget and selected_model is None)

        if selected_model is not None:
            return np.random.choice(self.model_result_array[selected_model][id_t])
        return None

    def update_env_stats(self, selected_model, reward):
        if selected_model is not None:
            self.max_reward_t = max(self.max_reward_t, reward)
            self.cumulative_reward_t += reward
            self.cost_t += self.model_cost[selected_model]
            self.rd_t += 1
            self.visitation_t[selected_model] += 1
            self.actions_t.append(selected_model)

    def reset_within_step_stats(self):        # Reset within-step stats
        self.max_reward_t = 0.
        self.cumulative_reward_t = 0.
        self.cost_t = 0.
        self.rd_t = 0
        self.ref_v_model_t = None
        self.ref_v_t = None
        self.visitation_t = np.zeros((len(self.visitation_t),))
        self.actions_t = []
        self.opr_t = 0

    def update_entire_process_stats(self, t):      # Tracing statistics
        self.total_cost[t:] += self.cost_t
        alg_v_t = self.max_reward_t - self.cost_para * self.cost_t
        self.alg_v[t:] += alg_v_t
        self.ref_v[t:] += self.ref_v_t

        num_opr_t = 0
        if len(self.actions_t):
            for a in self.actions_t:
                num_opr_t += (self.ref_v_model_t[a] == self.ref_v_t)
            self.opr[t:] += float(num_opr_t) / len(self.actions_t)
        else:
            self.opr[t:] += float(np.max(self.ref_v_t) <= 0)

        self.pick_ratio[t:] += self.visitation_t / max(self.rd_t, 1)
        self.cumulative_reward[t:] += self.cumulative_reward_t
        if t == self.T - 1:
            self.epoch += 1

    def update_model_pool(self):
        self.G += 1
        if self.pick_ratio.shape[1] < self.G:
            self.pick_ratio = np.concatenate((self.pick_ratio, np.zeros((self.T, 1,))), axis=1)
        if len(self.visitation_t) < self.G:
            self.visitation_t = np.concatenate((self.visitation_t, np.zeros((1,))))

    def reset_env(self):
        self.G = 2

    def save_stats(self, save_filename, args_dict):
        np.savez(save_filename,
                 total_cost=self.total_cost / (np.arange(1, self.T + 1) * self.epoch),
                 total_success=self.cumulative_reward / (np.arange(1, self.T + 1) * self.epoch),
                 alg_v=self.alg_v / (np.arange(1, self.T + 1) * self.epoch),
                 opr=self.opr / (np.arange(1, self.T + 1) * self.epoch),
                 ref_v=self.ref_v / (np.arange(1, self.T + 1) * self.epoch),
                 args=args_dict,
                 model_set=np.array(self.model_set, dtype=str),
                 model_cost=np.array(self.model_cost, dtype=float),
                 )
