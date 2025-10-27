import numpy as np
from termcolor import colored
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from algorithms.aux import LEARNER
from utils.aux import cost_aware_env


MODEL_SET = ['MODEL_A', 'MODEL_B', 'MODEL_C', 'MODEL_D', 'MODEL_E']
MODEL_COST = [12.50, 0.75, 10.00, 1.37, 1.60]

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_dim', type=int, default=768)
parser.add_argument('--learner', type=str, default='promptwise')

parser.add_argument('--exp_para', type=float, default=None)

parser.add_argument('--kernel_method', type=str, default='rbf')
parser.add_argument('--kernel_para_c', type=float, default=1.)
parser.add_argument('--kernel_para_d', type=float, default=3.)
parser.add_argument('--kernel_para_gamma', type=float, default=5.0)
parser.add_argument('--krr_alpha', type=float, default=1.)
parser.add_argument('--exp_eta', type=float, default=1.0)
parser.add_argument('--tau_exp', type=int, default=1)
parser.add_argument('--reg_method', type=str, default='klr')

parser.add_argument('--cost_para', type=float, default=.05)
parser.add_argument('--rd_budget', type=int, default=5)

parser.add_argument('--eval_epochs', type=int, default=20)
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--n_data', type=int, default=164)
parser.add_argument('--save_filename', type=str, default=None)
parser.add_argument('--save_file', type=int, default=1)
parser.add_argument('--seed', type=int, default=1234, help='Random seed')


def main():
    args = parser.parse_args()
    T = args.T
    n_data = args.n_data
    G = len(MODEL_SET)
    num_epoch = args.eval_epochs
    num_dim = args.num_dim
    cost_para = args.cost_para
    tau_exp = args.tau_exp
    rd_budget = args.rd_budget
    reg_method = args.reg_method
    np.random.seed(args.seed)

    prompt_reps = np.random.randn(n_data, num_dim,)

    model_result_array = np.random.randint(2, size=(G, n_data, 10))
    print(colored(f"Model data loaded: {MODEL_SET}. Cost: {MODEL_COST}", 'blue'), '\n')

    env = cost_aware_env(G=G, task='test-example', T=T, num_epoch=num_epoch, model_set=MODEL_SET,
                         model_cost=MODEL_COST, rd_budget=rd_budget, cost_para=cost_para,
                         model_result_array=model_result_array)

    for epoch in range(1, num_epoch + 1):
        learner = LEARNER[args.learner](G=G, T=T, num_dim=num_dim,
                                        kernel_method=args.kernel_method, krr_alpha=args.krr_alpha,
                                        kernel_para_c=args.kernel_para_c, kernel_para_d=args.kernel_para_d,
                                        kernel_para_gamma=args.kernel_para_gamma,
                                        cost_para=cost_para, rd_budget=rd_budget, model_cost=MODEL_COST,
                                        exp_para=args.exp_para,
                                        exp_eta=args.exp_eta, tau_exp=tau_exp, reg_method=reg_method)

        for t in range(T):
            env.reset_within_step_stats()

            id_t = np.random.randint(n_data)
            env.get_reference_value(id_t)

            context_t = prompt_reps[id_t].reshape(1, -1)
            skip_prompt = False
            while not skip_prompt:
                model_t = learner.select_arm(context=context_t)
                reward = env.get_reward(selected_model=model_t, id_t=id_t)
                skip_prompt = learner.rd_skip
                learner.update_stats(g=model_t, context=context_t, reward=reward)
                env.update_env_stats(selected_model=model_t, reward=reward)
            env.update_entire_process_stats(t=t)  # Update statistics

            if (t + 1) % 100 == 0:
                print(colored(f'TEST.cost-aware.selection, '
                              f'model.set: {MODEL_SET}, model.cost: {MODEL_COST}, '
                              f'learner: {args.learner}, '
                              f'epoch {epoch}, step {t + 1}', 'red'))
                print(colored(f'Num.Round: {env.rd_t}, Reward: {env.cumulative_reward_t}, '
                              f'Value: {env.alg_v[t] / ((t + 1) * epoch)}, '
                              f'Oracle.Value: {env.ref_v[t] / ((t + 1) * epoch)}, '
                              f'OPR: {env.opr[t] / ((t + 1) * epoch)}, '
                              f'Success/Reward: {env.cumulative_reward[t] / ((t + 1) * epoch)}, '
                              f'Visitation: {env.pick_ratio[t] / ((t + 1) * epoch)}', 'blue'), '\n')


if __name__ == '__main__':
    main()
