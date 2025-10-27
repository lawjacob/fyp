from .ca_pak_ucb_tS import cost_aware_pak_ucb_till_succeed
from .promptwise import promptwise

from .baselines import random_selector, greedy_selector, greedy_till_succeed, random_till_succeed
from .baselines import lowest_cost_selector, highest_cost_selector


LEARNER = {
    "random": random_selector,
    "greedy": greedy_selector,
    "RtS": random_till_succeed,
    "GtS": greedy_till_succeed,
    "lowest-cost": lowest_cost_selector,
    "highest-cost": highest_cost_selector,
    "ca-pak-ucb-tS": cost_aware_pak_ucb_till_succeed,
    "promptwise": promptwise,
}
