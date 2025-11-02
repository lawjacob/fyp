# PromptWise
Official repository of the paper "PromptWise: Online Learning for Cost-Aware Prompt Assignment in Generative Models"

[Xiaoyan Hu](https://yannxiaoyanhu.github.io), [Laren Pick](https://lmpick.github.io/), [Ho-fung Leung](http://www.cse.cuhk.edu.hk/~lhf/), [Farzan Farnia](https://www.cse.cuhk.edu.hk/~farnia/Home.html) [[Paper](https://arxiv.org/abs/2505.18901)]

![Figure](https://github.com/yannxiaoyanhu/PromptWise/blob/main/Fig1.png)
![Figure](https://github.com/yannxiaoyanhu/PromptWise/blob/main/Interaction_Protocol.png)

## Usage Examples

PromptWise: ```python test.py --learner promptwise --kernel_method lin --reg_method mle --cost_para 0.001 --rd_budget 5```

PromptWise-KLR: ```python test.py --learner promptwise --kernel_method rbf --reg_method klr --cost_para 0.001 --rd_budget 5```

## Citation
```
@misc{hu2025promptwiseonlinelearningcostaware,
      title={PromptWise: Online Learning for Cost-Aware Prompt Assignment in Generative Models}, 
      author={Xiaoyan Hu and Lauren Pick and Ho-fung Leung and Farzan Farnia},
      year={2025},
      eprint={2505.18901},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.18901}, 
}
```
