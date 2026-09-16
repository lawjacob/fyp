# PromptWise
Official repository of the paper "PromptWise: Online Learning for Cost-Aware Prompt Assignment in Generative Models"

[Xiaoyan Hu](https://yannxiaoyanhu.github.io), [Laren Pick](https://lmpick.github.io/), [Ho-fung Leung](http://www.cse.cuhk.edu.hk/~lhf/), [Farzan Farnia](https://www.cse.cuhk.edu.hk/~farnia/Home.html) [[Paper](https://arxiv.org/abs/2505.18901)]

![Figure](https://github.com/yannxiaoyanhu/PromptWise/blob/main/Fig1.png)
![Figure](https://github.com/yannxiaoyanhu/PromptWise/blob/main/Interaction_Protocol.png)

## Usage Examples

PromptWise: ```python test.py --learner promptwise --kernel_method lin --reg_method mle --cost_para 0.001 --rd_budget 5```

PromptWise-KLR: ```python test.py --learner promptwise --kernel_method rbf --reg_method klr --cost_para 0.001 --rd_budget 5```

## Acknowledgements

The authors would like to acknowledge the following repositories:

1. HumanEval: [https://github.com/openai/human-eval](https://github.com/openai/human-eval)
2. CodeGeeX: [https://github.com/zai-org/CodeGeeX](https://github.com/zai-org/CodeGeeX)
3. BigCodeBench: [https://github.com/bigcode-project/bigcodebench](https://github.com/bigcode-project/bigcodebench)

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

## Retry recovery diagnostic (Qwen + exponential weighting)

For CPU-only replay of existing six-model responses, use the
[DARS replay guide](docs/dars-replay.md). It includes the four policies, audited
data selection, paired comparisons, and commands requiring no GPU inference.
The [follow-up protocol](docs/dars-followup-protocol.md) isolates recovery after
a shared failure and tests the uncertainty prior; see the
[follow-up results](docs/dars-followup-results.md) for findings and data-expansion options.

See [the overnight benchmark guide](docs/overnight-benchmark.md) for GPU setup,
resumable Qwen collection, four routing policies, cost/attempt limits, and replay
traces. This is a new diagnostic, not a numerical reproduction of the paper.

```bash
python -m pip install -r requirements-benchmark.txt  # install CUDA PyTorch separately
mkdir -p runs
nohup bash scripts/overnight.sh > runs/overnight.log 2>&1 &
```
