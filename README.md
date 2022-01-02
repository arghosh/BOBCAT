# BOBCAT: Bilevel Optimization-Based Computerized Adaptive Testing

This is an official repository of the paper [BOBCAT: Bilevel Optimization-Based Computerized Adaptive Testing](https://arxiv.org/pdf/2108.07386.pdf) to be presented at IJCAI 2021. 


`In this paper, we propose BOBCAT, a Bilevel Optimization-Based framework for CAT to directly learn a data-driven question selection algorithm from training data. We show that BOBCAT outperforms existing CAT methods (sometimes significantly) at reducing test length.`


## Environment Setup
This repository uses the following packages in Python3.
```
torch==1.7.1
```

## Training
You can download the preprocessed datasets from [Google Drive](https://drive.google.com/file/d/1BItI5PVl4-iZAKd-39kjdsnHIPmRG3ld/view?usp=sharing) in `/data/` folder. Preprocessing scirpts can be found in `utils/` folder.

```(bash)
python train.py\
    --dataset {eedi-1 or eedi-3 or assit2009 or junyi or ednet}
    --model {base-sampling where base is binn/biirt and sampling is random/active/unbiased/biased}\ 
    --n_query {1, 3, 5, or 10}
    --cuda
```

Hyperparameter ranges are:
```(bash)
hyperparameters = [
    [('dataset',), ['ednet', 'eedi-1', 'eedi-3', 'assist2009', 'junyi']],
    [('model',), ['biirt-active', 'biirt-random', 'biirt-unbiased','biirt-biased', 'binn-active', 'binn-random', 'binn-unbiased','binn-biased']],
    [('fold',), [ 1, 2, 3, 4, 5 ]],
    [('hidden_dim'), [256]],
    [('lr',), [ 1e-3 ]],
    [('inner_lr',), [ 2e-1, 1e-1, 5e-2]],
    [('meta_lr',), [ 1e-4 ]],
    [('inner_loop',), [ 5 ]],
    [('policy_lr',), [2e-3,  2e-4]],
    [('n_query',), [1, 3, 5, 10]]
]
```

## Citation
If you find this code useful in your research then please cite  
```(bash)
@inproceedings{ghosh-bobcat,
  title     = {BOBCAT: Bilevel Optimization-Based Computerized Adaptive Testing},
  author    = {Ghosh, Aritra and Lan, Andrew},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {2410--2417},
  year      = {2021},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2021/332},
  url       = {https://doi.org/10.24963/ijcai.2021/332},
}
``` 

Contact:  Aritra Ghosh (aritraghosh.iem@gmail.com).
