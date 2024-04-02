# Toy-RecLM

A toy large model for recommender system based on Meta's [actions-speak-louder-than-words](https://arxiv.org/pdf/2402.17152.pdf) and [SASRec](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf).


## Basic Model

1 Training Framework

+ [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

2 Model Architecture

+ Version 1: Basic Implementation -- Combination of LLaMA2 and SASRec.

    LLaMA2 model as backbone based on [baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese) and [SASRec](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)([SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch/tree/master)) at the prediction part. 

    + Part 1: At first, we stack LLaMA2's Transformer Blocks. Note that since LLaMA2 uses decoder-only framework, it utilizes casual mask just the same as SASRec. 

    <div  align="center">    
        <img src="https://github.com/glb400/Toy-RecLM/blob/main/figs/llama1.png" width = "300" align=center />
        <p>LLaMA2 Transformer Block</p>
    </div>

    + Part 2.1: After Transformer Blocks, we implement prediction layer in SASRec. Specifically, we adopt an MF layer to predict the relevance of item $i$ by sharing item embedding.

    <div  align="center">    
        <img src="https://github.com/glb400/Toy-RecLM/blob/main/figs/sasrec1.png" width = "300" align=center />
        <p>MF Layer using Shared Item Embedding</p>
    </div>
     
    + Part 2.2: we generate an embedding by considering all actions of a user following SASrec.

    <div  align="center">    
        <img src="https://github.com/glb400/Toy-RecLM/blob/main/figs/sasrec2.png" width = "300" align=center />
        <p>Explicit User Modeling by Generating Embedding</p>
    </div>

+ Version 2. **[actions-speak-louder-than-words](https://arxiv.org/pdf/2402.17152.pdf)'s modification for Model** -- Hierarchical Sequential Transduction Unit (HSTU)

    ***------------------------------ !!! TBD !!! ------------------------------***

    + HSTU

        + Self-Attention

        Our code for Modified model's HSTU

        ```python


        ```        

        + Interaction


3 Model Training

We convert each user sequence (excluding the last action) $(\mathcal{S}_{1}^{u},\mathcal{S}_{2}^{u},\cdots,\mathcal{S}_{|\mathcal{S}^{u}|-1}^{u})$ to a fixed length sequence $s = \{s_1, s_2, . . . , s_n\}$ via truncation or padding items. We define $o_t$ as the expected output at time step $t$ and  adopt the binary cross entropy loss as the objective function as in SASRec.

<div  align="center">    
    <img src="https://github.com/glb400/Toy-RecLM/blob/main/figs/sasrec3.png" width = "300" align=center />
    <br>
    <img src="https://github.com/glb400/Toy-RecLM/blob/main/figs/sasrec4.png" width = "300" align=center />
    <p>Model Training following SASRec</p>
</div>

## Part1: Basic Implementation for **Matching**

**[actions-speak-louder-than-words](https://arxiv.org/pdf/2402.17152.pdf)'s design for Matching** 

1 Data Process

Input is dataset of samples of ***user historical behavior sequences*** as follows.

```
<user_1 profile> <item_1 id features> ... <item_n id features>
<user_2 profile> <item_1 id features> ... <item_n id features>
```

Moreover, ***auxiliary time series tokens*** could be added into seqs above if available.

2 Training

Predict $p(\hat{s}_{i+1}|s_1,\cdots,s_i )$, and ***non-behavioral tokens & negative feedback*** will not included in loss calculation.

To train this model, use the command (set mode by *\$eval_only\$*)
```bash
torchrun --standalone --nproc_per_node=2 main.py --eval_only=false
```

3 Evaluation

Use **NDCG\@10** and **HR\@10** to evaluate performance on whole dataset.

To evalutate this model by NDCG and hit ratio, use the command (set mode by *\$eval_only\$* and load checkpoint by *\$ckpt_name\$*)
```bash
torchrun --standalone --nproc_per_node=2 main.py --eval_only=true --ckpt_name='epoch_15.pth'
```

About evaluation function, please refer to https://pmixer.github.io/posts/Argsort.

Dataset: 

+ [Movielens1M_m1](https://huggingface.co/datasets/reczoo/Movielens1M_m1)


## Part2: Basic Implementation for **CTR Prediction**

***------------------------------ !!! TBD !!! ------------------------------***


<!-- ## v1.1: Support [Deepspeed](https://github.com/microsoft/DeepSpeed)


## v1.2: Support [Megatron](https://github.com/alibaba/Megatron-LLaMA)

## v1.3 Support GQA and SWA based on [mistral](https://github.com/mistralai/mistral-src)

## v1.4: Support Low-memory & Acceleration Optimization

+ Support Quantization and Parameter-efficient Fine-tuning(PEFT) methods based on [lit-llama](https://github.com/Lightning-AI/lit-llama).

+ Support Low-memory Optimizers, e.g., [Adafactor](https://arxiv.org/abs/1804.04235), [Sophia](https://arxiv.org/abs/2305.14342), [LOMO](https://github.com/OpenLMLab/LOMO).


## v1.5: Add Time-series Prediction Methods


## v1.6: Support Multi-modal Features -->

