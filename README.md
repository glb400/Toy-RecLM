# Toy-RecLM

A toy large model for recommender system based on [LLaMA2](https://arxiv.org/pdf/2307.09288.pdf), [SASRec](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf), and Meta's [actions-speak-louder-than-words](https://arxiv.org/pdf/2402.17152.pdf).


## Basic Model

### Training Framework

+ [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### Model Architecture

+ Version 1: Basic Implementation -- Combination of LLaMA2 and SASRec.

    LLaMA2 model as backbone based on [baby-llama2-chinese](https://github.com/DLLXW/baby-llama2-chinese) and [SASRec](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)([SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch/tree/master)) at the prediction part. 

    + Part 1: At first, we stack LLaMA2's Transformer Blocks. Note that since LLaMA2 uses decoder-only framework, it utilizes casual mask just the same as SASRec. 

    <div  align="center">    
        <img src="https://github.com/glb400/Toy-RecLM/blob/main/figs/llama1.png" width = "200" align=center />
        <p>LLaMA2 Transformer Block</p>
    </div>

    + Part 2.1: After Transformer Blocks, we implement prediction layer in SASRec. Specifically, we adopt an MF layer to predict the relevance of item $i$ by sharing item embedding.

    <div  align="center">    
        <img src="https://github.com/glb400/Toy-RecLM/blob/main/figs/sasrec1.png" width = "200" align=center />
        <p>MF Layer using Shared Item Embedding</p>
    </div>
     
    + Part 2.2: we generate an embedding by considering all actions of a user following SASrec.

    <div  align="center">    
        <img src="https://github.com/glb400/Toy-RecLM/blob/main/figs/sasrec2.png" width = "200" align=center />
        <p>Explicit User Modeling by Generating Embedding</p>
    </div>

+ Version 2. **[actions-speak-louder-than-words](https://arxiv.org/pdf/2402.17152.pdf)'s modification for Model** -- Hierarchical Sequential Transduction Unit (HSTU)

    ***Note that HSTU adopts a new pointwise aggregated attention mechanism instead of softmax attention in Transformers. (Just as in [Deep Interest Network](https://github.com/zhougr1993/DeepInterestNetwork)).***

    + HSTU
        
        <div  align="center">    
            <img src="https://github.com/glb400/Toy-RecLM/blob/main/figs/metallm2.png" width = "200" align=center />
            <img src="https://github.com/glb400/Toy-RecLM/blob/main/figs/metallm3.png" width = "200" align=center />
            <p>HSTU formulae & Structure</p>
        </div>
    
### Model Training

We convert each user sequence (excluding the last action) $(\mathcal{S}_{1}^{u},\mathcal{S}_{2}^{u},\cdots,\mathcal{S}_{|\mathcal{S}^{u}|-1}^{u})$ to a fixed length sequence $s = \{s_1, s_2, . . . , s_n\}$ via truncation or padding items. We define $o_t$ as the expected output at time step $t$ and  adopt the binary cross entropy loss as the objective function as in SASRec.

<div  align="center">    
    <img src="https://github.com/glb400/Toy-RecLM/blob/main/figs/sasrec3.png" width = "200" align=center />
    <br>
    <img src="https://github.com/glb400/Toy-RecLM/blob/main/figs/sasrec4.png" width = "200" align=center />
    <p>Model Training following SASRec</p>
</div>

## Implementation for **Matching Task**

**[actions-speak-louder-than-words](https://arxiv.org/pdf/2402.17152.pdf)'s design for Matching** 

### Data Process

Input is dataset of samples of ***user historical behavior sequences*** as follows.

```
<user_1 profile> <item_1 id features> ... <item_n id features>
<user_2 profile> <item_1 id features> ... <item_n id features>
```

Moreover, ***auxiliary time series tokens*** could be added into seqs above if available.

<!-- ### docker -->

### Installation

use the command to setup environment
```bash
# add conda-forge
conda config --add channels conda-forge

# install env
conda env create -f env.yml -n reclm
conda activate reclm
```

### Training

Predict $p(\hat{s}_{i+1}|s_1,\cdots,s_i )$, and ***non-behavioral tokens & negative feedback*** will not included in loss calculation.

To train this model, use the command (set mode by *\$eval_only\$* and choose backbone by *\$model_name\$*)
```bash
# LLaMA2 as backbone
torchrun --standalone --nproc_per_node=2 main.py --eval_only=false --model_name='llama'

# HSTU as backbone
torchrun --standalone --nproc_per_node=2 main.py --eval_only=false --model_name='hstu'
```

### Evaluation

Use **NDCG\@10** and **HR\@10** to evaluate performance on whole dataset.

To evalutate this model by NDCG and hit ratio, use the command (set mode by *\$eval_only\$* and load checkpoint by *\$ckpt_name\$*,  and choose backbone by *\$model_name\$*)
```bash
# LLaMA2 as backbone
torchrun --standalone --nproc_per_node=2 main.py --eval_only=true --ckpt_name='epoch_15.pth' --model_name='llama'

# HSTU as backbone
torchrun --standalone --nproc_per_node=2 main.py --eval_only=true --ckpt_name='epoch_15.pth' --model_name='hstu'
```

About evaluation function, please refer to https://pmixer.github.io/posts/Argsort.

Dataset

+ [Movielens1M_m1](https://huggingface.co/datasets/reczoo/Movielens1M_m1)

### Support Acceleration by DeepSpeed 

To accelerate by [DeepSpeed](https://github.com/microsoft/DeepSpeed), use the command
```bash
# pip install deepspeed

# recommended local compilation
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

#### Multi-node Configuration: Setup Passwordless SSH

For resource configuration of multi-node, we need generate ssh-key by ***ssh-keygen*** and pass key to other nodes by ***ssh-copy-id***. Here we have one node with 2 gpus(NVIDIA RTX A6000). Still, we set multi-node configuration for guidance and in this case host just need to communicate with itself.

+ First generate ssh-key

```bash
ssh-keygen
```

+ Write ~/.ssh/config to get nickname and identity file of host and as follows.

```bash
Host host1
    User guyr
        Hostname 127.0.0.1
            port 22
                IdentityFile ~/.ssh/id_rsa
```

+ Write hostfile for deepspeed to get multi-node.

```bash
host1 slots=2
```

+ Use ***ssh-copy-id*** to copy identity file to other nodes

```bash
ssh-copy-id -i ~/.ssh/id_rsa host1

# test
ssh host1
```

#### Argument Parser Configuration

To successfully run deepspeed, we need to add ***local_rank*** & ***deepspeed*** parameters in argparser, since deepspeed will add these hyperparameters to each process when launching tasks.

```python
# when use deepspeed
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--deepspeed", type=str, default="ds_config.json")
```

Finally run deepspeed using the command

```bash
deepspeed --hostfile ./hostfile --master_port 12345 --include="host1:0,1" main.py --eval_only=false --model_name='llama' --deepspeed ds_config.json
```

## News

+ [2024/04]: Support [Deepspeed](https://www.deepspeed.ai/getting-started/).


<!-- + [2024/04]: Support docker. -->


<!-- ## Implementation for **CTR Prediction**

***------------------------------ !!! TBD !!! ------------------------------*** -->

<!-- 

## Support **Mixture of Experts(MoEs)** and **Sliding Window Attention(SWA)** based on [mistral](https://github.com/mistralai/mistral-src)

## Support Low-memory & Acceleration Optimization

+ Support Quantization and Parameter-efficient Fine-tuning(PEFT) methods based on [lit-llama](https://github.com/Lightning-AI/lit-llama).

+ Support Low-memory Optimizers, e.g., [Adafactor](https://arxiv.org/abs/1804.04235), [Sophia](https://arxiv.org/abs/2305.14342), [LOMO](https://github.com/OpenLMLab/LOMO).


## Add Time-series Prediction Methods

## vSupport Multi-modal Features -->

