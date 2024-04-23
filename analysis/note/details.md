# Details

In this part, we discuss the details of this generative-recommender, e.g., 

+ data preprocess
+ training process
+ model architecture

Notably, this work inherits many ideas from [Revisiting Neural Retrieval on Accelerators](https://arxiv.org/abs/2306.04039), and these work are from same authors.

## Data Preprocess

Here we takes [Amazon books](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv) as example.

+ format of sample of Amazon books:

    ```
    index,user_id,sequence_item_ids,sequence_ratings,sequence_timestamps
    428566,428566,"627204,513752,573980,594511,574513","5.0,5.0,5.0,5.0,5.0","1362355200,1366761600,1370390400,1380067200,1393977600"
    606756,606756,"384084,399081,72516,336635,155917,72377,140831,347262","5.0,5.0,5.0,5.0,2.0,5.0,1.0,4.0","1339459200,1344384000,1345507200,1349395200,1349395200,1354924800,1358208000,1401062400"
    ```

    The number of expected_num_unique_items is 695762.

+ preprocess procedure
    
    1. filter users and items with presence < 5
    2. categorize user id and item id to be memory-efficient
    3. sort and group items of same user by timestamp
    4. join user data(optional) and unify format

## Training Process

+ configuration

    [gin](https://github.com/google/gin-config) is applied to manage configurations.

+ args parser

    [absl](https://github.com/abseil/abseil-py) is used to manage command line. ***app*** takes charge of parsing command line args and ***flags*** takes charge of definition and management of args. 

+ parallel programming

    Authors use [TCP initialization with ***torch.multiprocessing(MP)***](https://zhuanlan.zhihu.com/p/393648544) for ***torch.nn.parallel.DistributedDataParallel(DDP)***, which does not need ***torch.distributed.launch***. Instead, the information to initial process group is required for ***init_process_group()***. Here ***MP*** is used to simplify this procedure, and ***MP.spawn()*** is to create multiple process for manually launch them. 

+ dataset
    
    In dataset, historical item seqs is sorted in reverse chronological order. ***_sample_ratio*** decides the number kept in item seqs and ***sampling_kept_mask*** indicates the remaining items.

+ embedding
    
    The embs are initialized by ***truncated_normal()***, which implements a truncated normal distribution to avoid extreme values leading to grad vanishing. Specifically, the interval to truncates is [-2,2]. 
    
    The only different of categorical and local embs is that categorical embs need remap from item to categorical since pd.Categorical is applied at data preprocess step.

## Model Architecture

+ interaction module

    The core module is to define how input embs interact with target items, possibly computing scores or similarities.

    Its specific implementation is ***GeneralizedInteractionModule*** class, which computes similarity by input embs & item embs & item sideinfo. All models are this class.

    Concretely, it has 2 kinds of similarities, e.g., 

    + ***DotProduct***: applied by ***DotProductSimilarity*** class, which adjusts the shape of potential difference of batchsize & number of items.

        ```python
        if item_embeddings.size(0) == 1:
            # [B, D] x ([1, X, D] -> [D, X]) => [B, X]
            return torch.mm(input_embeddings, item_embeddings.squeeze(0).t()), {}  # [B, X]
        elif input_embeddings.size(0) != item_embeddings.size(0):
            # (B * r, D) x (B, X, D).
            B, X, D = item_embeddings.size()
            return torch.bmm(input_embeddings.view(B, -1, D), item_embeddings.permute(0, 2, 1)).view(-1, X)
        else:
            # assert input_embeddings.size(0) == item_embeddings.size(0)
            # [B, X, D] x ([B, D] -> [B, D, 1]) => [B, X, 1] -> [B, X]
            return torch.bmm(item_embeddings, input_embeddings.unsqueeze(2)).squeeze(2)
        ```

    + ***MoL***: applied by ***DotProductSimilarity*** class, which implements ***MoL(Mixture-of-Logits)learned similarity*** in [Revisiting Neural Retrieval on Accelerators](https://arxiv.org/abs/2306.04039).
    
        ![](../../figs/mol.png)


+ sequential encoder

