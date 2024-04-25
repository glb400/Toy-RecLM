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

    [gin](https://github.com/google/gin-config) is applied to manage configurations. The configs(.gin files) are stored in the folder 'configs/'.

+ args parser

    [absl](https://github.com/abseil/abseil-py) is used to manage command line. ***app*** takes charge of parsing command line args and ***flags*** takes charge of definition and management of args. 

+ parallel programming

    Authors use [TCP initialization with ***torch.multiprocessing(MP)***](https://zhuanlan.zhihu.com/p/393648544) for ***torch.nn.parallel.DistributedDataParallel(DDP)***, which does not need ***torch.distributed.launch***. Instead, the information to initial process group is required for ***init_process_group()***. Here ***MP*** is used to simplify this procedure, and ***MP.spawn()*** is to create multiple process for manually launch them. 

+ dataset
    
    In dataset, historical item seqs is sorted in reverse chronological order. ***_sample_ratio*** decides the number kept in item seqs and ***sampling_kept_mask*** indicates the remaining items.

## Model Architecture

### Main Structure

Main structure lies in ***get_sequential_encoder()***, where exists 2 module type, i.e., 'SASRec' & 'HSTU'. At a high level, this model aims at ***generating user (behavioral seqs) embs***. 

+ SASRec

```python
# run a transformer block
# user (behavioral seqs) embs as input and output
def _run_one_layer(
    self,
    i: int,
    user_embeddings: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    Q = F.layer_norm(
        user_embeddings, normalized_shape=(self._embedding_dim,), eps=1e-8,
    )
    mha_outputs, _ = self.attention_layers[i](
        query=Q,
        key=user_embeddings,
        value=user_embeddings,
        attn_mask=self._attn_mask,
    )
    user_embeddings = self.forward_layers[i](
        F.layer_norm(
            Q + mha_outputs,
            normalized_shape=(self._embedding_dim,),
            eps=1e-8,
        )
    )
    user_embeddings *= valid_mask
    return user_embeddings

# run transformer blocks and get corresponding final user embs
def generate_user_embeddings(
    self,
    past_lengths: torch.Tensor,
    past_ids: torch.Tensor,
    past_embeddings: torch.Tensor,
    past_payloads: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Args:
        past_ids: (B, N,) x int

    Returns:
        (B, N, D,) x float
    """
    past_lengths, user_embeddings, valid_mask = self._input_features_preproc(
        past_lengths=past_lengths,
        past_ids=past_ids,
        past_embeddings=past_embeddings,
        past_payloads=past_payloads,
    )

    for i in range(len(self.attention_layers)):
        if self._activation_checkpoint:
            user_embeddings = torch.utils.checkpoint.checkpoint(
                self._run_one_layer, i, user_embeddings, valid_mask,
                use_reentrant=False,
            )
        else:
            user_embeddings = self._run_one_layer(i, user_embeddings, valid_mask)

    return self._output_postproc(user_embeddings)

def forward(
    self,
    past_lengths: torch.Tensor,
    past_ids: torch.Tensor,
    past_embeddings: torch.Tensor,
    past_payloads: Dict[str, torch.Tensor],
    batch_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        past_ids: [B, N] x int64 where the latest engaged ids come first. In
            particular, [:, 0] should correspond to the last engaged values.
        past_ratings: [B, N] x int64.
        past_timestamps: [B, N] x int64.

    Returns:
        encoded_embeddings of [B, N, D].
    """
    encoded_embeddings = self.generate_user_embeddings(
        past_lengths,
        past_ids,
        past_embeddings,
        past_payloads,
    )
    return encoded_embeddings
```

***Notice: in FFN, conv1d is applied to replace linear layer to be efficient.***

```python
class StandardAttentionFF(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        activation_fn: str,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        assert activation_fn == "relu" or activation_fn == "gelu", \
            f"Invalid activation_fn {activation_fn}"

        self._conv1d = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=hidden_dim,
                kernel_size=1,
            ),
            torch.nn.GELU() if activation_fn == "gelu" else torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=embedding_dim,
                kernel_size=1,
            ),
            torch.nn.Dropout(p=dropout_rate),
        )

    def forward(self, inputs) -> torch.Tensor:
        # Conv1D requires (B, D, N)
        return self._conv1d(inputs.transpose(-1, -2)).transpose(-1, -2) + inputs
```

+ HSTU

```python

```


### Important Components

Specifically, this main structure is composed by several important components, e.g., 
+ embedding module
+ interaction module
+ input features preprocessor module
+ output postprocessor module

The below is the explanation of each module.

+ embedding module (implemented by ***CategoricalEmbeddingModule*** & ***LocalEmbeddingModule***)

    The embs are initialized by ***truncated_normal()***, which implements a truncated normal distribution to avoid extreme values leading to grad vanishing. Specifically, the interval to truncates is [-2,2]. 

    ```python
    def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x
    ```

    The only different of categorical and local embs is that categorical embs need remap from item to categorical since pd.Categorical is applied at data preprocess step.

    ```python
    # local 
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
    return self._item_emb(item_ids)

    # categorical
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
    item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
    return self._item_emb(item_ids)
    ```

+ interaction module (implemented by ***DotProductSimilarity*** & ***create_mol_interaction_module***)

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

    + ***MoL***: applied by ***create_mol_interaction_module*** function, which implements ***MoL(Mixture-of-Logits)learned similarity*** in [Revisiting Neural Retrieval on Accelerators](https://arxiv.org/abs/2306.04039).
    
        ![MoL Structure](../../figs/mol.png)

    ***DEFAULT SETTING is DotProduct.***

+ input features preprocessor module (implemented by ***LearnablePositionalEmbeddingInputFeaturesPreprocessor*** & ***LearnablePositionalEmbeddingRatedInputFeaturesPreprocessor*** & ***CombinedItemAndRatingInputFeaturesPreprocessor***)

    This module aims to perform preprocess operations such as add positional embs. *RatedPreprocessor* handles additional features like rating by concating the user embs and rating embs. *CombinedItemAndRatingPreprocessor* handles items and corresponding rating as individual inputs, which doubles the seq length.

    ***DEFAULT SETTING is LearnablePositionalEmbeddingInputFeaturesPreprocessor.***

+ output postprocessor module (implemented by ***L2NormEmbeddingPostprocessor*** & ***LayerNormEmbeddingPostprocessor***)

    This module aims at apply norm to the output of transformer block.

    ***DEFAULT SETTING is L2NormEmbeddingPostprocessor.***

### Loss

+ BCELoss

+ SampledSoftmaxLoss

***DEFAULT SETTING is SampledSoftmaxLoss.***

loss_module: str = "SampledSoftmaxLoss",

### Negative Sampling

+ sampling_strategy == "in-batch"
    
    InBatchNegativesSampler
+ sampling_strategy == "local"

    LocalNegativesSampler

***DEFAULT SETTING is in-batch.***

### Top-$k$ Methods


