import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import argparse
import torch
import wandb
from model import LLaMA2_SASRec, HSTU_SASRec, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import PretrainDataset_NEW
import logging

from utils import *


# For example, to run with DDP on 4 gpus on 1 node:
# torchrun --standalone --nproc_per_node=4 pretrain.py OR python -m torch.distributed.launch --nproc_per_node=4 pretrain.py
        
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# ----- New Training Epoch: for Recommender System -- logs_, Y refers to Tokens, Targets -----

bce_criterion = torch.nn.BCEWithLogitsLoss()

def train_epoch(epoch, device, model_name):
    l2_emb = 0.0
    start_time=time.time()
    for step, (user_arr, seq_arr, pos_arr, neg_arr) in enumerate(train_loader):
        # load data to device
        user_arr=user_arr.to(device)
        seq_arr=seq_arr.to(device)
        pos_arr=pos_arr.to(device)
        neg_arr=neg_arr.to(device)
    
        lr = get_lr(epoch*iter_per_epoch+step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # and using the GradScaler if data type is float16
        #for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = 0 == gradient_accumulation_steps - 1
        with ctx:
            pos_logits, neg_logits = model(user_arr, seq_arr, pos_arr, neg_arr)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)

            indices = torch.where(pos_arr != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            # restrict the l2-norm of item embedding table parameters
            for param in model.module.item_emb.parameters(): loss += l2_emb * torch.norm(param)
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        #
        if (step + 1) % gradient_accumulation_steps == 0:
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
        # print log
        if step % log_interval == 0:
            spend_time=time.time()-start_time
            logger.info(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch,
                        max_epoch, 
                        step, 
                        iter_per_epoch,
                        loss.item(), 
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))
            wandb.log({'loss': loss.item()})
        #
        if step % save_interval == 0:
            if ddp:
                if torch.distributed.get_rank() == 0:
                    model.eval()
                    torch.save(model.module.state_dict(),'{}/{}_iter_{}.pth'.format(save_dir,model_name,int(step+epoch*iter_per_epoch)))
                    model.train()
            else:
                model.eval()
                torch.save(model.state_dict(),'{}/{}_iter_{}.pth'.format(save_dir,model_name,int(step+epoch*iter_per_epoch)))
                model.train()


def init_model(usernum, itemnum, device, model_name="llama", ckpt_name="epoch_0.pth"):
    # model init
    model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        # vocab_size=64793, # defined later by tokenizer -> In recommender system, it replaces by itemnum
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
        maxlen=maxlen
    )  # start with model_args from command line
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        if model_name == "llama":
            model = LLaMA2_SASRec(usernum, itemnum, device, gptconf)
        elif model_name == "hstu":
            model = HSTU_SASRec(usernum, itemnum, device, gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {save_dir}")
        # resume training from a checkpoint.
        ckpt_name = model_name + "_" + ckpt_name
        ckpt_path = os.path.join(save_dir, ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=device)
        # create the model
        gptconf = ModelArgs(**model_args)
        if model_name == "llama":
            model = LLaMA2_SASRec(usernum, itemnum, device, gptconf)
        elif model_name == "hstu":
            model = HSTU_SASRec(usernum, itemnum, device, gptconf)
        state_dict = checkpoint
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    return model

# I/O
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_only', default=False, type=str2bool)
    parser.add_argument('--ckpt_name', default="epoch_0.pth", type=str)
    parser.add_argument('--model_name', default="llama", type=str)

    # when use deepspeed
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--deepspeed", type=str, default="ds_config.json")
    

    args = parser.parse_args()

    run = wandb.init(
    # Set the project where this run will be logged
    project="Toy-RecLM",
    # Track hyperparameters and run metadata
    config={
    })

    # Mode 
    eval_only = args.eval_only # if True, script exits right after the first eval
    # model_name
    model_name = args.model_name

    out_dir = 'out'
    max_epoch = 50
    eval_interval = 1
    log_interval = 100
    save_interval = 10000
    eval_iters = 200
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
    #
    gradient_accumulation_steps = 1 # used to simulate larger batch sizes
    batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
    # model config
    max_seq_len = 512
    dim = 512
    n_layers = 8
    n_heads = 8
    multiple_of = 32
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate = 3e-4 # max learning rate
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 1000 # how many steps to warm up for
    lr_decay_iters = 80000 # should be ~= max_iters per Chinchilla
    min_lr = 1e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = False # use PyTorch 2.0 to compile the model to be faster

    # ----- sasrec param -----
    maxlen = 50

    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    # exec(open("configurator.py").read())  # overrides from command line or config file
    # config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    save_dir =os.path.join(out_dir , 'train')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    logger = get_logger(os.path.join(save_dir,'log.log'))
    # various inits, derived attributes, I/O setup
   # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?

    if ddp:
        # Check if the operating system is Windows
        if os.name == 'nt':
            # Diff between backends: https://pytorch.org/docs/stable/distributed.html
            init_process_group(backend="gloo")
        else:
            # If the operating system is Linux based, os.name == 'posix'
            init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        #assert gradient_accumulation_steps % ddp_world_size == 0
        #gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    # tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * maxlen
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {maxlen} maxlen")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.cuda.amp.autocast()
    )
    #
    best_val_loss = 1e9
    #
    # init dataloader
    data_path_list=[
        './data/ml-1m',
    ]
    dataset = data_partition(data_path_list[0])
    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    num_batch = len(user_train) // batch_size # tail? + ((len(user_train) % batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    train_ds = PretrainDataset_NEW(user_train, usernum, itemnum, maxlen)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0 if os.name == 'nt' else 4,
        sampler=train_sampler
    )
    #init model
    print(f"model_name:{model_name}")
    model=init_model(usernum, itemnum, device, model_name)
    model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0
    # wrap model into DDP container
    if ddp:
        # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support `ComplexFloat`
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])
        #
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    # training loop
    iter_per_epoch=len(train_loader)
    if not eval_only:
        for epoch in range(max_epoch):
            if eval_only: break
            train_epoch(epoch, device, model_name)
            if ddp:
                if torch.distributed.get_rank() == 0:  # master node
                    torch.save(raw_model.state_dict(),'{}/{}_epoch_{}.pth'.format(save_dir,model_name,epoch))
            else:
                torch.save(raw_model.state_dict(),'{}/{}_epoch_{}.pth'.format(save_dir,model_name,epoch))
    else:
        # simple test on cpu
        device = 'cpu'
        init_from = 'resume'
        ckpt_name = args.ckpt_name
        model=init_model(usernum, itemnum, device, model_name, ckpt_name)
        model.eval()
        model.to(device)
        t_test = evaluate(model, dataset, maxlen, device)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    if ddp:
        destroy_process_group()


