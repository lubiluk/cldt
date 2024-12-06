"""
Full definition of a Decision Transformer Model, all of it in this single file.
Based on Andrej Karpathy's nanoGPT implementation of OpenAI's GPT-2. 

References:
1) nanoGPT implementation of OpenAI's GPT-2: https://github.com/karpathy/nanoGPT/blob/master/model.py
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, dropout, bias, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Consider installing Flash Attention for faster training"
            )

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x, attn_mask=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Build a mask to prevent attention to future tokens and padding tokens
        if attn_mask is not None:
            attn_mask = attn_mask.view(B, 1, 1, T)
            attn_mask = attn_mask & self.bias[:, :, :T, :T]
        else:
            attn_mask = self.bias[:, :, :T, :T]

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            float_mask = torch.zeros(B, 1, 1, T).to(
                device=attn_mask.device
            )  # Start with a tensor of zeros
            float_mask = float_mask.masked_fill(
                ~attn_mask, -10000
            )  # Fill masked positions with -10000

            dropout_p = self.dropout if self.training else 0
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=float_mask, dropout_p=dropout_p
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Causal mask (for future tokens and padding tokens)
            # -10000.0 used instead of -inf to prevent nans when all values are masked
            # due to padding masking and causal masking
            att = att.masked_fill(~attn_mask, -10000.0)

            # Apply softmax to get attention probabilities
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # Attention output
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout, bias, block_size):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias, block_size)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class DecisionTransformer(nn.Module):
    """This is basically a GPT-2 model with a few tweaks for Decision Transformer"""

    def __init__(
        self,
        n_layer: int = 3,
        n_head: int = 1,
        n_embd: int = 128,
        dropout: float = 0.1,
        bias: bool = False,
        K: int = 20,
        max_ep_len: int = 1000,
        state_dim: int = 17,
        act_dim: int = 6,
        act_discrete: bool = False,
        act_vocab_size: int = 1,
        act_tanh: bool = False,
        tanh_embeddings: bool = False,
    ):
        super().__init__()
        assert state_dim is not None
        assert act_dim is not None
        assert K is not None

        self.args = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            K=K,
            max_ep_len=max_ep_len,
            bias=bias,
            dropout=dropout,
            state_dim=state_dim,
            act_dim=act_dim,
            act_discrete=act_discrete,
            act_vocab_size=act_vocab_size,
            act_tanh=act_tanh,
            tanh_embeddings=tanh_embeddings,
        )

        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.K = K
        self.max_ep_len = max_ep_len
        self.act_discrete = act_discrete
        self.act_tanh = act_tanh
        self.tanh_embeddings = tanh_embeddings

        block_size = K * 3  # each block is composed of 3 tokens: R, s, a

        self.transformer = nn.ModuleDict(
            dict(
                te=nn.Embedding(max_ep_len, n_embd),
                re=nn.Linear(1, n_embd),
                se=nn.Linear(state_dim, n_embd),
                ae=(
                    nn.Embedding(act_vocab_size, n_embd)
                    if act_discrete
                    else nn.Linear(act_dim, n_embd)
                ),
                drop=nn.Dropout(dropout),
                h=nn.ModuleList(
                    [
                        Block(n_embd, n_head, dropout, bias, block_size)
                        for _ in range(n_layer)
                    ]
                ),
                ln_f=LayerNorm(n_embd, bias=bias),
                ln_e=LayerNorm(n_embd, bias=bias),
            )
        )

        # TODO: consider bias=False in the act_head as in the original GPT lm_head
        if act_discrete:
            self.act_head = nn.Linear(n_embd, act_vocab_size)
        else:
            self.act_head = nn.Linear(n_embd, act_dim)

        # TODO: Let's experiment later if we can use weight tying for DT
        # self.transformer.ae.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.te.weight.numel()
            n_params -= self.transformer.se.weight.numel()
            n_params -= self.transformer.ae.weight.numel()
            n_params -= self.transformer.re.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, states, actions, rtgs, tsteps, attn_mask=None, targets=None):
        device = states.device
        b, t = states.shape[0], states.shape[1]
        assert t <= self.K, f"Cannot forward sequence of length {t}, K is only {self.K}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        state_emb = self.transformer.se(
            states
        )  # state embeddings of shape (b, t, n_embd)
        action_emb = self.transformer.ae(
            actions.type(torch.long).squeeze(-1) if self.act_discrete else actions
        )  # action embeddings of shape (b, t, n_embd)
        rtg_emb = self.transformer.re(
            rtgs
        )  # return-to-go embeddings of shape (b, t, n_embd)
        tstep_emb = self.transformer.te(
            tsteps
        )  # time / position embeddings of shape (t, n_embd)

        if self.tanh_embeddings:
            state_emb = torch.tanh(state_emb)
            action_emb = torch.tanh(action_emb)
            rtg_emb = torch.tanh(rtg_emb)

        # time embeddings are treated similar to positional embeddings
        state_emb = state_emb + tstep_emb
        action_emb = action_emb + tstep_emb
        rtg_emb = rtg_emb + tstep_emb

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_emb = (
            torch.stack((rtg_emb, state_emb, action_emb), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(b, 3 * t, self.n_embd)
        )
        # TODO: Check if this LayerNorm is needed (some implementations don't have it)
        stacked_emb = self.transformer.ln_e(stacked_emb)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attn_mask = (
            torch.stack((attn_mask, attn_mask, attn_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(b, 3 * t)
        )

        x = self.transformer.drop(stacked_emb)
        for block in self.transformer.h:
            x = block(x, stacked_attn_mask)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.act_head(x)

            if self.act_tanh:
                logits = torch.tanh(logits)

            logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings

            # On cpu there are useful asserts
            # logits = logits.to("cpu")
            # targets = targets.to("cpu")

            if self.act_discrete:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
            else:
                act_dim = logits.shape[2]
                logits = logits.reshape(-1, act_dim)[attn_mask.reshape(-1) > 0]
                targets = targets.reshape(-1, act_dim)[attn_mask.reshape(-1) > 0]
                loss = F.mse_loss(logits, targets)
        else:
            # inference-time mini-optimization: only forward the act_head on the very last position
            logits = self.act_head(
                x[:, [-2], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


class DecisionTransformerDataCollator:
    def __init__(
        self,
        state_mean,
        state_std,
        K=20,
        max_ep_len=1000,
        reward_scale=1000.0,
        act_discrete=False,
    ):
        self.state_mean = state_mean
        self.state_std = state_std
        self.K = K
        self.max_ep_len = max_ep_len
        self.reward_scale = reward_scale
        self.act_discrete = act_discrete

    def __call__(self, batch):
        max_len = self.K
        s, a, r, rtg, timesteps, mask = [], [], [], [], [], []
        state_dim = batch[0]["observations"].shape[-1]
        act_dim = batch[0]["actions"].shape[-1]
        max_ep_len = self.max_ep_len
        state_mean = self.state_mean
        state_std = self.state_std
        scale = self.reward_scale
        act_dtype = torch.long if self.act_discrete else torch.float32

        for traj in batch:
            si = random.randint(0, traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
            a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            # TODO: Investigate this
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                self.discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                    : s[-1].shape[1]  # + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -1.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=act_dtype)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.bool)

        return s, a, r, rtg, timesteps, mask

    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum


class DecisionTransformerTrainer:
    def __init__(
        self,
        model,
        dataset,
        batch_size,
        learning_rate,
        weight_decay,
        beta1,
        beta2,
        max_iters,
        warmup_iters,
        device,
        decay_lr=False,
        lr_decay_iters=600000,
        min_lr=1e-5,
        pct_traj=1.0,
        reward_scale=1000.0,
        grad_clip=0.25,
        gradient_accumulation_steps=1,
        always_save_checkpoint=False,
        out_dir="out",
        eval_only=False,
        eval_interval=1000,
        eval_iters=100,
        log_interval=100,
    ):
        self.model = model
        self.trajectories = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.decay_lr = decay_lr
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.device = device
        self.pct_traj = pct_traj
        self.reward_scale = reward_scale
        self.grad_clip = grad_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.always_save_checkpoint = always_save_checkpoint
        self.out_dir = out_dir
        self.eval_only = eval_only
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.log_interval = log_interval

        self.args = dict(
            model_args=model.args,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            max_iters=max_iters,
            warmup_iters=warmup_iters,
            decay_lr=decay_lr,
            lr_decay_iters=lr_decay_iters,
            min_lr=min_lr,
            device=device,
            pct_traj=pct_traj,
            reward_scale=reward_scale,
            grad_clip=grad_clip,
            gradient_accumulation_steps=gradient_accumulation_steps,
            always_save_checkpoint=always_save_checkpoint,
            out_dir=out_dir,
            eval_only=eval_only,
            eval_interval=eval_interval,
            log_interval=log_interval,
        )

    def train(self):
        states, traj_lens, returns = [], [], []
        for path in self.trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.vstack(states)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        num_timesteps = sum(traj_lens)

        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]

        # used to reweight sampling so we sample according to timesteps instead of trajectories
        p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

        # TODO: Support resuming from checkpoint
        # optimizer
        optimizer = self.configure_optimizers(
            self.weight_decay, self.learning_rate, (self.beta1, self.beta2), self.device
        )

        dataset = TrajectoryDataset(self.trajectories)
        collator = DecisionTransformerDataCollator(
            state_mean=state_mean,
            state_std=state_std,
            K=self.model.K,
            max_ep_len=self.model.max_ep_len,
            reward_scale=self.reward_scale,
            act_discrete=self.model.act_discrete,
        )
        # WeightedRandomSampler
        n_samples = self.max_iters * self.batch_size * self.gradient_accumulation_steps
        sampler = WeightedRandomSampler(
            weights=p_sample,
            num_samples=n_samples,  # Sample as many as the dataset size per epoch
            replacement=True,  # Allow replacement for sampling
        )
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=collator, sampler=sampler
        )
        dataloader_iter = iter(dataloader) 

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        self.model.to(self.device)

        # training loop
        states, actions, rewards, rtgs, tsteps, mask = next(
            dataloader_iter
        )  # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        raw_model = self.model  # unwrap DDP container if needed
        running_mfu = -1.0
        while True:
            # determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num) if self.decay_lr else self.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % self.eval_interval == 0:
                losses = self.estimate_loss(dataloader_iter)
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if losses["val"] < best_val_loss or self.always_save_checkpoint:
                    best_val_loss = losses["val"]
                    if iter_num > 0:
                        checkpoint = {
                            "model": raw_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_args": self.model.args,
                            "iter_num": iter_num,
                            "best_val_loss": best_val_loss,
                            "trainer_args": self.args,
                        }
                        print(f"saving checkpoint to {self.out_dir}")
                        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
                        torch.save(checkpoint, os.path.join(self.out_dir, "ckpt.pt"))
            if iter_num == 0 and self.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.gradient_accumulation_steps):
                # move to device
                states, actions, rewards, rtgs, tsteps, mask = (
                    states.to(self.device),
                    actions.to(self.device),
                    rewards.to(self.device),
                    rtgs.to(self.device),
                    tsteps.to(self.device),
                    mask.to(self.device),
                )

                logits, loss = self.model(
                    states, actions, rtgs, tsteps, mask, targets=actions
                )
                loss = (
                    loss / self.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                # I guess this makes sense when ddp is used, but it was removed
                states, actions, rewards, rtgs, tsteps, mask = next(dataloader_iter)
                # backward pass, with gradient scaling if training in fp16
                loss.backward()
            # clip the gradient
            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # step the optimizer and scaler if training in fp16
            optimizer.step()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = self.estimate_mfu(
                        self.batch_size * self.gradient_accumulation_steps, dt
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                print(
                    f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
                )
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > self.max_iters:
                break

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self, dataloader_iter):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                states, actions, rewards, rtgs, tsteps, mask = next(dataloader_iter)
                states, actions, rewards, rtgs, tsteps, mask = (
                    states.to(self.device),
                    actions.to(self.device),
                    rewards.to(self.device),
                    rtgs.to(self.device),
                    tsteps.to(self.device),
                    mask.to(self.device),
                )
                logits, loss = self.model(
                    states, actions, rtgs, tsteps, mask, targets=actions
                )
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.model.get_num_params()
        cfg = self.model
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.max_ep_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


class NanoDTAgent:
    def __init__(
        self,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True,
        K: int = 20,
        max_ep_len: int = 1000,
        state_dim: int = 1,
        act_dim: int = 1,
        act_discrete: bool = True,
        act_vocab_size: int = 1,
        act_tanh: bool = False,
        tanh_embeddings: bool = False,
    ):
        self.model = DecisionTransformer(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
            K=K,
            max_ep_len=max_ep_len,
            state_dim=state_dim,
            act_dim=act_dim,
            act_discrete=act_discrete,
            act_vocab_size=act_vocab_size,
            act_tanh=act_tanh,
            tanh_embeddings=tanh_embeddings,
        )

    def learn_offline(self, dataset, observation_space, action_space, *args, **kwargs):
        trainer = DecisionTransformerTrainer(self.model, dataset, *args, **kwargs)
        trainer.train()
        return trainer

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    @staticmethod
    def load(self, path, env=None):
        self.model.load_state_dict(torch.load(path))
        return self
