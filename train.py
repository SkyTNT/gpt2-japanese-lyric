import math
import pickle
import random
import time
import os
import collections

import numpy as np
import torch
import torch.optim as optim
import torch.cuda.amp as amp
from torch.nn import functional as F
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import T5Tokenizer

from torch.optim.lr_scheduler import LambdaLR

seed = 42
data_dir = "lyric_ids.pkl"
model_size = 'small'
max_seq_len = 320
batch_size = 5
n_accum_steps = 8
n_warmup_steps = 2e3
n_training_steps = 1e6
n_epochs = 300

check_loss_after_n_step = 40
save_after_n_step = 1000

use_amp = True

l2_penalty = 0.01
init_lr = 6e-4
max_grad_norm = 1.0

checkpoint_path = "./exp/last.checkpoint"
output_dir = "./exp/"


def mlog(s):
    with open(output_dir + "log.txt", "a+", encoding="utf-8") as log_f:
        log_f.write(s + "\n")
    print(s)


def collate_fn(batch_data):
    return batch_data


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class DataSource(torch.utils.data.Dataset):

    def __init__(self, max_seq_len_, tokenizer, docs, stage, randomize):
        # Attributes
        self.max_seq_len = max_seq_len_
        # Other attributes
        self.tokenizer = tokenizer
        self.stage = stage
        self.randomize = randomize
        self.statistics = {"n_doc": 0}

        # Load dataset
        self.docs = docs

        # Calculate basic statistics
        self.statistics["n_doc"] = len(self.docs)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        doc = [self.tokenizer.bos_token_id] + doc + [self.tokenizer.eos_token_id]
        if self.randomize:
            start_idx = random.randrange(0, max(1, len(doc) - self.max_seq_len))
            end_idx = min(start_idx + self.max_seq_len, len(doc) - 1)
            doc = doc[start_idx: end_idx + 1]
        else:
            doc = doc[:self.max_seq_len]

        return doc


class StatisticsReporter:
    def __init__(self):
        self.statistics = collections.defaultdict(list)

    def update_data(self, d):
        for k, v in d.items():
            if isinstance(v, (int, float)):
                self.statistics[k] += [v]

    def clear(self):
        self.statistics = collections.defaultdict(list)

    def to_string(self):
        string_list = []
        for k, v in sorted(list(self.statistics.items()), key=lambda x: x[0]):
            mean = np.mean(v)
            string_list.append("{}: {:.5g}".format(k, mean))
        return ", ".join(string_list)

    def get_value(self, key):
        if key in self.statistics:
            value = np.mean(self.statistics[key])
            return value
        else:
            return None

    def items(self):
        for k, v in self.statistics.items():
            yield k, v


def forward_step(model, tokenizer, batch_data):
    max_len = max([len(seq) for seq in batch_data])

    # padding input sequences
    batch_data = [seq + [tokenizer.pad_token_id] * (max_len - len(seq)) for seq in batch_data]

    # convert to tensors
    batch_tensor = torch.LongTensor(batch_data).to(model.device)

    # get inputs and outputs
    input_ids = batch_tensor[:, :-1].contiguous()
    output_ids = batch_tensor[:, 1:].contiguous()

    # forward
    gpt2_outputs = model(input_ids=input_ids, return_dict=True)
    loss = F.cross_entropy(
        gpt2_outputs["logits"].view(-1, len(tokenizer)),
        output_ids.view(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction="mean"
    )
    with torch.no_grad():
        ppl = loss.exp()

    return loss, ppl


def gen_lyric(prompt_text: str):
    prompt_tokens = tokenizer.tokenize(prompt_text)
    prompt_token_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
    prompt_tensor = torch.LongTensor(prompt_token_ids).to(device)
    prompt_tensor = prompt_tensor.view(1, -1)
    # model forward
    output_sequences = model.generate(
        input_ids=prompt_tensor,
        max_length=512,
        top_p=0.95,
        top_k=40,
        temperature=1.0,
        do_sample=True,
        early_stopping=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1
    )

    # convert model outputs to readable sentence
    generated_sequence = output_sequences.tolist()[0]
    generated_tokens = tokenizer.convert_ids_to_tokens(generated_sequence)
    generated_text = tokenizer.convert_tokens_to_string(generated_tokens)
    generated_text = "\n".join([s.strip() for s in generated_text.split('\n')]).replace(' ', '\u3000').replace(
        '</s>', '\n\n---end---')
    return generated_text


def get_conf(n_embd, n_layer, n_head, n_inner, n_ctx):
    return GPT2Config(
        vocab_size=32000,
        n_positions=1024,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False,
        model_type="gpt2-ja",
        n_ctx=n_ctx,
        task_specific_params={"text-generation": {"do_sample": True, "max_length": 500}},
        tokenizer_class="T5Tokenizer",
        architectures=["GPT2LMHeadModel"]
    )


model_conf = {
    'xsmall': get_conf(512, 6, 8, 2304, 768),
    'small': get_conf(768, 12, 12, 3072, 1024),
    'medium': get_conf(1024, 24, 16, 4096, 1024),
}

if __name__ == '__main__':
    tokenizer = T5Tokenizer(
        vocab_file="jp_google_sp3.model",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        extra_ids=0,
        additional_special_tokens=['\n'],
        do_lower_case=True
    )
    tokenizer.sanitize_special_tokens()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    with open(data_dir, "rb") as f:
        docs = pickle.load(f)
    random.shuffle(docs)
    eval_docs = docs[0:1024]
    docs = docs[1024:]

    train_data_source = DataSource(max_seq_len, tokenizer, docs, "train", randomize=True)
    print(str(train_data_source.statistics))
    train_data_sampler = RandomSampler(
        train_data_source,
        replacement=False
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_data_source,
        batch_size=batch_size,
        sampler=train_data_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )
    eval_data_source = DataSource(max_seq_len, tokenizer, eval_docs, "eval", randomize=False)
    print(str(eval_data_source.statistics))
    eval_data_sampler = RandomSampler(
        eval_data_source,
        replacement=False
    )
    eval_dataloader = torch.utils.data.DataLoader(
        train_data_source,
        batch_size=batch_size,
        sampler=eval_data_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = GPT2LMHeadModel(model_conf[model_size])
    model.train()
    model = model.to(device)

    # build optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln']  # no decay for bias and LayerNorm (ln)
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': l2_penalty},
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=init_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=l2_penalty
    )

    # build lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=n_warmup_steps,
        num_training_steps=n_training_steps,
    )

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("loading model state dict...")
        model.load_state_dict(checkpoint["model"])
        model.tie_weights()  # NOTE: don't forget to tie weights after loading weights
        print("loading optimizer state dict...")
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("recovering lr scheduler...")
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        print("recovering others...")
        n_step = checkpoint["n_step"]
        start_n_epoch = checkpoint["n_epoch"]
        best_ppl = checkpoint.get("best_ppl", float("inf"))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        n_step = 0
        start_n_epoch = 0
        best_ppl = float("inf")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    trn_reporter = StatisticsReporter()
    eval_reporter = StatisticsReporter()

    if isinstance(train_data_sampler, DistributedSampler):
        train_data_sampler.set_epoch(start_n_epoch)
    if use_amp:
        scaler = amp.GradScaler()
    torch.cuda.empty_cache()
    for epoch_idx in range(start_n_epoch, n_epochs):
        for batch_data in train_dataloader:
            n_step += 1

            # stop if reaches the maximum tranining step
            if n_step >= n_training_steps:
                break

            # forward
            model.train()
            with amp.autocast():
                loss, ppl = forward_step(model, tokenizer, batch_data)
            trn_reporter.update_data({"ppl": ppl.item(), "loss": loss.item()})

            # backward
            loss /= n_accum_steps
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            del loss

            if n_step % n_accum_steps == 0:
                # clip gradient
                if max_grad_norm > 0.0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # update model parameters
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # zero gradients
                optimizer.zero_grad()

            # check loss
            if n_step > 0 and n_step % check_loss_after_n_step == 0:
                lr = list(lr_scheduler.optimizer.param_groups)[0]["lr"]
                log_s = f"{time.time() - start_time:.2f}s Epoch {epoch_idx}, step {n_step}, lr {lr:.5g} - "
                log_s += trn_reporter.to_string()
                mlog(log_s)
                trn_reporter.clear()

            # save
            if n_step > 0 and n_step % save_after_n_step == 0:
                model.eval()
                model_to_save = model.module if hasattr(model, 'module') else model
                with torch.no_grad():
                    for eval_batch_idx, eval_batch_data in enumerate(eval_dataloader):
                        with amp.autocast():
                            loss, ppl = forward_step(model_to_save, tokenizer, eval_batch_data)
                        eval_reporter.update_data({"ppl": ppl.item(), "loss": loss.item()})

                        if eval_batch_idx == len(eval_dataloader) - 1:
                            break
                del loss
                log_s = f"<Eval> - {time.time() - start_time:.3f}s - "
                log_s += eval_reporter.to_string()
                mlog(log_s)

                print(gen_lyric(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(random.choice(docs))).split('\n')[0]))

                # save current model
                checkpoint = {
                    "model": model_to_save.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "n_epoch": epoch_idx,
                    "n_step": n_step,
                    "best_ppl": best_ppl
                }
                torch.save(
                    checkpoint,
                    f"{output_dir}last.checkpoint"
                )
                mlog(f"checkpoint saved to {output_dir}last.checkpoint")
                # save best model
                cur_ppl = eval_reporter.get_value("ppl")
                if cur_ppl < best_ppl:
                    best_ppl = cur_ppl

                    torch.save(
                        checkpoint,
                        f"{output_dir}best.checkpoint"
                    )
                    mlog(f"best checkpoint saved to {output_dir}best.checkpoint")
                eval_reporter.clear()
            # decay learning rate
            lr_scheduler.step()
