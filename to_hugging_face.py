import os
import torch
from transformers import GPT2LMHeadModel, GPT2Config, TFGPT2LMHeadModel
from transformers import T5Tokenizer


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
    save_dir = "hf/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 不加sep_token
    tokenizer = T5Tokenizer(
        vocab_file="google_sp.model",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        extra_ids=0,
        additional_special_tokens=(),
        do_lower_case=True
    )

    tokenizer.save_pretrained(save_dir)
    tokenizer.save_vocabulary(save_dir)

    checkpoint = torch.load("./exp/last.checkpoint", map_location="cpu")
    model = GPT2LMHeadModel(model_conf["small"])
    model.load_state_dict(checkpoint["model"])
    model.save_pretrained(save_dir)

    tf_model = TFGPT2LMHeadModel.from_pretrained(f"{save_dir}", from_pt=True)
    tf_model.save_pretrained(save_dir)
