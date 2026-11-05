# Token-history Markov attention

The Qwen3-4B baseline backbone is unchanged. The new `token_attention` head
uses 8 query heads, 8 KV heads, head dimension 128, and Markov rank 256.
It replaces target-feature KV with token embeddings from the same W1 used by
the queries:

```
e[p] = W1(token[p])
q[p] = RoPE(QNorm(Q(e[p])), p)
k[j] = RoPE(KNorm(K(e[j])), j)
v[j] = V(e[j])
a[p] = O(Attention(q[p], k[j <= p], v[j <= p]))
logits[p+1] = backbone_logits[p+1] + W2(OutputNorm(e[p] + a[p]))
```

Q/K/V project 256 to 1024; O projects 1024 to 256. OutputNorm is RMSNorm
with trainable scale and bias, initialized to one and zero. There is no
activation or gate after the residual. This normalization means zero attention
does not exactly recover the vanilla head.

Training constructs KV once per sequence. Each anchor query can read its own
input token and earlier tokens, including teacher-forced tokens inside the block,
but cannot read the token it predicts. Batch elements are independent. The
existing right-padded training layout and valid-label mask ensure padding lies
beyond all scored queries. Prompt tokens remain visible even when their loss mask
is zero; loss masks must not be used as context masks.

Use the existing online training recipe with:

```bash
model.draft_model_config=configs/qwen3-4b-dspark-markov-token-attn.json
```

Use a new run/output directory. Checkpoint/export uses the same new JSON:

```bash
specforge export --to hf --checkpoint "$CHECKPOINT" \
  --draft-config configs/qwen3-4b-dspark-markov-token-attn.json \
  --output-dir "$EXPORT_DIR"
```

Old context-GQA checkpoints are not compatible with this head's KV projections
and output normalization. The new head has a separate resume contract.

The training-side generation helper appends sampled input-token KV inside each
block and rebuilds the committed prefix for the next block, excluding rejected
tokens. This is a correctness-oriented helper, not a persistent serving cache.
vLLM support is a separate change; the existing context-GQA inference branch
cannot serve this architecture.
