import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from lightning.fabric.strategies import FSDPStrategy

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls = {LlamaDecoderLayer}
)

fsdp_strategy = FSDPStrategy(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    forward_prefetch=True,
    auto_wrap_policy=auto_wrap_policy,
    limit_all_gathers=True,
    activation_checkpointing=[LlamaDecoderLayer],
)

def _strategy():
    return fsdp_strategy