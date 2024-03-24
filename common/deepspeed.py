import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from lightning.fabric.strategies import DeepSpeedStrategy

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls = {LlamaDecoderLayer}
)

ds_strategy = DeepSpeedStrategy(
    stage=3,
    config={
        # "fp16": {
        #     "enabled": "auto",
        #     "loss_scale": 0,
        #     "loss_scale_window": 1000,
        #     "initial_scale_power": 16,
        #     "hysteresis": 2,
        #     "min_loss_scale": 1
        # },
        "bf16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
            "ignore_unused_parameters": True,
        },
        "zero_allow_untested_optimizer": True,
    }
)   


def _strategy():
    return ds_strategy