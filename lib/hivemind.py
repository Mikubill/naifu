
from functools import partial
from hivemind import Float16Compression, Uniform8BitQuantization
from hivemind.compression import SizeAdaptiveCompression
from lightning.pytorch.strategies import HivemindStrategy
from hivemind.optim.power_sgd_averager import PowerSGDGradientAverager
from .model import get_class


def init_hivemind(config):
    compression = SizeAdaptiveCompression(
        threshold=2 ** 16 + 1, less=Float16Compression(), greater_equal=Uniform8BitQuantization()
    )

    return HivemindStrategy(
        scheduler_fn=partial(
            get_class(config.lr_scheduler.name),
            **config.lr_scheduler.params
        ),
        grad_compression=compression,
        state_averaging_compression=compression,
        grad_averager_factory=partial(PowerSGDGradientAverager, averager_rank=32, min_compression_ratio=0.5),
        **config.hivemind
    )
