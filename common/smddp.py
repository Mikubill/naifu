
import os
from lightning.pytorch.plugins.environments import LightningEnvironment

from common.fairscale import DDPShardedStrategy
ddp_strategy = DDPShardedStrategy

# from lightning.fabric.strategies import DDPStrategy
# ddp_strategy = DDPStrategy

def setup():
    env = LightningEnvironment()
    env.world_size = lambda: int(os.environ["WORLD_SIZE"])
    env.global_rank = lambda: int(os.environ["RANK"])
    strategy = ddp_strategy(
        cluster_environment=env, 
        accelerator="gpu"
    )
    return strategy