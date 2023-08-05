# Compatibility for different versions of libraries
import lightning.pytorch as pl

def pl_compat_fix(config, callbacks):
    
    # remove some params since its changed from lightning.pytorch 1.9
    major, minor, _ = pl.__version__.split('.')
    if int(major) >= 2:
        # gpus=x => devices=x
        x = config.lightning.get("gpus")
        if x:
            config.lightning.devices = config.lightning.gpus
            
        x = config.lightning.get("auto_select_gpus")
        if x:
            config.lightning.devices = "auto"
            
        x = config.trainer.get("init_batch_size")
        if x:
            config.trainer.batch_size = x
        
        # before: 
        # trainer = Trainer(accumulate_grad_batches={"1": 5, "10": 3})
        #
        # after: 
        # from lightning.pytorch.callbacks import GradientAccumulationScheduler
        # trainer = Trainer(callbacks=GradientAccumulationScheduler({"1": 5, "10": 3}))
        if isinstance(config.lightning.get("accumulate_grad_batches"), dict):
            from lightning.pytorch.callbacks import GradientAccumulationScheduler
            callbacks.append(GradientAccumulationScheduler(config.lightning.accumulate_grad_batches))
            del config.lightning.accumulate_grad_batches
            
        x = config.lightning.get("auto_lr_find")
        if x: 
            from lightning.pytorch.callbacks import BatchSizeFinder
            callbacks.append(BatchSizeFinder())
            
        x = config.lightning.get("auto_scale_batch_size")
        if x:
            from lightning.pytorch.callbacks import LearningRateFinder
            callbacks.append(LearningRateFinder())
            
        x = config.lightning.get("replace_sampler_ddp")
        if x != None:
            config.lightning.use_distributed_sampler = x
            
        for keys in ["gpus", "auto_select_gpus", "auto_lr_find", "auto_scale_batch_size", "replace_sampler_ddp", "move_metrics_to_cpu"]:
            if keys in config.lightning:
                del config.lightning[keys]
        
    return config, callbacks