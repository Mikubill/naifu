
import argparse
from typing import Dict, List, Tuple

import os
import hivemind
import requests
import wandb
from hivemind.dht.crypto import RSASignatureValidator
from hivemind.dht.schema import BytesWithPublicKey, SchemaValidator
from hivemind.dht.validation import RecordValidatorBase
from hivemind.utils.logging import get_logger
from hivemind.optim.state_averager import TrainingStateAverager
from hivemind.utils.logging import get_logger, use_hivemind_log_handler

from pydantic import BaseModel, StrictFloat, confloat, conint
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="",
)
args = parser.parse_args()

config = OmegaConf.load(args.config)
logger = get_logger(__name__)

class LocalMetrics(BaseModel):
    step: conint(ge=0, strict=True)
    samples_per_second: confloat(ge=0.0, strict=True)
    samples_accumulated: conint(ge=0, strict=True)
    loss: StrictFloat
    mini_steps: conint(ge=0, strict=True)


class MetricSchema(BaseModel):
    metrics: Dict[BytesWithPublicKey, LocalMetrics]
    
    
class CheckpointHandler:
    def __init__(self, dht: hivemind.DHT):
        self.previous_step = -1

        config = AlbertConfig.from_pretrained(monitor_args.model_config_path)
        self.model = AlbertForPreTraining(config)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        opt = Lamb(
            optimizer_grouped_parameters,
            lr=0.00176,
            weight_decay=0.01,
            clamp_value=10000.0,
            debias=True,
        )

        self.state_averager = TrainingStateAverager(
            dht=dht,
            optimizer=opt,
            scheduler=get_linear_schedule_with_warmup(opt, num_warmup_steps=5000, num_training_steps=125_000),
            prefix=f"{run_id}_state_averager",
            state_compression=hivemind.Float16Compression(),
            bandwidth=optimizer_args.bandwidth,
            client_mode=optimizer_args.client_mode,
            start=True,
            **asdict(averager_args),
        )
        self.previous_timestamp = time.time()


    def is_time_to_save_state(self, cur_step):
        if self.save_checkpoint_step_interval is None:
            return False
        elif cur_step - self.previous_step >= self.save_checkpoint_step_interval:
            return True
        else:
            return False


    def save_state(self, cur_step):
        logger.info("Saving state from peers")
        self.state_averager.load_state_from_peers()
        self.previous_step = cur_step


    def is_time_to_upload(self):
        if self.repo_path is None:
            return False
        elif time.time() - self.previous_timestamp >= self.upload_interval:
            return True
        else:
            return False


    def upload_checkpoint(self, current_loss):
        logger.info("Saving optimizer")
        torch.save(self.state_averager.optimizer.state_dict(), f"{self.repo_path}/optimizer_state.pt")
        self.previous_timestamp = time.time()
        logger.info("Started uploading to Model Hub")
        self.model.push_to_hub(
            repo_name=self.repo_path,
            repo_url=self.repo_url,
            commit_message=f"Step #{current_step}, loss {current_loss:.3f}",
        )
        logger.info("Finished uploading to Model Hub")


def make_validators(run_id: str) -> Tuple[List[RecordValidatorBase], bytes]:
    signature_validator = RSASignatureValidator()
    validators = [SchemaValidator(MetricSchema, prefix=run_id), signature_validator]
    return validators, signature_validator.local_public_key

if __name__ == "__main__":

    dht = hivemind.DHT(
        host_maddrs=["/ip4/0.0.0.0/tcp/0", "/ip4/0.0.0.0/udp/0/quic"], 
        start=True,
        use_ipfs=True,
    )
    print('\n'.join(str(addr) for addr in dht.get_visible_maddrs()))
    print("Global IP:", hivemind.utils.networking.choose_ip_address(dht.get_visible_maddrs()))
    wandb.init(project=args.project)
    current_step = 0
    
    while True:
        metrics_dict = dht.get("metrics", latest=True)
        if metrics_dict is not None:
            metrics_dict = metrics_dict.value
            metrics = [LocalMetrics.parse_obj(metrics_dict[peer].value) for peer in metrics_dict]
            latest_step = max(item.step for item in metrics)

            if latest_step != current_step:
                logger.debug(f"Got metrics from {len(metrics)} peers")

                for i, metrics_for_peer in enumerate(metrics):
                    logger.debug(f"{i} peer {metrics_for_peer}")

                current_step = latest_step
                alive_peers = 0
                sum_loss = 0.0
                num_samples = 0
                sum_perf = 0
                sum_mini_steps = 0

                for item in metrics:
                    sum_loss += item.loss
                    alive_peers += 1
                    sum_perf += item.samples_per_second
                    num_samples += item.samples_accumulated
                    sum_mini_steps += item.mini_steps
                    
                current_loss = sum_loss / sum_mini_steps
                logger.info(f"Step #{current_step}\tloss = {current_loss:.5f}")

                if args.project is not None:
                    wandb.log(
                        {
                            "loss": current_loss,
                            "alive peers": alive_peers,
                            "samples": num_samples,
                            "performance": sum_perf,
                            "step": latest_step,
                        }
                    )

                if args.store_checkpoints:
                    if checkpoint_handler.is_time_to_save_state(current_step):
                        checkpoint_handler.save_state(current_step)
                        if checkpoint_handler.is_time_to_upload():
                            checkpoint_handler.upload_checkpoint(current_loss)
        logger.debug("Peer is still alive...")
        time.sleep(monitor_args.refresh_period)