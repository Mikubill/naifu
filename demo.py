
import torch, transformers, time, os
from lib.sgm import UNetModel

# Hide welcome message from bitsandbytes
os.environ.update({"BITSANDBYTES_NOWELCOME": "1"})

def main():

    batch_size = 4
    steps = 50
    ADM_IN_CHANNELS = 2816
    
    # Model Train (50 steps)
    model = UNetModel(
        adm_in_channels=2816,
        num_classes='sequential',
        use_checkpoint=True,
        in_channels=4,
        out_channels=4,
        model_channels=320,
        attention_resolutions=[4, 2],
        num_res_blocks=2,
        channel_mult=[1, 2, 4],
        num_head_channels=64,
        use_spatial_transformer=True,
        use_linear_in_transformer=True,
        transformer_depth=[1, 2, 10],  # note: the first is unused (due to attn_res starting at 2) 32, 16, 8 --> 64, 32, 16
        context_dim=2048,
        spatial_transformer_attn_type='softmax',
        legacy=False,
    ).cuda()
    
    optimizer = transformers.optimization.Adafactor(model.parameters(), relative_step=True)  
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for step in range(steps):
        print(f"step {step}")
        if step == 1:
            time_start = time.perf_counter()

        x = torch.randn(batch_size, 4, 128, 128).cuda()  # 1024x1024
        t = torch.randint(low=0, high=10, size=(batch_size,), device="cuda")
        ctx = torch.randn(batch_size, 77, 2048).cuda()
        y = torch.randn(batch_size, ADM_IN_CHANNELS).cuda()

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output = model(x, t, ctx, y)
            target = torch.randn_like(output)
            loss = torch.nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    time_end = time.perf_counter()
    print(f"elapsed time: {time_end - time_start} [sec] for last {steps - 1} steps")
    

if __name__ == "__main__":
    main()
    