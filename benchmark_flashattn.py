import torch
import torch.utils.benchmark as benchmark
from flash_attn import flash_attn_func

def fattn(q, k, v):
    out = flash_attn_func(q, k, v)
    return out

def torch_fattn(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)

def run_benchmark():
    device = "cuda"
    dtype = torch.float16
    torch.random.manual_seed(0)

    batch_size = 16
    nheads = 48
    nheads_kv = 48
    d = 64
    seqlen_q = 20480
    seqlen_k = 20480

    # Setup for torch_fattn
    q_torch = torch.randn(batch_size, nheads, seqlen_q, d, device=device, dtype=dtype, requires_grad=True)
    k_torch = torch.randn(batch_size, nheads, seqlen_k, d, device=device, dtype=dtype, requires_grad=True)
    v_torch = torch.randn(batch_size, nheads, seqlen_k, d, device=device, dtype=dtype, requires_grad=True)

    # Setup for fattn3
    q_flash = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k_flash = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True)
    v_flash = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype, requires_grad=True)

    # Benchmark torch_fattn
    t0 = benchmark.Timer(
        stmt='torch_fattn(q, k, v)',
        setup='from __main__ import torch_fattn',
        globals={'q': q_torch, 'k': k_torch, 'v': v_torch}
    )
    
    # Benchmark fattn3
    t1 = benchmark.Timer(
        stmt='fattn(q, k, v)',
        setup='from __main__ import fattn',
        globals={'q': q_flash, 'k': k_flash, 'v': v_flash}
    )

    torch.cuda.synchronize()
    print("torch.nn.functional.scaled_dot_product_attention:")
    print(t0.timeit(1000))
    
    torch.cuda.synchronize()
    print("\nflash_attn:")
    print(t1.timeit(1000))

if __name__ == "__main__":
    run_benchmark()