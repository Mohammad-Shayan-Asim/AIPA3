import torch
import gc
import calflops
import time
import threading
from torch import nn


def median(x):
    return sorted(x)[len(x)//2]


def profile(model: nn.Module, input_shape: tuple, device: torch.device, vocab_size: int = 50257):
    with torch.no_grad():
        model.to(device)
        warmup_its = 10
        bench_its  = 30

        gc.collect()
        torch.cuda.empty_cache()

        x = torch.randint(0, vocab_size, input_shape, device=device)

        for _ in range(warmup_its):
            model(x, use_cache=False)

        torch.cuda.synchronize()

        flops, macs, params = calflops.calculate_flops(
            model=model, input_shape=input_shape
        )

        gc.collect()
        torch.cuda.empty_cache()

        latencies       = []  # seconds
        peak_mem_usages = []  # bytes
        mem_usages      = []  # bytes
        power_samples   = []  # milliwatts

        for _ in range(bench_its):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            stop_flag        = threading.Event()
            run_power_samples = []
            run_mem_usages   = []

            def sample_power():
                while not stop_flag.is_set():
                    try:
                        run_power_samples.append(torch.cuda.power_draw())
                        mem_usages.append(torch.cuda.memory_allocated() - start_mem)
                    except RuntimeError:
                        pass
                    stop_flag.wait(0.01)

            sampler = threading.Thread(target=sample_power)
            sampler.start()

            start_mem  = torch.cuda.memory_allocated()
            start_time = time.perf_counter()
            model(x, use_cache=False)

            torch.cuda.synchronize()
            end_time         = time.perf_counter()
            peak_mem_usage   = torch.cuda.max_memory_allocated() - start_mem
            latencies.append(end_time - start_time)
            peak_mem_usages.append(peak_mem_usage)
            stop_flag.set()
            sampler.join()

            power_samples.extend(run_power_samples)
            mem_usages.extend(run_mem_usages)

        total_latency = median(latencies)
        peak_mem      = median(peak_mem_usages)

        mem_avg     = sum(mem_usages)    / len(mem_usages)    if mem_usages    else 0.0
        avg_power_w = sum(power_samples) / len(power_samples) if power_samples else 0.0
        energy_j    = avg_power_w * median(latencies)

        return {
            "name":          model.__class__.__name__,
            "total_latency": total_latency,
            "parameters":    params,
            "macs":          macs,
            "flops":         flops,
            "gpu_mem_peak":  peak_mem,
            "gpu_mem_avg":   mem_avg,
            "avg_power_w":   avg_power_w,
            "energy_j":      energy_j,
        }
