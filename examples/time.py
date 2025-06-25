"""Timing the performance of scalar logging with goggles."""

import time

import goggles

goggles.new_wandb_run("latency-test", {})

N = 1000
start = time.time()
for i in range(N):
    goggles.scalar("test_scalar", i)
end = time.time()

total = end - start
print(f"{N} calls in {total:.4f}s → {total/N*1e6:.1f} μs per call")
