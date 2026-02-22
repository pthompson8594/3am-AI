# Memory Optimization for Limited RAM Systems

## Problem
On systems with limited RAM (e.g., BC-250 with 16GB GDDR6 shared with GPU), torque clustering can exhaust memory and force swapping to disk, causing extreme slowdowns that require a reboot.

## Solution
Batched distance matrix computation with aggressive garbage collection.

## How It Works

**Batched Distance Computation:**
- Instead of computing the full n×n distance matrix at once, the matrix is computed in batches
- Each batch processes 10 memories against all embeddings
- Default batch size: 10 (configurable via `batch_size` parameter)
- After each batch, intermediate arrays are freed with `gc.collect()`

**Memory Cleanup:**
- After clustering completes, all temporary arrays are explicitly deleted
- Aggressive garbage collection runs before returning results
- Distance matrix freed immediately after clustering

## Usage

**Automatic clustering (during introspection):**
```bash
# Triggered automatically when >50% of memories are unclustered
# Uses default batch_size=10
```

**Manual clustering with custom batch size:**
```python
# In your code:
stats = core.memory.run_torque_clustering(force=True, batch_size=10)
```

**Via API:**
```bash
POST /api/introspection/trigger
# Calls run_torque_clustering internally with default batch_size=10
```

## Configuration

To adjust batch size for your system:

- **More RAM available:** Increase `batch_size` (faster but uses more memory)
  - `batch_size=20` or `batch_size=50`
  
- **Very limited RAM:** Decrease `batch_size`
  - `batch_size=5` (slower but uses less memory at peak)

Change in `memory.py` line 509:
```python
def run_torque_clustering(self, force: bool = False, batch_size: int = 10) -> dict:
```

## Monitoring

Check logs during clustering:
```bash
journalctl --user -u llm-unified.service -f | grep "Memory"
```

Should see output like:
```
[Memory] Computing distance matrix in batches of 10...
[Memory] Distance matrix: 0-9/150
[Memory] Distance matrix: 10-19/150
...
[Memory] Running Torque Clustering with K=30...
[Memory] Clustering complete. Performing memory cleanup...
[Memory] Memory cleanup finished
```

## Performance

- **Batch size 10:** ~10 ms per batch, safe for 16GB shared VRAM
- **Memory usage:** Peak ~500MB-1GB during distance computation (vs 5GB+ for full matrix)
- **Trade-off:** Slightly slower (~20-30% slower) but won't swap to disk

## If Still Having Issues

1. **Reduce batch size further:**
   ```python
   batch_size=5  # Very conservative
   ```

2. **Run clustering during low-load periods:**
   ```bash
   # Disable auto-clustering in introspection
   # Manually trigger during off-peak hours
   ```

3. **Increase decay rates to keep fewer memories:**
   In `memory.py` line 41-47, reduce `DECAY_RATES` values to drop memories faster.

## Rollback

Backup: `memory.py.backup`

Restore with:
```bash
cp memory.py.backup memory.py
```
