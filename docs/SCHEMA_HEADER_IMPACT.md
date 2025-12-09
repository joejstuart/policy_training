# Schema Header Impact on Token Length

## Analysis

Schema headers add approximately **25-45 tokens** per example (depending on query type):
- Task queries: ~38 tokens
- Material queries: ~30 tokens  
- Subject queries: ~28 tokens
- Bundle queries: ~45 tokens (longer due to ref structure explanation)

## Current Situation

**Without schema headers:**
- Average: ~755 tokens
- 95th percentile: ~1,453 tokens
- Coverage at 1536: 95.2% (2,674/2,808 examples)

**With schema headers:**
- Average: ~785 tokens (+30 tokens)
- 95th percentile: ~1,480 tokens (+27 tokens)
- Estimated coverage at 1536: ~94.5-95.0% (slight decrease)

## Recommendation

### Option 1: Keep max_seq_len=1536 (Recommended)

**Pros:**
- ✅ Minimal impact (~0.5% coverage loss)
- ✅ No memory increase needed
- ✅ Still covers ~95% of examples
- ✅ Schema headers help model learn faster (worth small coverage trade-off)

**Cons:**
- ⚠️ ~10-15 more examples will be truncated (still acceptable)

**Verdict**: **Keep 1536** - The benefit of schema headers outweighs the small coverage loss.

### Option 2: Increase to max_seq_len=1600

**Pros:**
- ✅ Maintains ~95.2% coverage
- ✅ Only +64 tokens (4% increase)
- ✅ Small memory impact (~8% more attention memory)

**Cons:**
- ⚠️ Slightly more memory usage
- ⚠️ May push some GPUs closer to limits

**Verdict**: **Optional** - Only if you want to maintain exact same coverage.

### Option 3: Increase to max_seq_len=1700

**Pros:**
- ✅ Covers ~96% of examples
- ✅ Good margin for schema headers

**Cons:**
- ⚠️ +164 tokens (11% increase)
- ⚠️ Significant memory impact (~24% more attention memory)
- ⚠️ May cause NVML errors on some GPUs

**Verdict**: **Not recommended** - Too much memory increase for marginal benefit.

## Memory Impact Calculation

Memory scales roughly quadratically with sequence length (attention mechanism):

| max_seq_len | Memory vs 1536 | Status |
|-------------|----------------|--------|
| 1536 | Baseline | ✅ Current stable |
| 1600 | +8% | ⚠️ May work, test first |
| 1700 | +24% | ❌ Likely too much |
| 2048 | +78% | ❌ Too much |

## Final Recommendation

**Keep `max_seq_len=1536`** with your current configuration:
- Batch size: 4
- Grad accum: 4
- Schema headers: Enabled

**Reasoning:**
1. Schema headers add only ~30 tokens on average
2. Coverage loss is minimal (~0.5%)
3. Schema headers provide significant learning benefit
4. No memory increase needed
5. Current setup is stable

**If you want to maintain exact coverage**, you can try `max_seq_len=1600`, but test for NVML errors first. The small coverage loss at 1536 is acceptable given the learning benefits of schema headers.

