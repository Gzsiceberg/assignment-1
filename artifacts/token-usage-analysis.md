# Token usage: compaction vs. no compaction

## Scope and measurement

The main comparison uses the same Django SWE-bench task and model (`deepseek-v4-flash`):

- **No compaction:** `django__django-15368-baseline-trajectory.json`
- **Compaction:** `django__django-15368-trajectory.json`, run with `COMPACT_THRESHOLD=6000`, a 1,800-token summary-completion limit, and one recent complete assistant/tool step retained.

Counts below come from recorded API `usage`, not character-based token estimates. Action calls are in `responses`; summary calls are stored separately in `compactions[*].compaction_response`. All recorded calls have usage data. Input totals include cached input; output totals include reasoning. Cached and reasoning tokens are subsets, not additional tokens to add to those totals.

These are individual runs, not repeated controlled trials. Their action sequences and lengths differ, so differences cannot be attributed solely to compaction.

## Main results

| Metric | No compaction | Compaction |
|---|---:|---:|
| Action/model calls | 26 | 43 |
| Summary calls | 0 | 9 |
| Total API calls | 26 | 52 |
| Action input tokens | 238,785 | 165,101 |
| Action output tokens | 3,309 | 5,834 |
| Summary input tokens | 0 | 23,030 |
| Summary output tokens | 0 | 6,493 |
| **All input tokens** | **238,785** | **188,131** |
| **All output tokens** | **3,309** | **12,327** |
| **All tokens, including summary overhead** | **242,094** | **200,458** |
| Mean input per action call | 9,184 | 3,840 |
| Median input per action call | 9,045.5 | 3,979 |
| Peak input per action call | 14,623 | 5,953 |
| First → last action input | 1,392 → 14,623 | 1,392 → 4,122 |

Compaction reduced total recorded tokens by **41,636 (17.2%)**, including its own overhead. Total input fell **21.2%**. Mean and peak action-input sizes fell **58.2%** and **59.3%**, respectively.

However, the compacted run made **65.4% more action calls**. Its nine summary calls added **29,523 tokens**, or **14.7%** of its total usage. Output usage increased from 3,309 to 12,327 tokens; summarization alone produced more output tokens than all action responses in the baseline.

## How context changes over time

Without compaction, action input increased monotonically from **1,392** to **14,623** tokens: each request resent the accumulated history, including earlier tool observations.

With compaction, input followed a sawtooth pattern. For example:

- Action calls 10–12: **5,518 → 5,953 → 2,276** input tokens.
- Calls 18–21: **3,970 → 4,959 → 5,560 → 2,163**.
- Calls 36–39: **5,209 → 5,397 → 2,697 → 3,122**.

The drops occur when old context is replaced by a checkpoint while recent messages remain verbatim. This run kept action inputs below 6,000 tokens, but the configured threshold is an estimated trigger, **not a guaranteed hard ceiling**.

### Summary-call overhead

Action count indicates how many action requests had completed when compaction occurred. The before/after columns are recorded **rough active-context estimates**, not API-measured summary-input tokens.

| After action | Estimated context before → after | Summary API input | Summary API output |
|---|---:|---:|---:|
| 11 | 6,443 → 1,399 | 3,588 | 1,113 |
| 14 | 5,920 → 2,004 | 2,395 | 683 |
| 17 | 6,218 → 3,257 | 1,993 | 760 |
| 20 | 5,753 → 1,374 | 2,570 | 707 |
| 22 | 7,520 → 4,880 | 1,788 | 420 |
| 23 | 5,766 → 1,617 | 1,374 | 330 |
| 26 | 5,543 → 1,667 | 2,464 | 994 |
| 31 | 5,934 → 1,642 | 2,968 | 694 |
| 38 | 6,554 → 1,713 | 3,890 | 792 |

The combined before/after estimates decreased from **55,651 to 19,553**, a **64.9%** reduction across these snapshots. This is not an additional API-token saving to add to the totals above. All nine summaries finished with `stop`; none exhausted the 1,800-token limit. Summary output included **3,074 reasoning tokens** out of 6,493 output tokens, illustrating why visible-summary length alone does not determine the required completion budget.

## Cache tradeoff: fewer tokens does not necessarily mean lower cost

| Input-cache metric, all calls | No compaction | Compaction |
|---|---:|---:|
| Reported cached input tokens | 219,648 | 113,664 |
| Cached fraction of input | 92.0% | 60.4% |
| Input minus reported cached tokens | 19,137 | 74,467 |

The baseline's stable, growing conversation prefix is highly cacheable. Replacing old history changes that prefix, and summary requests introduce additional inputs. In this run, compaction used **3.89× as many input tokens not reported as cached**, despite using fewer input tokens overall. Of the compacted run's cached input, only 1,536 tokens came from summary calls.

### Cache-weighted cost estimate

The configured endpoint is **Poe**, so use Poe's model rates rather than assuming DeepSeek's direct-API pricing. Poe's [public model catalog](https://api.poe.com/v1/models) lists these rates for `deepseek-v4-flash` at analysis time (USD per million tokens):

| Token class | Rate |
|---|---:|
| Uncached input | $0.4444 |
| Cached input (`input_cache_read`) | $0.0283 |
| Output, including reasoning | $1.3333 |

Cached input is approximately **15.7× cheaper** than uncached input. Count it once at its discounted rate—not as free input, and not as additional tokens on top of `prompt_tokens`:

```text
uncached_input = prompt_tokens - cached_tokens
input_cost = (uncached_input × 0.4444 + cached_tokens × 0.0283) / 1,000,000
output_cost = completion_tokens × 1.3333 / 1,000,000
total_cost = input_cost + output_cost
```

| Estimated cost, all action and summary calls | No compaction | Compaction |
|---|---:|---:|
| Uncached input | $0.008504 | $0.033093 |
| Cached input | $0.006216 | $0.003217 |
| **All input** | **$0.014721** | **$0.036310** |
| Output | $0.004412 | $0.016436 |
| **Total** | **$0.019132** | **$0.052745** |

After accounting for the cache discount, the compacted run's **input cost is approximately 2.47× higher**, and its **total cost is approximately 2.76× higher**, despite consuming 17.2% fewer raw tokens. The baseline repeatedly sends a large prefix, but most of it is cheap cache-hit input. Compaction reduces that prefix while increasing full-price input and expensive output generation.

These are **rate-based estimates, not verified bills**: they apply the currently published catalog prices to the cache counts present in the saved responses. The files do not establish historical prices or actual charges. Poe's [OpenAI-compatible API documentation](https://creator.poe.com/docs/external-applications/openai-compatible-api) describes `prompt_tokens_details` as empty, whereas these trajectories contain `cached_tokens`; this discrepancy is another reason to verify charges through the [Poe Usage API](https://creator.poe.com/docs/resources/usage-api), which reports costs and cache discounts. No billing-history request was made for this analysis.

Latency also cannot be determined from these usage records: shorter action contexts compete with more sequential requests and summarization work.

## Interpretation and tradeoffs

- **Compaction bounds active context.** It avoids repeatedly carrying large old observations, leaves room for future actions, and reduces context-window pressure.
- **Compaction is lossy.** Summaries and truncated serialization can omit details needed later, potentially causing re-reading or repeated investigation. Retaining the latest complete tool pair helps, but does not preserve all historical evidence.
- **No compaction preserves detail and cache locality.** Exact prior commands, outputs, and reasoning remain available, at the cost of increasing context size and eventual context-window pressure.
- **Compaction adds work.** Here it doubled total API calls (26 → 52) and increased output usage substantially. The longer action trajectory is consistent with extra investigation, but one run cannot establish that information loss caused it.
- **Result quality must be checked separately.** Both runs ended through `send_message`, and their saved patch files are byte-identical. That is useful evidence of matching deliverables, not proof that the patch passes evaluation.

**Bottom line:** for these runs, compaction substantially reduced per-action context and modestly reduced aggregate tokens, but sacrificed cache reuse and added summary/output overhead. At the cited Poe rates, its estimated total cost is **$0.052745 versus $0.019132 without compaction (2.76× higher)**. Compaction helped context capacity, not estimated cost in this comparison. Repeated runs with patch evaluation, timing, and verified billing would be needed to compare efficiency and reliability conclusively.
