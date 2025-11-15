# ROOT CAUSE ANALYSIS: Why v1 Keeps Winning

## The Problem

Both Ethereum and IBM user-level datasets showed the **same pattern**:
- v1 (simple graph conv) achieves near-perfect performance
- v3-v5 (complex temporal models) degrade or don't improve

## The Root Cause: Feature Engineering Paradox

### What We Did Wrong

```
Raw Transactions → Aggregate by User → "Perfect" Features → Train GNN
     (1M txns)         (75 users)      (95.7% F1 baseline)   (Can't improve!)
```

When we aggregate transactions into user-level statistics:
- **Amount mean/std/min/max** → Already captures spending patterns
- **Transaction count** → Already captures activity level
- **Temporal aggregates** → Already captures time patterns
- **Merchant diversity** → Already captures behavior variety

**Result**: Features contain ALL the fraud signal. A simple logistic regression achieves 95.7% F1!

### Graph Structure Problems

**IBM User-Level Graph (1M transactions → 75 users):**
```
Nodes: 75
Edges: 100,000
Connected nodes: 18 (24%)
Isolated nodes: 57 (76%)
Max degree: 20,504 (super-hub dominates)
```

**Why this breaks GNNs:**
1. **Features too good** (95.7% F1 baseline) → No room for graph/temporal to help
2. **76% isolated nodes** → Message passing doesn't work
3. **Super-hub nodes** → Gradient flow dominated by few nodes
4. **Tiny graph** (75 nodes) → Can't learn generalizable patterns

### Why v1 Wins (and v3-v5 Fail)

**v0 (MLP)**: Uses perfect features → 96% AUC
**v1 (Graph)**: Perfect features + graph structure → 100% AUC ✅
**v2 (Temporal)**: Adds temporal encoding → Degrades (redundant with features)
**v3 (Memory)**: 130K params on 75 nodes → Catastrophic overfitting (33% F1!)
**v4 (Multi-Path)**: Splits 100K edges by time → Random patterns
**v5 (Full)**: 209K params, all components → Some recovery but still worse

## The Solution: Transaction-Level Graph

Instead of aggregating, use **individual transactions as nodes**:

```
Raw Transactions → Transaction Graph → Train Temporal GNN
   (50K txns)      (1.1M temporal      (Learn sequential
                     edges)              fraud patterns!)
```

### Transaction-Level Graph (50K transactions)

```
Nodes: 50,000 transactions
Edges: 1,133,075 temporal connections
Features: 7 raw features (amount, time, MCC, chip, etc.)
Fraud rate: 0.13% (realistic imbalance)
Splits: Chronological (train on older, test on newer)
```

**Why this should work:**
1. **Raw features** → Room for temporal models to learn patterns
2. **Large graph** (50K nodes) → Can learn generalizable patterns
3. **Temporal edges** → Connect transactions chronologically (same user)
4. **Sequential data** → What temporal GNNs were DESIGNED for!

## Expected Results

**Hypothesis:**
- v1 (Graph): Should work well on spatial patterns
- v3 (Memory): Should learn fraud sequences (e.g., "small test → large fraud")
- v4 (Multi-Path): Should distinguish recent vs historical behavior
- v5 (Full): Should combine all signals for best performance

**If v3-v5 > v1 on transaction-level data:**
✅ Architecture is validated!
✅ Proof that model complexity must match data structure
✅ Excellent research contribution!

**If v1 still wins:**
→ May need different graph construction (user-merchant bipartite?)
→ May need more sophisticated temporal features
→ Class imbalance (0.13%) may require special handling

## Key Insight

> "Feature engineering can make your data TOO good for complex models to help.
> When features already contain the answer, adding graph/temporal components
> is like adding complexity to a solved problem."

The real novelty isn't just the architecture - it's understanding **when** and **how** to apply it!
