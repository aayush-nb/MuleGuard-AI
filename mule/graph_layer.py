"""Pass 1 (graph): read the relationships between accounts, as of the past only.

Two rules keep this layer free of leakage:

  * Counters answer "what happened strictly before this step". Rows in the same
    hour cannot see each other, because they have no reliable order.
  * PageRank and components come from a daily batch. A row on day d reads the
    snapshot built from days before d, which is how a nightly graph job would
    feed a feature store.

Labels are never read here.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from . import config as cfg
from .data import TYPE_CODE

STEP_STRIDE = 1 << 24  # Larger than any step, so account * stride + step sorts by both.


class PastSum:
    """Prefix sums over one account's events, queried strictly before a step."""

    def __init__(self, account, step, value=None):
        key = account.astype(np.int64) * STEP_STRIDE + step
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        values = np.ones(len(key)) if value is None else value[order]
        self.prefix = np.concatenate([[0.0], np.cumsum(values)])

    def before(self, account, step, window=None):
        base = account.astype(np.int64) * STEP_STRIDE
        start = base if window is None else base + np.maximum(step - window, 0)
        lo = np.searchsorted(self.key, start, side="left")
        hi = np.searchsorted(self.key, base + step, side="left")  # Excludes this step.
        return self.prefix[hi] - self.prefix[lo]


def money_edges(df):
    """Return (source, target) in the direction money moves.

    CASH_IN credits nameOrig, so its edge runs against the log's orig -> dest order.
    """
    orig, dest = df["orig"].to_numpy(), df["dest"].to_numpy()
    cash_in = df["type"].to_numpy() == TYPE_CODE["CASH_IN"]
    return np.where(cash_in, dest, orig), np.where(cash_in, orig, dest)


def first_contacts(account, counterparty, step):
    """Keep the earliest row of each (account, counterparty) pair."""
    pair = account.astype(np.int64) * (int(counterparty.max()) + 1) + counterparty
    order = np.argsort(step, kind="stable")
    _, first = np.unique(pair[order], return_index=True)
    keep = order[first]
    return account[keep], step[keep]


def pagerank_and_components(source, target):
    """Return (nodes, pagerank, component_size) for a snapshot of the graph.

    PageRank is scaled to average 1, so scores compare across days as the graph grows.
    """
    nodes, flat = np.unique(np.concatenate([source, target]), return_inverse=True)
    n = len(nodes)
    src, dst = flat[: len(source)], flat[len(source):]
    out_degree = np.bincount(src, minlength=n).astype(np.float64)
    transition = csr_matrix((1.0 / out_degree[src], (dst, src)), shape=(n, n))
    dangling = out_degree == 0
    rank = np.full(n, 1.0 / n)
    d = cfg.PAGERANK_DAMPING
    for _ in range(cfg.PAGERANK_ITERATIONS):
        rank = d * (transition @ rank + rank[dangling].sum() / n) + (1 - d) / n
    _, labels = connected_components(transition, directed=False)
    return nodes, rank * n, np.bincount(labels)[labels]


def daily_snapshot_features(step, source, target, query_accounts, query_steps, n_accounts):
    """Read PageRank and component size from the latest finished day's snapshot."""
    query_day = (query_steps - 1) // cfg.HOURS_PER_DAY
    results = {name: [np.zeros(len(a), np.float32) for a in query_accounts]
               for name in ("pagerank", "component_size")}

    for day in np.unique(query_day):
        if day == 0:
            continue  # No day has finished yet, so these stay at zero.
        history = step <= day * cfg.HOURS_PER_DAY
        nodes, rank, size = pagerank_and_components(source[history], target[history])
        rank_lookup = np.zeros(n_accounts, np.float32)
        size_lookup = np.zeros(n_accounts, np.float32)
        rank_lookup[nodes] = rank
        size_lookup[nodes] = size
        rows = query_day == day
        for i, accounts in enumerate(query_accounts):
            results["pagerank"][i][rows] = rank_lookup[accounts[rows]]
            results["component_size"][i][rows] = size_lookup[accounts[rows]]
    return results


def graph_features(df, scored, n_accounts):
    """Build point-in-time graph features for the rows in `scored`.

    `df` is the full log, every type, so an account's history is complete.
    `scored` holds the rows being scored, with `account` and `counterparty` columns.
    """
    source, target = money_edges(df)
    step = df["step"].to_numpy().astype(np.int64)
    amount = df["amount"].to_numpy()
    q_step = scored["step"].to_numpy().astype(np.int64)
    parties = {"acct": scored["account"].to_numpy(), "cpty": scored["counterparty"].to_numpy()}

    first_seen = np.full(n_accounts, np.iinfo(np.int64).max)
    np.minimum.at(first_seen, source, step)
    np.minimum.at(first_seen, target, step)

    out = {}
    inbound = PastSum(target, step)
    for side, accounts in parties.items():
        out[f"{side}_in_degree"] = inbound.before(accounts, q_step)
        out[f"{side}_in_degree_24h"] = inbound.before(accounts, q_step, cfg.VELOCITY_WINDOW_HOURS)
    del inbound

    outbound = PastSum(source, step)
    for side, accounts in parties.items():
        out[f"{side}_out_degree"] = outbound.before(accounts, q_step)
    del outbound

    in_flow, out_flow = PastSum(target, step, amount), PastSum(source, step, amount)
    flows = {side: (in_flow.before(a, q_step), out_flow.before(a, q_step))
             for side, a in parties.items()}
    del in_flow, out_flow

    contacts = PastSum(*first_contacts(
        np.concatenate([source, target]),
        np.concatenate([target, source]),
        np.concatenate([step, step]),
    ))
    for side, accounts in parties.items():
        out[f"{side}_unique_counterparties"] = contacts.before(accounts, q_step)
    del contacts

    for side, accounts in parties.items():
        received, sent = flows[side]
        out[f"{side}_log_in_flow"] = np.log1p(received)
        out[f"{side}_log_out_flow"] = np.log1p(sent)
        # NaN (no inflow yet) is different from 0 (received money, kept all of it).
        out[f"{side}_pass_through_ratio"] = np.where(
            received > 0, sent / np.where(received > 0, received, 1), np.nan)
        seen = first_seen[accounts] < q_step
        out[f"{side}_age_hours"] = np.where(seen, q_step - first_seen[accounts], -1)

    received, degree = flows["acct"][0], out["acct_in_degree"]
    has_inflow = received > 0
    out["amount_vs_acct_avg_inflow"] = np.where(
        has_inflow,
        scored["amount"].to_numpy() * degree / np.where(has_inflow, received, 1),
        np.nan,
    )

    snapshots = daily_snapshot_features(
        step, source, target, list(parties.values()), q_step, n_accounts)
    for name, per_side in snapshots.items():
        for side, values in zip(parties, per_side):
            out[f"{side}_{name}"] = values

    return pd.DataFrame(out, index=scored.index).astype(np.float32)
