"""Limit-order-book reconstruction from Databento MBO (market-by-order) messages, compiled with numba.

Conventions (Databento MBO): action A=add, C=cancel (size = quantity removed), M=modify (new price/size; a price change
loses queue priority), R=clear book, T=trade and F=fill do not change the book (CME sends the corresponding order
reductions as separate C/M messages). Books are only consistent at the end of an event packet, flagged F_LAST (128),
so we record snapshots only there.
"""
import numpy as np
import numba as nb
from numba import types
from numba.typed import Dict

F_LAST = 128


@nb.njit(cache=True)
def _rebuild(action, side, px, size, oid, flags, base, nlev, depth):
    n = len(action)
    NP = 20000                                   # price grid (ticks) around `base`
    bid = np.zeros(NP, np.int64)
    ask = np.zeros(NP, np.int64)
    o_side = Dict.empty(key_type=types.uint64, value_type=types.int64)
    o_px = Dict.empty(key_type=types.uint64, value_type=types.int64)
    o_sz = Dict.empty(key_type=types.uint64, value_type=types.int64)
    best_b, best_a = -1, NP
    out_idx = np.empty(n, np.int64)
    out_bp = np.empty(n, np.int64); out_ap = np.empty(n, np.int64)
    out_bs = np.empty((n, depth), np.int64); out_as = np.empty((n, depth), np.int64)
    k = 0
    for i in range(n):
        a = action[i]
        if a == 82:                              # 'R' clear
            bid[:] = 0; ask[:] = 0
            o_side.clear(); o_px.clear(); o_sz.clear()
            best_b, best_a = -1, NP
        elif a == 65:                            # 'A' add
            p = px[i] - base
            if 0 <= p < NP:
                o_side[oid[i]] = side[i]; o_px[oid[i]] = p; o_sz[oid[i]] = size[i]
                if side[i] == 66:
                    bid[p] += size[i]
                    if p > best_b: best_b = p
                else:
                    ask[p] += size[i]
                    if p < best_a: best_a = p
        elif a == 67 or a == 77:                 # 'C' cancel / 'M' modify
            o = oid[i]
            if o in o_px:
                p0 = o_px[o]; s0 = o_sz[o]; sd = o_side[o]
                # remove the old quantity
                rem = size[i] if a == 67 else s0
                if rem > s0: rem = s0
                if sd == 66: bid[p0] -= rem
                else: ask[p0] -= rem
                if a == 67:
                    if s0 - rem <= 0:
                        del o_side[o]; del o_px[o]; del o_sz[o]
                    else:
                        o_sz[o] = s0 - rem
                else:                            # modify: re-insert at (possibly) new price/size
                    p = px[i] - base
                    if 0 <= p < NP and size[i] > 0:
                        o_px[o] = p; o_sz[o] = size[i]
                        if sd == 66:
                            bid[p] += size[i]
                            if p > best_b: best_b = p
                        else:
                            ask[p] += size[i]
                            if p < best_a: best_a = p
                    else:
                        del o_side[o]; del o_px[o]; del o_sz[o]
                # repair best prices if the best level emptied
                while best_b >= 0 and bid[best_b] <= 0: best_b -= 1
                while best_a < NP and ask[best_a] <= 0: best_a += 1
            elif a == 77:                        # modify of an unknown order = add
                p = px[i] - base
                if 0 <= p < NP:
                    o_side[o] = side[i]; o_px[o] = p; o_sz[o] = size[i]
                    if side[i] == 66:
                        bid[p] += size[i]
                        if p > best_b: best_b = p
                    else:
                        ask[p] += size[i]
                        if p < best_a: best_a = p
        if (flags[i] & 128) and best_b >= 0 and best_a < NP:
            out_idx[k] = i; out_bp[k] = best_b; out_ap[k] = best_a
            for j in range(depth):
                out_bs[k, j] = bid[best_b - j] if best_b - j >= 0 else 0
                out_as[k, j] = ask[best_a + j] if best_a + j < NP else 0
            k += 1
    return out_idx[:k], out_bp[:k], out_ap[:k], out_bs[:k], out_as[:k]


def rebuild_book(ev, tick_fixed, depth=5):
    """ev: DataFrame of one instrument's MBO messages (fixed-point prices). Returns top-of-book snapshots
    (one per completed event packet) with prices in ticks and sizes for `depth` levels per side."""
    import pandas as pd
    act = ev["action"].str.encode("ascii").str[0].astype(np.uint8).values if ev["action"].dtype == object or str(ev["action"].dtype).startswith("str") \
        else ev["action"].values
    sd = ev["side"].map(lambda s: ord(s)).astype(np.int64).values
    price = ev["price"].values.astype(np.int64)
    valid = price != np.iinfo(np.int64).max
    ticks = np.where(valid, price // tick_fixed, 0)
    base = int(np.median(ticks[valid])) - 10000
    idx, bp, ap, bs, as_ = _rebuild(act.astype(np.int64), sd, ticks, ev["size"].values.astype(np.int64),
                                    ev["order_id"].values.astype(np.uint64), ev["flags"].values.astype(np.int64),
                                    base, 20000, depth)
    out = pd.DataFrame({"ts": ev["ts_recv"].values[idx], "bid": (bp + base), "ask": (ap + base)})
    for j in range(depth):
        out[f"bsz{j+1}"] = bs[:, j]; out[f"asz{j+1}"] = as_[:, j]
    return out
