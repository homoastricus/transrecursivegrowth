import math
from typing import List

def _nested(step: int, depth: int, value: int) -> int:
    if step <= 0:
        return value
    cur = value
    for i in range(C(step)):
        temp_cur = cur
        for j in range(C(step)):
            number = C(temp_cur)
            temp_cur = _nested(step - 1, depth + 1, C(number))
        cur = _nested(step - 1, depth + 1, temp_cur)
    return cur

def re(n):
    if n <= 1:
        return n
    if n == 2:
        return C(_nested(C(n), C(n), C(n)))
    return C(_nested(re(n - 1), re(n - 1), re(n - 1)))

def hyper_scaling(n, iteration=0):
    if iteration > n and n > 0:
        return n
    def scale_map(x, level):
        if level == 0:
            return x
        def tower(h, b=math.e):
            return 1 if h <= 0 else b ** tower(h - 1, b)
        scale_height = max(1, int(abs(x)))
        return tower(scale_height)
    cur = n
    steps = n ** n
    for i in range(steps):
        cur_scale = scale_map(cur, i)
        next_val = cur ** cur_scale
        cur = hyper_scaling(next_val, iteration + 1)
    return cur

def scale(n, depth=0):
    if depth > n:
        return n
    scaled_n = hyper_scaling(n)
    raw_length = conway_chain([scaled_n] * n)
    chain_length = scale(raw_length, depth + 1) or 1
    chain = []
    for i in range(chain_length):
        element = scale(scaled_n + i, depth + 1)
        chain.append(element)
    return conway_chain(chain)

def hyper_conway(n, depth=0):
    if depth > n ** n: return n
    labels = boom(n - 1, mode="boom") if n > 1 else n
    chain = [meta_conway([n] * n, depth + 1, max_depth=n ** n, labels=labels)] * n
    strong_chain = conway_chain(chain)
    for s in range(strong_chain):
        chain = [meta_conway([strong_chain]*n, depth + 1, max_depth=n ** n, labels=labels)] * n
    return conway_chain(chain)

def meta_conway(chain: List[int], depth: int = 1, max_depth: int = None, labels: int = 3) -> int:
    if max_depth is None:
        max_depth = len(chain) ** len(chain)
    if depth > max_depth:
        return conway_chain(chain)
    if len(chain) == 1:
        return conway_chain( [chain[0]] * chain[0])
    meta_elements = []
    for x in chain:
        sub_arrays = []
        dynamic_labels = meta_conway([x] * x, depth + 1, max_depth, labels) if x > 1 else labels
        for label in range(1, min(dynamic_labels, labels) + 1):
            sub_chain = [(x + label)] * (x + label)
            sub_val = meta_conway(sub_chain, depth + 1, max_depth, labels - 1 if labels > 1 else 1)
            sub_arrays.append(sub_val)
        tree_val = conway_chain(sub_arrays)
        meta_elements.append(tree_val)
    return conway_chain(meta_elements)

def conway_chain(chain: List[int]) -> int:
    if len(chain) == 1:
        return chain[0]
    elif len(chain) == 2:
        a, b = chain[0], chain[1]
        return a ** b
    else:
        a, b = chain[0], chain[1]
        tail = chain[2:]
        if b == 1:
            return conway_chain([a] + tail)
        else:
            inner_chain = [a] + [b - 1] + tail
            inner_result = conway_chain(inner_chain)
            new_chain = [a, inner_result] + tail[:-1] if len(tail) > 1 else [a, inner_result]
            return conway_chain(new_chain)

def C(n: int):
    if n <= 1:
        return scale(hyper_conway(n))
    chain_up = n
    for s in range(C(n - 1)):
        chain_up = scale(hyper_conway(chain_up))
    return chain_up

def boom(n, depth=0, mode="boom", is_main_boom=False):
    if n <= 1: return 1
    if depth > C(re(n)) and mode == "init": return n
    r = n
    if mode == "init":
        for _ in range(C(re(n))):
            r = C(re(boom(r - 1, depth + 1, "init")))
        return r
    elif mode == "iter":
        cur = boom(C(re(n)), 0, "init")
        for s in range(C(re(cur))):
            cur = boom(C(re(s)), 0, "init")
        return cur
    else:  # "boom"
        for _ in range(C(re(n))):
            r = C(re(boom(r - 1, depth + 1, "boom")))
        cur = r
        for step in range(C(re(n))):
            cur = boom(C(re(step)), 0, "boom")
        if is_main_boom and n > 1:
            for _ in range(boom(C(re(n)), mode="iter")):
                cur = boom(C(re(cur)), mode="boom", is_main_boom=False)
        return cur

if __name__ == "__main__":
    print(boom(1, mode="boom", is_main_boom=True))
