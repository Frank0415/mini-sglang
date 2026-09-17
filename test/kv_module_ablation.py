"""Test-side module-boundary ablation; does not patch production classes.

Run: .venv/bin/python test/kv_module_ablation.py
The harness represents a proposed caller, NOT an existing engine implementation.
"""
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mem_cache.allocator import KVPoolConfig, MHATokenToKVPool, PagedTokenToKVPoolAllocator
from src.mem_cache.memory_pool import ReqToTokenPool, ReqToTokenPoolConfig


class Rejected(ValueError):
    pass


@dataclass(frozen=True)
class Guards:
    config: bool
    length: bool
    finish: bool


class ModuleHarness:
    """No arbitrary internal indices accepted: caller supplies request id/count."""
    def __init__(self, guards, page_size=4, pages=5, capacity=20, context=12):
        self.guards = guards
        if guards.config:
            if page_size <= 0 or pages < 0 or capacity < 0:
                raise Rejected('invalid startup configuration')
            if capacity < page_size * pages:
                raise Rejected('allocator can address slots outside KV storage')
        self.rows = ReqToTokenPool(ReqToTokenPoolConfig(4, context, torch.device('cpu')))
        self.alloc = PagedTokenToKVPoolAllocator(page_size, pages)
        self.kv = MHATokenToKVPool(KVPoolConfig(2, capacity, 3, 4, torch.float32, torch.device('cpu')))
        self.kv.k_buffer.zero_()
        self.kv.v_buffer.zero_()
        self.context = context
        self.requests = {}
        self.closed = {}
        self.initial_pages = set(self.alloc.free_pages)

    def append(self, rid, count):
        old_record = self.requests.get(rid)
        old = [] if old_record is None else old_record['slots']
        if self.guards.length and (count < 0 or len(old) + count > self.context):
            raise Rejected('request length outside supported range')
        fresh = old_record is None
        if fresh:
            row_ids = self.rows.alloc_rows(1)
            if row_ids is None:
                return False
            row = row_ids[0]
        else:
            row = old_record['row']
        result = self.alloc.alloc_extend(len(old), len(old) + count,
                                       old[-1] if old else -1, count)
        if result is None:
            if fresh:
                self.rows.free_rows([row])
            return False
        self.rows.write((row, slice(len(old), len(old) + count)), torch.tensor(result, dtype=torch.long))
        for layer in range(2):
            values = self.values(rid, layer, len(old), count)
            self.kv.set_kv_buffer(layer, torch.tensor(result, dtype=torch.long), values, -values)
        self.requests[rid] = {'row': row, 'slots': old + result}
        self.closed.pop(rid, None)
        return True

    @staticmethod
    def values(rid, layer, start, count):
        # Exact small FP32 integers encode request, position, and layer.
        return (torch.arange(start, start + count, dtype=torch.float32) + rid * 100 + layer * 1000).view(count, 1, 1).expand(count, 3, 4)

    def finish(self, rid):
        if rid in self.closed:
            if self.guards.finish:
                return
            record = self.closed[rid]  # Inject a repeated completion event.
        else:
            record = self.requests.pop(rid)
            self.closed[rid] = record
        self.alloc.free(record['slots'])
        self.rows.free_rows([record['row']])

    def check(self):
        p = self.alloc.page_size
        occupied = []
        row_ids = []
        for rid, record in self.requests.items():
            slots, row = record['slots'], record['row']
            assert len(slots) <= self.context, 'request exceeded mapping width'
            assert self.rows.req_to_token[row, :len(slots)].tolist() == slots, 'mapping lost or changed a token'
            occupied.extend({slot // p for slot in slots})
            row_ids.append(row)
            indices = self.rows.req_to_token[row, :len(slots)]
            for layer in range(2):
                expected = self.values(rid, layer, 0, len(slots))
                assert torch.equal(self.kv.get_key_buffer(layer, indices), expected), 'K corrupted'
                assert torch.equal(self.kv.get_value_buffer(layer, indices), -expected), 'V corrupted'
        assert len(occupied) == len(set(occupied)), 'requests share pages'
        assert len(self.alloc.free_pages) == len(set(self.alloc.free_pages)), 'duplicate free pages'
        assert set(occupied).isdisjoint(self.alloc.free_pages), 'live page marked free'
        assert set(occupied) | set(self.alloc.free_pages) == self.initial_pages, 'page accounting mismatch'
        assert len(self.rows.free_slots) == len(set(self.rows.free_slots)), 'duplicate free request rows'
        assert set(row_ids).isdisjoint(self.rows.free_slots), 'live row marked free'
        assert set(row_ids) | set(self.rows.free_slots) == {1, 2, 3, 4}, 'row accounting mismatch'


def scenario(name, guards):
    try:
        if name == 'zero_page_size':
            m = ModuleHarness(guards, page_size=0)
            m.append(1, 1)
        elif name == 'negative_page_size':
            ModuleHarness(guards, page_size=-1)
            return 'accepted_invalid_config'
        elif name == 'negative_page_count':
            ModuleHarness(guards, pages=-1)
            return 'accepted_invalid_config'
        elif name == 'capacity_mismatch':
            m = ModuleHarness(guards, pages=2, capacity=7)
            m.append(1, 8)  # allocator legitimately produces slot 7; storage has 0..6
            m.check()
        elif name == 'context_overflow':
            m = ModuleHarness(guards, context=4)
            m.append(1, 4)
            m.append(1, 1)  # all counts and indices are generated internally
            m.check()
        elif name == 'duplicate_finish':
            m = ModuleHarness(guards)
            m.append(1, 3)
            m.finish(1)
            m.finish(1)
            m.check()
        return 'safe'
    except Rejected:
        return 'rejected_at_boundary'
    except Exception as exc:
        return f'{type(exc).__name__}: {exc}'


def normal_flow(guards, seed):
    rng = random.Random(seed)
    p = rng.choice([1, 2, 4, 7, 16])
    m = ModuleHarness(guards, page_size=p, pages=8, capacity=p * 8, context=p * 4)
    stats = {'steps': 0, 'capacity_denials': 0, 'allocations': 0, 'finishes': 0}
    for _ in range(100):
        rid = rng.randrange(6)
        if rid in m.requests and rng.random() < .3:
            m.finish(rid)
            stats['finishes'] += 1
        else:
            length = len(m.requests.get(rid, {'slots': []})['slots'])
            count = rng.randrange(min(2 * p, m.context - length) + 1)
            result = m.append(rid, count)
            stats['allocations' if result else 'capacity_denials'] += 1
        m.check()
        stats['steps'] += 1
    for rid in list(m.requests):
        m.finish(rid)
    m.check()
    return stats


def main():
    sources = [ROOT / 'src/mem_cache/allocator.py', ROOT / 'src/mem_cache/memory_pool.py']
    def hashes():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    before = hashes()
    result = {'caller_is_test_harness': True, 'ablations': [], 'source_hashes': before}
    cases = ['zero_page_size', 'negative_page_size', 'negative_page_count', 'capacity_mismatch', 'context_overflow', 'duplicate_finish']
    for flags in itertools.product([False, True], repeat=3):
        g = Guards(*flags)
        row = {'guards': vars(g), 'fault_scenarios': {case: scenario(case, g) for case in cases}}
        total = {'steps': 0, 'capacity_denials': 0, 'allocations': 0, 'finishes': 0}
        for seed in range(10):
            stats = normal_flow(g, seed)
            for key in total:
                total[key] += stats[key]
        row['normal_flow'] = total
        result['ablations'].append(row)
        print(flags, total, row['fault_scenarios'])
    result['source_unchanged'] = before == hashes()
    assert result['source_unchanged'], 'Source changed during experiment'
    (ROOT / 'test/kv_module_ablation_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
