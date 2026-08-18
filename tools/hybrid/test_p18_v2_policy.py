#!/usr/bin/env python3
from types import SimpleNamespace
try:
    from leviathan_hybrid_uci import allocate_integer_budget
    from leviathan_hybrid_uci_v2 import HybridProxyV2
except ImportError:
    from .leviathan_hybrid_uci import allocate_integer_budget
    from .leviathan_hybrid_uci_v2 import HybridProxyV2


def policy():
    p=HybridProxyV2.__new__(HybridProxyV2);p.risk_weight=1.0;p.regret_weight=1.0
    safe=SimpleNamespace(reply_probability=.55,risk=.05,expected_regret_cp=4.0)
    danger=SimpleNamespace(reply_probability=.30,risk=.80,expected_regret_cp=80.0)
    assert p.candidate_value(danger)>p.candidate_value(safe), (p.candidate_value(danger),p.candidate_value(safe))
    vals=[p.candidate_value(x) for x in (safe,danger)]
    alloc=allocate_integer_budget(8,vals,1)
    assert sum(alloc)==8 and min(alloc)>=1
    print({'safe':p.candidate_value(safe),'danger':p.candidate_value(danger),'threads':alloc})


def budgets():
    for total in range(1,17):
        for n in range(1,9):
            a=allocate_integer_budget(total,[1+i for i in range(n)],1)
            assert sum(a)<=total
            assert all(x>=0 for x in a)

if __name__=='__main__':policy();budgets()
