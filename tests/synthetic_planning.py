"""Explicit planning-counter injection for local HTTP fixture integrations.

Matches the synthetic request-count estimate used by those HTTP tests. This
does not verify a real tokenizer, provider bill, or real workload capacity.
Production functions and their fail-closed tokenizer requirements are unchanged.
"""
from app.agents import episode_budget


def patch_planning(monkeypatch):
    source, tail = episode_budget.plan_source_episode, episode_budget.plan_tail

    def source_with_counter(*args, **kwargs):
        kwargs.setdefault("counter", lambda texts: [100] * len(texts))
        return source(*args, **kwargs)

    def tail_with_counter(*args, **kwargs):
        kwargs.setdefault("counter", lambda texts: [100] * len(texts))
        return tail(*args, **kwargs)

    monkeypatch.setattr(episode_budget, "plan_source_episode", source_with_counter)
    monkeypatch.setattr(episode_budget, "plan_tail", tail_with_counter)
