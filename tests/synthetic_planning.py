"""Explicit planning-counter injection for local HTTP fixture integrations.

The default matches the synthetic request-count estimate used by those HTTP
tests. A caller may inject a real public test encoder for wire/forecast
consistency. Neither case establishes provider billing or workload capacity.
Production functions and their fail-closed tokenizer requirements are unchanged.
"""
from app.agents import episode_budget


def patch_planning(monkeypatch, *, counter=None):
    source, tail = episode_budget.plan_source_episode, episode_budget.plan_tail
    effective_counter = counter if counter is not None else lambda texts: [100] * len(texts)

    def source_with_counter(*args, **kwargs):
        kwargs.setdefault("counter", effective_counter)
        return source(*args, **kwargs)

    def tail_with_counter(*args, **kwargs):
        kwargs.setdefault("counter", effective_counter)
        return tail(*args, **kwargs)

    monkeypatch.setattr(episode_budget, "plan_source_episode", source_with_counter)
    monkeypatch.setattr(episode_budget, "plan_tail", tail_with_counter)
