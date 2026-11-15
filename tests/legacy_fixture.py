"""Explicit legacy graph fixtures only; no production API compatibility path.

These tests exercise original extraction/error/export plumbing. New strict
initial/refine integration tests use app.main.create_app directly and must not
import this helper. No successful scope stage or NLI outcome is fabricated.
"""
from app.main import create_app


def create_legacy_app(settings, provider, **kwargs):
    if provider is None:
        return create_app(settings=settings, provider=None, **kwargs)
    if getattr(provider, "test_only_legacy_precision", False) is not True:
        raise ValueError("Legacy app fixture requires an explicitly marked test double")
    application = create_app(settings=settings, provider=provider, **kwargs)
    repository = application.state.repo
    original = repository.config
    # Record the exact graph this explicit test fixture executes, so real API
    # progress remains 7 question stages + publication rather than fake NLI.
    repository.config = lambda: {**original(), "strict_scope_enabled": False, "premise_bank_enabled": False}
    return application
