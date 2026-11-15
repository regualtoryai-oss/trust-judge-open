"""Current raw-only complete-parent delivery policy.

V1 remains reproducible from each immutable source archive; the compatible
receipt API now uses rank-first literal multi-view policy V2.
"""
from app.domain.multiview_parent_delivery import POLICY, pack_original_parents
