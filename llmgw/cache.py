# Models provider prefix caching the way the preprint this gateway builds on measured it:
# a cache hit requires the new request's prompt to share an identical prefix with the
# previous request sent to the same (team, backend) pair. It is purely a longest-common-
# prefix check, not a content-addressable cache over arbitrary substrings, which is why
# layout (what comes first in the prompt) is the entire lever this module's callers have.
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Layout:
    """prefix_tokens is whatever comes first in the prompt; suffix_tokens is the rest.
    cache_aware() puts the team's fixed static instruction block first so it can be the
    cached prefix across requests; naive() puts the per-request variable context first,
    which never repeats, so the static block (now in the suffix) never gets to be a cache
    hit even though its content is byte-identical every time."""

    prefix_signature: str
    prefix_tokens: int
    suffix_tokens: int


def cache_aware_layout(static_block_id: str, static_tokens: int, variable_tokens: int) -> Layout:
    return Layout(prefix_signature=f"static:{static_block_id}", prefix_tokens=static_tokens, suffix_tokens=variable_tokens)


def naive_layout(variable_signature: str, static_tokens: int, variable_tokens: int) -> Layout:
    return Layout(prefix_signature=f"variable:{variable_signature}", prefix_tokens=variable_tokens, suffix_tokens=static_tokens)


@dataclass
class PrefixCache:
    """One entry per (team, backend): the prefix signature and token count of the last
    request sent. A hit requires an exact signature match, matching a real provider's
    prefix cache (it would also require the actual bytes to match; this simulation's
    signature stands in for that byte-identity check)."""

    _last: dict[tuple[str, str], tuple[str, int]] = field(default_factory=dict)

    def consume(self, team: str, backend: str, layout: Layout) -> int:
        """Returns the number of prefix tokens that were a cache hit (0 on the first
        request for this (team, backend), or whenever the prefix signature changed),
        then records this request's prefix as the new one to compare the next call to."""
        key = (team, backend)
        previous = self._last.get(key)
        cached = layout.prefix_tokens if previous is not None and previous[0] == layout.prefix_signature else 0
        self._last[key] = (layout.prefix_signature, layout.prefix_tokens)
        return cached
