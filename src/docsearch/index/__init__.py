from collections.abc import Iterable

from docsearch.config import PUBLIC
from docsearch.models import Chunk


def is_visible(chunk: Chunk, user_visibility: Iterable[str]) -> bool:
    """D10: public chunks are visible to everyone, others need an overlapping tag."""
    return PUBLIC in chunk.visibility or bool(set(user_visibility) & set(chunk.visibility))
