"""Terminal presentation layer."""

from .components import (
    L,
    answer_footer,
    answer_view,
    banner,
    daily_card,
    deepest_section,
    library_table,
    prose,
    retrieval_table,
    rights_summary,
    rule,
    source_cards,
    sources_table,
    status_bar,
    tag_cloud,
)
from .theme import gradient_text, make_console, score_bar
from .views import (
    Spinner,
    StreamView,
    doctor_view,
    error_panel,
    help_view,
    ingest_summary,
    make_ingest_progress,
)

__all__ = [
    "L", "Spinner", "StreamView", "answer_footer", "answer_view", "banner",
    "daily_card", "deepest_section", "doctor_view", "error_panel", "gradient_text",
    "help_view", "prose",
    "ingest_summary", "library_table", "make_console", "make_ingest_progress",
    "retrieval_table", "rights_summary", "rule", "score_bar", "source_cards", "sources_table",
    "status_bar", "tag_cloud",
]
