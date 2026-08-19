"""Terminal presentation layer."""

from .components import (
    L,
    answer_footer,
    answer_view,
    banner,
    chronicle_table,
    council_footer,
    council_view,
    daily_card,
    decision_view,
    deepest_section,
    echo_line,
    recap_view,
    library_table,
    mood_cards,
    mood_view,
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
    "chronicle_table", "council_footer", "council_view", "decision_view",
    "echo_line", "mood_cards", "mood_view", "recap_view",
    "daily_card", "deepest_section", "doctor_view", "error_panel", "gradient_text",
    "help_view", "prose",
    "ingest_summary", "library_table", "make_console", "make_ingest_progress",
    "retrieval_table", "rights_summary", "rule", "score_bar", "source_cards", "sources_table",
    "status_bar", "tag_cloud",
]
