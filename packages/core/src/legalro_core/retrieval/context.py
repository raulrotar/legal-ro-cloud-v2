"""Context assembly: expand chunks into full context for LLM.

Parent-document retrieval:
  The top `settings.search.parent_doc_top_n` chunks (by RRF rank) that have
  `act_full_text` use the full act text instead of just the matched chunk.
  This recovers context that would otherwise be split across multiple chunks.

  Deduplication: when multiple chunks from the same act all qualify for
  parent expansion, the act text is included only once.
"""
from legalro_core.config import Settings


def assemble_context(chunks: list[dict], settings: Settings) -> str:
    top_n = getattr(settings.search, "parent_doc_top_n", 3)
    max_chars = getattr(settings.search, "max_parent_chars", 8000)

    context_parts = []
    seen_act_keys: set[str] = set()

    for rank, chunk in enumerate(chunks):
        doc_type = chunk.get("document_type", "")
        title = chunk.get("title", "")
        source_issue = chunk.get("source_issue_id", "")
        full_path = chunk.get("full_path", "")
        authority = chunk.get("issuing_authority", "")
        act_number = chunk.get("act_number", "")
        act_year = chunk.get("act_year", "")
        locality = chunk.get("locality", "")
        law_id = chunk.get("law_id", "")

        header_parts = []
        if doc_type:
            header_parts.append(doc_type.replace("_", " "))
        if act_number and str(act_number) != "0":
            header_parts.append(f"Nr. {act_number}")
        if act_year:
            header_parts.append(f"/{act_year}")
        if authority:
            header_parts.append(authority)
        if law_id:
            header_parts.append(f"[{law_id}]")
        if source_issue:
            header_parts.append(f"MO: {source_issue}")
        if locality:
            header_parts.append(f"Jud. {locality}")

        # Parent-doc path: top-N chunks by RRF rank that have act_full_text
        act_full_text = chunk.get("act_full_text", "")
        use_parent = bool(act_full_text) and rank < top_n
        chunk_text = chunk.get("text", "")

        if use_parent:
            # Deduplicate: same act may appear via multiple chunks
            act_key = f"{source_issue}:{chunk.get('act_index_in_issue', law_id)}"
            if act_key in seen_act_keys:
                # Act already shown — still include chunk if it's not in the
                # already-assembled parent window (e.g. annex article beyond truncation)
                parent_window = act_full_text[:max_chars]
                if chunk_text and chunk_text not in parent_window:
                    body = chunk_text
                    if full_path:
                        header_parts.append(f"§{full_path}")
                    context_parts.append(f"[{' | '.join(header_parts)}]\n{body}")
                continue
            seen_act_keys.add(act_key)
            parent_window = act_full_text[:max_chars]
            if chunk_text and chunk_text not in parent_window:
                # Chunk is beyond the parent window — include the full window
                # (which contains the legal basis and early articles) plus the
                # matched chunk so the LLM sees both the act structure and the
                # specific section that matched.
                body = parent_window + "\n...\n" + chunk_text
                header_parts.append("§[full act + chunk]")
            else:
                body = parent_window
                header_parts.append("§[full act]")
        else:
            body = chunk_text
            if full_path:
                header_parts.append(f"§{full_path}")

        header = " | ".join(header_parts)
        if title:
            header = f"{header}\nTitlu: {title}"

        context_parts.append(f"[{header}]\n{body}")

    return "\n---\n".join(context_parts)
