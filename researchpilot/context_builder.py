def build_context(results: list[dict]) -> tuple[str, list[dict]]:
    sources = [{"id": i, "document": d["source"], "page": d["page"],
                "text": d["text"]} for i, d in enumerate(results, 1)]
    context = "\n\n---\n\n".join(
        f"[{s['id']}] Document: {s['document']} Page: {s['page']}\nContent: {s['text']}"
        for s in sources)
    return context, sources
