def passage_text(item: dict, concat_title: bool = True) -> str:
    title = (item.get("title") or "").strip()
    text = (item.get("text") or "").strip()
    if concat_title and title:
        return f"{title} {text}".strip()
    return text
