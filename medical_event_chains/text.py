import re
from .schema import Span


def split_spans(text, clauses=False):
    pattern = r"[。！？!?\n]+|\.(?=\s|$)"
    if clauses:
        pattern += r"|[，,；;]+"
    spans, start = [], 0
    for match in re.finditer(pattern, text):
        end = match.end()
        if text[start:end].strip():
            spans.append(Span(text[start:end], start, end))
        start = end
    if text[start:].strip():
        spans.append(Span(text[start:], start, len(text)))
    return spans


def assertion(text, span):
    clause = next((s for s in split_spans(text, True) if s.start <= span.start < s.end), Span(text, 0, len(text)))
    prefix = text[clause.start:span.start]
    uncertain = bool(re.search(r"不排除|考虑|可能|疑似|待排|不能排除", prefix))
    cleaned = re.sub(r"不排除|不能排除|无(?:明显)?(?:诱因|原因)", "", prefix)
    negated = bool(re.search(r"未见|否认|不伴|无(?:明显)?|未(?:予|行|用|服|发现)?|没有", cleaned))
    return negated, uncertain


def time_blocks(text, times):
    boundaries = sorted({0, len(text), *(t.span.start for t in times)})
    return [{"start": a, "end": b, "text": text[a:b],
             "time_id": next((t.id for t in reversed(times) if t.span.start <= a), None)}
            for a, b in zip(boundaries, boundaries[1:]) if a < b]


def associate_time(text, event, times):
    sentence = next((s for s in split_spans(text) if s.start <= event.object.start < s.end), None)
    if sentence:
        local = [t for t in times if sentence.start <= t.span.start < sentence.end]
        prior = [t for t in local if t.span.start <= event.object.start]
        if prior:
            return prior[-1]
        # Handles postposed times such as “皮疹4天前出现”.
        if local:
            return local[0]
    return next((t for t in reversed(times) if t.span.start <= event.object.start), None)
