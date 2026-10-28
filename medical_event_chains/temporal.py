"""Conservative Chinese time normalization; unresolved expressions retain offsets."""
import calendar
import re
from datetime import date, timedelta
from .schema import Span, Time

NUM = r"[0-9零〇一二两三四五六七八九十百]+"
PATTERNS = [
    rf"(?:\d{{4}}年\d{{1,2}}月\d{{1,2}}[日号]?|\d{{4}}[-/]\d{{1,2}}[-/]\d{{1,2}})",
    r"\d{4}年\d{1,2}月", r"\d{4}年",
    rf"(?:入院|就诊)?(?:前)?{NUM}(?:个)?(?:年|月|周|天|日)(?:前|后)",
    rf"(?:入院前|就诊前){NUM}(?:个)?(?:年|月|周|天|日)",
    r"\d{1,2}月\d{1,2}[日号]", r"今日|今天|入院当日|就诊当日|昨日|昨天|次日|翌日|术后|出院后",
]
TIME_RE = re.compile("|".join(f"(?:{p})" for p in PATTERNS))


def chinese_number(text):
    if text.isdigit():
        return int(text)
    digits = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
    total = current = 0
    for char in text:
        if char in digits:
            current = digits[char]
        elif char in "十百":
            total += (current or 1) * {"十": 10, "百": 100}[char]
            current = 0
        else:
            raise ValueError("Unsupported numeral")
    return total + current


def shift_months(anchor, months):
    index = anchor.year * 12 + anchor.month - 1 + months
    year, month = divmod(index, 12)
    month += 1
    return date(year, month, min(anchor.day, calendar.monthrange(year, month)[1]))


def normalize(span, admission=None, previous=None, previous_id=None, index=0):
    result = Time(f"t{index}", span)
    value = span.text
    try:
        match = re.fullmatch(r"(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})[日号]?", value)
        if match:
            resolved = date(*map(int, match.groups()))
            result.start_date = result.end_date = resolved.isoformat()
            result.precision, result.status = "day", "resolved"
            return result
        match = re.fullmatch(r"(\d{4})年(?:(\d{1,2})月)?", value)
        if match:
            year, month = int(match[1]), match[2]
            first = date(year, int(month or 1), 1)
            last = date(year, int(month), calendar.monthrange(year, int(month))[1]) if month else date(year, 12, 31)
            result.start_date, result.end_date = first.isoformat(), last.isoformat()
            result.precision, result.status = ("month" if month else "year"), "resolved"
            return result
        base = date.fromisoformat(admission) if admission else None
        match = re.fullmatch(r"(\d{1,2})月(\d{1,2})[日号]", value)
        if match and base:
            # A month/day without year is anchored to admission year; no inferred rollover.
            resolved = date(base.year, int(match[1]), int(match[2]))
            result.anchor = "admission_year"
        elif value in ("今日", "今天", "入院当日", "就诊当日", "昨日", "昨天") and base:
            resolved = base - timedelta(days=value in ("昨日", "昨天"))
            result.anchor = "admission"
        elif value in ("次日", "翌日") and previous:
            resolved = date.fromisoformat(previous) + timedelta(days=1)
            result.anchor = previous_id
        else:
            match = re.fullmatch(rf"(?:入院|就诊)?(前)?({NUM})(?:个)?(年|月|周|天|日)(前|后)?", value)
            if not match or not base or not (match[1] or match[4]):
                return result
            amount = chinese_number(match[2]) * (-1 if match[1] or match[4] == "前" else 1)
            unit = match[3]
            resolved = shift_months(base, amount * (12 if unit == "年" else 1)) if unit in ("年", "月") else base + timedelta(days=amount * (7 if unit == "周" else 1))
            result.anchor = "admission"
        result.start_date = result.end_date = resolved.isoformat()
        result.precision, result.status = "day", "resolved"
    except (ValueError, OverflowError):
        result.status = "invalid"
    return result


def extract_times(text, admission=None, extra_spans=()):
    candidates = [Span(m[0], m.start(), m.end()) for m in TIME_RE.finditer(text)]
    candidates.extend(extra_spans)
    # Prefer the longest expression at a position; discard intersecting duplicates.
    selected = []
    for span in sorted(candidates, key=lambda s: (-(s.end - s.start), s.start)):
        span.validate(text)
        if not any(span.start < old.end and old.start < span.end for old in selected):
            selected.append(span)
    result, previous, previous_id = [], None, None
    for index, span in enumerate(sorted(selected, key=lambda s: s.start)):
        item = normalize(span, admission, previous, previous_id, index)
        result.append(item)
        # An unresolved or coarse expression breaks the exact-day reference chain.
        previous = item.start_date if item.precision == "day" else None
        previous_id = item.id
    return result
