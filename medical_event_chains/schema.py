"""Source-preserving, end-exclusive character spans (Unicode code points)."""
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Optional

TYPES = ("Occurrence", "Check", "Diagnosis", "Surgery", "Treatment")


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int

    def validate(self, source: str):
        if not 0 <= self.start < self.end <= len(source) or source[self.start:self.end] != self.text:
            raise ValueError(f"Invalid source span: {self!r}")
        return self


@dataclass
class Time:
    id: str
    span: Span
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    precision: str = "unknown"
    anchor: Optional[str] = None
    status: str = "unresolved"


@dataclass
class Mention:
    trigger: Span
    object: Span
    type: str
    negated: bool = False
    uncertain: bool = False
    provenance: str = "rule"


@dataclass
class Record:
    record_id: str
    patient_id: str
    text: str
    admission_date: Optional[str] = None

    @classmethod
    def parse(cls, item):
        if not isinstance(item, dict):
            raise ValueError("Each record must be an object")
        for key in ("record_id", "patient_id", "text"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Record requires a nonempty string: {key}")
        if len(item["text"]) > 100_000:
            raise ValueError("Record text exceeds 100000 characters")
        admission = item.get("admission_date")
        if admission is not None:
            if not isinstance(admission, str):
                raise ValueError("admission_date must be YYYY-MM-DD or null")
            date.fromisoformat(admission)
        return cls(**{key: item[key] for key in ("record_id", "patient_id", "text")}, admission_date=admission)


def jsonable(value):
    return asdict(value)
