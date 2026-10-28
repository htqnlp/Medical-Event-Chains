"""Offline rule baseline and optional PaddleNLP adapters. No import-time downloads."""
import re
from .schema import Span, Mention
from .text import assertion, split_spans

LEXICON = {
    "Occurrence": ["皮疹", "发热", "咳嗽", "头晕", "口干", "乏力", "四肢麻木", "视物模糊", "吞咽困难", "胸痛", "腹痛", "恶心", "呕吐", "血糖升高"],
    "Diagnosis": ["2型糖尿病", "糖尿病", "高血压", "冠心病", "肺炎", "食管癌", "湿疹", "药物性皮炎"],
    "Check": ["血常规", "血糖", "胸部CT", "CT检查", "CT", "胃镜", "病理检查", "血液检查"],
    "Treatment": ["抗过敏治疗", "抗感染治疗", "降糖治疗", "胰岛素", "甲钴胺", "氯雷他定", "二甲双胍", "饮食控制", "对症治疗", "化疗", "放疗"],
    "Surgery": ["食管切除术", "肿瘤切除术", "手术切除", "手术"],
}
TRIGGERS = {
    "Occurrence": ["出现", "发生", "发现", "伴有", "伴", "诉"],
    "Diagnosis": ["诊断为", "诊断", "确诊", "考虑", "患有"],
    "Check": ["检查", "检测", "查", "行", "示"],
    "Treatment": ["给予", "予以", "服用", "口服", "接受", "使用", "治疗", "用"],
    "Surgery": ["接受", "行", "实施", "进行"],
}


def find_terms(text, terms):
    if not terms:
        return []
    pattern = "|".join(re.escape(s) for s in sorted(set(terms), key=lambda s: (-len(s), s)))
    return [Span(m[0], m.start(), m.end()) for m in re.finditer(pattern, text)]


def nearest_trigger(text, obj, event_type, candidates=None):
    clause = next((s for s in split_spans(text, True) if s.start <= obj.start < s.end), None)
    found = candidates if candidates is not None else find_terms(text, TRIGGERS[event_type])
    found = [s for s in found if clause and clause.start <= s.start < clause.end]
    # Prefer preceding triggers; use the entity as an explicit implicit trigger otherwise.
    before = [s for s in found if s.end <= obj.start]
    return max(before, key=lambda s: s.start) if before else min(found, key=lambda s: abs(s.start - obj.start), default=obj)


class RuleBackend:
    metadata = {"name": "rule", "trained": False, "description": "Offline lexicon/regex baseline; not the paper's trained model"}

    def extract(self, text):
        mentions = []
        occupied = []
        # Prioritize long entities globally (e.g. 抗过敏治疗 before generic terms).
        entities = [(span, kind) for kind, terms in LEXICON.items() for span in find_terms(text, terms)]
        for obj, kind in sorted(entities, key=lambda x: (-(x[0].end-x[0].start), x[0].start)):
            if any(obj.start < end and start < obj.end for start, end in occupied):
                continue
            occupied.append((obj.start, obj.end))
            negated, uncertain = assertion(text, obj)
            mentions.append(Mention(nearest_trigger(text, obj, kind), obj, kind, negated, uncertain))
        return sorted(mentions, key=lambda m: m.object.start), []


class PaddleBackend:
    """UIE medical arguments plus WordTag triggers, based on the surviving script.

    This is an adapter, not the paper's BERT-BiLSTM-BiGRU trigger detector.
    Supply a trained NeuralTrigger predictor for the hybrid paper-style backend.
    """
    def __init__(self, device_id=-1, trigger_predictor=None, taskflow_factory=None):
        if taskflow_factory is None:
            try:
                from paddlenlp import Taskflow
            except ImportError as exc:
                raise RuntimeError("Install PaddleNLP in a separate compatible environment; see docs/paddle.md") from exc
            taskflow_factory = Taskflow
        self.uie = taskflow_factory("information_extraction", schema=["时间", "症状", "诊断", "检查", "治疗", "手术"], model="uie-medical-base", device_id=device_id)
        self.wordtag = taskflow_factory("knowledge_mining", model="wordtag", linking=True, device_id=device_id)
        self.trigger_predictor = trigger_predictor
        self.metadata = {"name": "hybrid" if trigger_predictor else "paddle", "trained": True,
                         "description": "UIE + WordTag arguments + " + ("reconstructed trained trigger model" if trigger_predictor else "WordTag scene triggers (legacy adapter)")}

    def extract(self, text):
        uie = self.uie(text)[0]
        tagged = self.wordtag(text)[0]
        if tagged.get("text", text) != text:
            raise ValueError("WordTag changed source text; cannot trust offsets")
        times, triggers, arguments = [], [], []
        for item in tagged.get("items", []):
            span = Span(item["item"], item["offset"], item["offset"] + item["length"]).validate(text)
            label = item.get("wordtag_label", "")
            if label == "时间类":
                times.append(span)
            if label == "场景事件":
                triggers.append(span)
            for key, kind in (("症状", "Occurrence"), ("疾病", "Diagnosis"), ("药物", "Treatment"), ("治疗", "Treatment"), ("检查", "Check"), ("手术", "Surgery")):
                if key in label:
                    arguments.append((span, kind))
                    break
        mapping = {"症状": "Occurrence", "诊断": "Diagnosis", "检查": "Check", "治疗": "Treatment", "手术": "Surgery"}
        for category, items in uie.items():
            for item in items:
                span = Span(item["text"], item["start"], item["end"]).validate(text)
                if category == "时间":
                    times.append(span)
                elif category in mapping:
                    arguments.append((span, mapping[category]))
        neural = self.trigger_predictor.predict(text) if self.trigger_predictor else None
        # UIE's explicit schema has priority over a broad WordTag label for the
        # same source span; WordTag still supplies additional arguments.
        uie_kinds = {}
        for category, items in uie.items():
            if category in mapping:
                for item in items:
                    uie_kinds[(item["start"], item["end"])] = mapping[category]
        unique, mentions = set(), []
        for obj, kind in arguments:
            kind = uie_kinds.get((obj.start, obj.end), kind)
            key = (obj.start, obj.end, kind)
            if key in unique:
                continue
            unique.add(key)
            if neural is not None:
                # Custom clinical labels must be one of the five chain types.
                # MLEE labels are intentionally not mapped to clinical types automatically.
                candidates = [Span(p["text"], p["start"], p["end"]) for p in neural if p["type"] == kind]
                if not candidates:
                    continue
                clause = next((s for s in split_spans(text, True) if s.start <= obj.start < s.end), None)
                if not any(clause and clause.start <= p.start < clause.end for p in candidates):
                    continue
            else:
                candidates = triggers
            trigger = nearest_trigger(text, obj, kind, candidates)
            negated, uncertain = assertion(text, obj)
            mentions.append(Mention(trigger, obj, kind, negated, uncertain, self.metadata["name"]))
        return sorted(mentions, key=lambda m: m.object.start), times
