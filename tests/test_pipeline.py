import json
from datetime import date
import pytest
from medical_event_chains import Pipeline
from medical_event_chains.schema import Span
from medical_event_chains.temporal import extract_times, shift_months, chinese_number
from medical_event_chains.text import split_spans
from medical_event_chains.cli import example_input
from medical_event_chains.export import standalone_html
from medical_event_chains.chains import temporal_relation


def record(text, ident="a", patient="A", admission="2026-10-20"):
    return {"record_id":ident,"patient_id":patient,"text":text,"admission_date":admission}


@pytest.mark.parametrize("text,expected", [("三",3),("十",10),("十二",12),("二十五",25),("一百零二",102),("12",12)])
def test_numerals(text, expected):
    assert chinese_number(text) == expected


def test_month_clamp():
    assert shift_months(date(2024,3,31),-1) == date(2024,2,29)
    assert shift_months(date(2024,2,29),12) == date(2025,2,28)


@pytest.mark.parametrize("text,start,end,precision,status", [
    ("2024年2月29日","2024-02-29","2024-02-29","day","resolved"),
    ("2026年2月","2026-02-01","2026-02-28","month","resolved"),
    ("2023年","2023-01-01","2023-12-31","year","resolved"),
    ("4天前","2026-10-16","2026-10-16","day","resolved"),
    ("入院前两周","2026-10-06","2026-10-06","day","resolved"),
    ("术后",None,None,"unknown","unresolved"),
    ("2026年2月30日",None,None,"unknown","invalid"),
])
def test_time_normalization(text,start,end,precision,status):
    times=extract_times(text,"2026-10-20")
    assert len(times)==1
    item=times[0]
    assert (item.start_date,item.end_date,item.precision,item.status)==(start,end,precision,status)
    item.span.validate(text)


def test_next_day_and_unresolved_anchor():
    values=extract_times("2026年10月18日出现皮疹，次日好转，术后次日复诊。","2026-10-20")
    assert values[1].start_date=="2026-10-19"
    assert values[-1].start_date is None
    assert extract_times("4天前出现皮疹")[0].status=="unresolved"


def test_english_and_decimal_segmentation():
    text="Glucose was 7.8 mmol/L. Rash appeared.\n今日皮疹。"
    spans=split_spans(text)
    assert len(spans)==3
    for span in spans: span.validate(text)


def test_demo_source_offsets_and_groups():
    payload=example_input()
    result=Pipeline().run(payload["records"],payload["knowledge_graph"])
    assert len(result["chains"])==2
    assert [g["concept"] for g in result["combined_chains"]]==["皮疹"]
    assert any(e["relation"]=="exact_label_match" for e in result["graph"]["edges"])
    assert not result["graph"]["causality_inferred"]
    for output in result["records"]:
        for event in output["events"]:
            for key in ("trigger","object"):
                Span(**event[key]).validate(output["text"])
            if event["time"]: Span(**event["time"]["span"]).validate(output["text"])
        assert ''.join(b["text"] for b in output["blocks"])==output["text"]
    ids={e["id"]:e for e in result["events"]}
    for chain in result["chains"]:
        for link in chain["links"]:
            assert ids[link["source"]]["patient_id"]==ids[link["target"]]["patient_id"]
    assert json.loads(json.dumps(result,ensure_ascii=False))==result


def test_negation_uncertainty_scoped_to_clause():
    result=Pipeline().run([record("今日无发热，出现皮疹，不排除肺炎，未服用二甲双胍。")])
    events={e["object"]["text"]:e for e in result["events"]}
    assert events["发热"]["negated"]
    assert not events["皮疹"]["negated"]
    assert events["肺炎"]["uncertain"] and not events["肺炎"]["negated"]
    assert events["二甲双胍"]["negated"]


def test_negative_and_uncertain_not_shared_concepts():
    result=Pipeline().run([record("今日出现皮疹。"),record("今日无皮疹，考虑肺炎。","b","B")])
    assert result["combined_chains"]==[]


def test_no_precipitating_cause_is_not_negative_symptom():
    result=Pipeline().run([record("今日无明显诱因出现皮疹，无明显发热。")])
    events={e["object"]["text"]:e for e in result["events"]}
    assert not events["皮疹"]["negated"]
    assert events["发热"]["negated"]


def test_postposed_time_and_unicode_offsets():
    result=Pipeline().run([record("🩺皮疹4天前出现。今日发热。")])
    event=result["events"][0]
    assert event["object"]["start"]==1
    assert event["time"]["start_date"]=="2026-10-16"


def test_coarse_and_unknown_do_not_become_exact_order():
    result=Pipeline().run([record("2023年出现皮疹。2023年5月发热。术后咳嗽。")])
    assert result["chains"][0]["ordering"]=="document_order"
    assert [e["relation"] for e in result["chains"][0]["links"]]==["time_overlap","document_order"]


def test_all_dated_history_sorted_and_records_combined_per_patient():
    result=Pipeline().run([record("今日出现皮疹。4天前发热。"),record("昨日咳嗽。","b","A")])
    by_id={e["id"]:e for e in result["events"]}
    assert len(result["chains"])==1
    dates=[by_id[e]["time"]["start_date"] for e in result["chains"][0]["event_ids"]]
    assert dates==sorted(dates)


@pytest.mark.parametrize("bad", [[],None,[{}],[record(" ")],[record("皮疹"),record("发热")]])
def test_invalid_records(bad):
    with pytest.raises(ValueError): Pipeline().run(bad)


def test_invalid_kg_rejected():
    with pytest.raises(ValueError): Pipeline().run([record("皮疹")],{"nodes":[],"edges":[{"source":"x","target":"y"}]})


def test_html_json_cannot_escape_script():
    result=Pipeline().run([record("</script><img src=x onerror=alert(1)>今日皮疹。")])
    html=standalone_html(result)
    assert "</script><img" not in html
    assert "\\u003c/script>" in html
    assert '<script src="/app.js"' not in html
    assert '<link rel="stylesheet"' not in html
