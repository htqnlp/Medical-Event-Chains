from medical_event_chains.backends import PaddleBackend
from medical_event_chains import Pipeline


def test_mock_adapter_missing_categories_boundary_and_dedup():
    text="今日出现皮疹。"
    def factory(task, **kwargs):
        if task=="information_extraction":
            return lambda text:[{"时间":[{"text":"今日","start":0,"end":2}],"症状":[{"text":"皮疹","start":4,"end":6}]}]
        return lambda text:[{"text":text,"items":[
            {"item":"今日","offset":0,"length":2,"wordtag_label":"时间类"},
            {"item":"出现","offset":2,"length":2,"wordtag_label":"场景事件"},
            {"item":"皮疹","offset":4,"length":2,"wordtag_label":"疾病症状"}]}]
    result=Pipeline(PaddleBackend(taskflow_factory=factory)).run([{"record_id":"r","patient_id":"p","text":text}])
    # The WordTag ambiguous label and UIE category are retained as typed mentions;
    # UIE and WordTag times merge, with a trigger at the boundary preserved.
    assert len(result["records"][0]["times"])==1
    assert all(e["trigger"]["text"]=="出现" for e in result["events"])


def test_hybrid_does_not_invent_trigger():
    class Predictor:
        def predict(self,text): return []
    def factory(task,**kwargs):
        if task=="information_extraction":return lambda text:[{"症状":[{"text":"皮疹","start":0,"end":2}]}]
        return lambda text:[{"text":text,"items":[]}]
    backend=PaddleBackend(trigger_predictor=Predictor(),taskflow_factory=factory)
    assert backend.extract("皮疹")[0]==[]
