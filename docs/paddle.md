# Optional PaddleNLP adapter

The dependency-free rule pipeline and the PyTorch trigger model do not need Paddle. The adapter is optional because the original script supplied no version lock, and the older Taskflow models depend on a different runtime stack.

Use a separate environment with a Python/Paddle/PaddleNLP combination supported by the [official Paddle installation selector](https://www.paddlepaddle.org.cn/install/quick) and [PaddleNLP Taskflow documentation](https://paddlenlp.readthedocs.io/en/stable/source/paddlenlp.taskflow.taskflow.html). Legacy Taskflow installations may require older Python versions. Do not mix incompatible Paddle and PyTorch CUDA wheels into the same environment without checking upstream requirements.

```bash
# Inside that compatible environment, after installing Paddle and PaddleNLP:
python -m pip install -e .
mec extract --input records.json --backend paddle --output artifacts/paddle
```

The first call may download `uie-medical-base` and `wordtag` model assets from their upstream hosts. The default adapter requests CPU (`device_id=-1`). The `--device` option applies to the optional PyTorch neural trigger model, not to Paddle device selection.

The code uses the surviving script's Taskflow interfaces:

```python
Taskflow("information_extraction", schema=["时间","症状","诊断","检查","治疗","手术"], model="uie-medical-base", device_id=-1)
Taskflow("knowledge_mining", model="wordtag", linking=True, device_id=-1)
```

Returned UIE spans must contain `text/start/end`. WordTag items must contain `item/offset/length/wordtag_label`. The adapter verifies offsets and rejects transformed source text. Broad WordTag labels containing 症状/疾病/药物/治疗/检查/手术 can supply arguments; UIE schema typing takes precedence for an identical span. WordTag 场景事件 provides scene triggers and 时间类 adds temporal candidates.

The adapter was tested using mocks shaped like these APIs. **A real Paddle model download and inference were not run in the reconstruction environment**, so installed-version differences or unavailable remote model assets may require adapter changes. Errors are explicit; there is no hidden fallback to rules. This status is also recorded in the project validation notes.

For `--backend hybrid`, also install the neural extras and supply a trained **clinical** checkpoint. The model predicts clinical triggers; UIE/WordTag supplies arguments. MLEE biomedical labels require a separate, explicit clinical label mapping or clinical retraining and are not relabeled automatically.
