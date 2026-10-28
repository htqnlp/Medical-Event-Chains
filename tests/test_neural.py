import argparse
import json
import pytest
torch=pytest.importorskip("torch")
from medical_event_chains.neural.model import TinyEncoder, TriggerModel
from medical_event_chains.neural.data import windows, CharacterTokenizer, sentence_features, training_samples, collate, load_documents, label_names
from medical_event_chains.neural.inference import NeuralTrigger, decode_bio
from medical_event_chains.neural.train import train, evaluate
from medical_event_chains.mlee import convert_document

torch.set_num_threads(2)


def fixture_docs():
    return [{"id":"d","text":"今日出现皮疹。昨日发热。","triggers":[{"start":2,"end":4,"type":"Occurrence","text":"出现"}]}]


def test_overlapping_windows_match_paper():
    assert windows(1)==[[0,0,0]]
    assert windows(2)==[[0,1,1]]
    assert windows(5)==[[0,1,2],[1,2,3],[2,3,4]]
    assert [sum(i in w for w in windows(5)) for i in range(5)]==[1,2,3,2,1]


def test_padding_invariance_and_backward():
    torch.manual_seed(3)
    model=TriggerModel(TinyEncoder(40,12),3,8,0).eval()
    ids=torch.tensor([[[2,4,3,0,0],[2,5,6,3,0],[2,7,3,0,0]]])
    mask=(ids!=0).long()
    labels=torch.where(mask.bool(),torch.zeros_like(ids),-100)
    output=model(ids,mask,labels)
    assert output["logits"].shape==(1,3,5,3)
    padded=model(torch.nn.functional.pad(ids,(0,4)),torch.nn.functional.pad(mask,(0,4)))
    assert torch.allclose(output["logits"][mask.bool()],padded["logits"][:,:,:5][mask.bool()],atol=1e-6)
    output["loss"].backward()
    assert model.sentence.weight_ih_l0.grad.abs().sum()>0
    assert model.context.weight_ih_l0.grad.abs().sum()>0
    assert model.encoder.embedding.weight.grad.abs().sum()>0


def test_duplicate_context_not_supervised_twice():
    docs=fixture_docs();tokenizer=CharacterTokenizer.build(docs);labels=label_names(docs)
    samples=training_samples(docs,tokenizer,labels,128)
    assert all(x==-100 for x in samples[0][2]["labels"])
    assert any(x!=-100 for x in samples[0][1]["labels"])


def test_real_bert_architecture_offline():
    transformers=pytest.importorskip("transformers")
    config=transformers.BertConfig(vocab_size=40,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,max_position_embeddings=32)
    model=TriggerModel(transformers.BertModel(config),3,8,0)
    ids=torch.randint(1,40,(2,3,9));mask=torch.ones_like(ids);labels=torch.zeros_like(ids)
    output=model(ids,mask,labels);output["loss"].backward()
    assert output["logits"].shape==(2,3,9,3)
    assert torch.isfinite(output["loss"])


def test_train_save_reload_predict_metrics(tmp_path):
    docs=fixture_docs();dataset=tmp_path/"train.jsonl"
    dataset.write_text('\n'.join(json.dumps(d,ensure_ascii=False) for d in docs),encoding="utf-8")
    args=argparse.Namespace(train=str(dataset),dev=None,output=str(tmp_path/"model"),encoder="tiny",bert="unused",epochs=8,batch_size=1,hidden_size=8,dropout=0.,max_length=128,lr=.02,seed=4,device="cpu",threads=2)
    history=train(args)
    assert history[-1]["loss"]<history[0]["loss"]
    predictor=NeuralTrigger(args.output)
    output=predictor.predict(docs[0]["text"])
    for ann in output: assert docs[0]["text"][ann["start"]:ann["end"]]==ann["text"]
    predictor2=NeuralTrigger(args.output)
    assert output==predictor2.predict(docs[0]["text"])
    class Gold:
        def predict(self,text):return docs[0]["triggers"]
    assert evaluate(Gold(),docs)["f1"]==1


def test_strict_offsets_and_max_length():
    docs=fixture_docs();tokenizer=CharacterTokenizer.build(docs)
    with pytest.raises(ValueError,match="max_length"):sentence_features(docs[0],tokenizer,label_names(docs),3)
    bad={"text":"出现。皮疹。","triggers":[{"start":1,"end":4,"type":"Occurrence"}]}
    with pytest.raises(ValueError,match="boundary"):sentence_features(bad,tokenizer,["O","B-Occurrence","I-Occurrence"],128)


def test_bio_decoder():
    out=decode_bio(["O","I-X","I-X","O"],[(0,0),(0,1),(1,2),(2,3)],"abc")
    assert out==[{"start":0,"end":2,"type":"X","text":"ab"}]


def test_mlee_conversion_and_offsets(tmp_path):
    source=tmp_path/"sample.txt";source.write_text("Protein activates cells.",encoding="utf-8")
    source.with_suffix('.a1').write_text("T1\tProtein 0 7\tProtein\n",encoding="utf-8")
    source.with_suffix('.a2').write_text("T2\tRegulation 8 17\tactivates\nE1\tRegulation:T2 Theme:T1\n",encoding="utf-8")
    converted=convert_document(source)
    assert converted["triggers"]==[{"start":8,"end":17,"text":"activates","type":"Regulation"}]


def test_bert_train_checkpoint_with_local_fast_tokenizer(tmp_path):
    transformers=pytest.importorskip("transformers")
    source=tmp_path/'local-bert';source.mkdir()
    docs=fixture_docs()
    vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]']+sorted(set(docs[0]['text']))
    vocab_path=source/'vocab.txt';vocab_path.write_text('\n'.join(vocab)+'\n',encoding='utf-8')
    tokenizer=transformers.BertTokenizerFast(vocab_file=str(vocab_path),do_lower_case=False)
    tokenizer.save_pretrained(source)
    config=transformers.BertConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,max_position_embeddings=64)
    transformers.BertModel(config).save_pretrained(source)
    dataset=tmp_path/'train.jsonl';dataset.write_text(json.dumps(docs[0],ensure_ascii=False)+'\n',encoding='utf-8')
    args=argparse.Namespace(train=str(dataset),dev=None,output=str(tmp_path/'trained'),encoder='bert',bert=str(source),epochs=1,batch_size=1,hidden_size=8,dropout=0.,max_length=64,lr=.01,seed=4,device='cpu',threads=2)
    train(args)
    predictor=NeuralTrigger(args.output)
    assert predictor.config['encoder_kind']=='bert'
    for ann in predictor.predict(docs[0]['text']):
        assert docs[0]['text'][ann['start']:ann['end']]==ann['text']
