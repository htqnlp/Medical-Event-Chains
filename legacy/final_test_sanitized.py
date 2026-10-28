# Archival surviving script; NOT the runnable reconstructed pipeline.
# Only the embedded narrative was replaced; other exploratory code is retained.
#python -m pip install --upgrade paddlenlp
#pip install LAC --upgrade

from paddlenlp import Taskflow
import cv2
import matplotlib

# from pprint import pprint
 
# from paddlenlp import Taskflow

matplotlib.use('Agg')

 
schema = ['时间', '症状', '诊断', '检查', '治疗', '手术'] # Define the schema for entity extraction
 
ie = Taskflow('information_extraction', schema=schema, model="uie-medical-base")


# ddp = Taskflow("dependency_parsing", return_visual=True)
ddp = Taskflow("dependency_parsing")
wordtag = Taskflow("knowledge_mining", model="wordtag", linking=True)

text = "4天前出现皮疹，今日接受抗过敏治疗。"  # Synthetic replacement for an unverified clinical narrative

wordtag_result = wordtag(text)


result = ie(text)


wordtag_result_0 = wordtag_result[0]

text = wordtag_result_0['text']

items = wordtag_result_0['items']


print('result:')
print(result)

print('items:')
print(items)


events = []   # 场景事件
times = []    # 时间
segment = []  # 句子分词
pos = []
times_pos = []
events_pos = []
# segment_pos = []
wordtag_label = []

for i in range(0, len(items)):
    items_i = items[i]
    if items_i['wordtag_label'] == '场景事件':
        events.append(items_i['item'])
        events_pos.append((items_i['offset'], items_i['length'] + items_i['offset'] - 1))
    if items_i['wordtag_label'] == '时间类':
        times.append(items_i['item'])
        times_pos.append((items_i['offset'], items_i['length'] + items_i['offset'] - 1))
    segment.append(items_i['item'])
    end1 = items_i['length'] + items_i['offset'] - 1
    pos.append((items_i['offset'], end1))
    wordtag_label.append(items_i['wordtag_label'])



# 分词加具体位置
segment_all = []
for i in range(0, len(segment)):
    segment_all.append((segment[i], pos[i]))


ddp_result = ddp.from_segments([segment])  # 根据分词结果做句法依存分析


# 由句法依存分析得出的句子关系
ddp_deprel = []
ddp_w = ddp_result[0]['word']
ddp_h = ddp_result[0]['head']
ddp_d = ddp_result[0]['deprel']


head_pos = []  # 关系位置
head_label = []  # 关系标签
for i in range(0, len(ddp_h)):
    num = ddp_h[i] - 1
    if num == -1:
        head_pos_in = segment_all[i][1]
        head_label_in = wordtag_label[i]
    else:
        head_pos_in = segment_all[num][1]
        head_label_in = wordtag_label[num]
    head_pos.append(head_pos_in)
    head_label.append(head_label_in)


for i in range(0, len(ddp_h)):
    word_0 = ddp_w[i]
    if ddp_h[i] == 0:
        relation_0 = '无，为HED词'
    else:
        relation_0 = ddp_w[ddp_h[i] - 1]
    deprel_0 = ddp_d[i]
    ddp_deprel.append((word_0, relation_0, deprel_0, segment_all[i][1], head_pos[i], head_label[i]))



# events加具体位置
events_all = []
for i in range(0, len(events)):
    events_all.append((events[i], events_pos[i]))


times_all = []
for i in range(0, len(times)):
    times_all.append((times[i], times_pos[i]))


event_stream = []  # 最终事件流

text_list = list(text)

result_0 = result[0]

# 找位置出来

# ill = result_0['疾病']

# print('ill:')
# print(ill)


if result_0.__contains__('时间'):
    time_uie = result_0['时间']
else:
    time_uie = []

if result_0.__contains__('手术'):
    operate_uie = result_0['手术']
else:
    operate_uie = []

if result_0.__contains__('检查'):
    examination_uie = result_0['检查']
else:
    examination_uie = []

if result_0.__contains__('治疗'):
    cure_uie = result_0['治疗']
else:
    cure_uie = []

if result_0.__contains__('症状'):
    symptom_uie = result_0['症状']
else:
    symptom_uie = []

if result_0.__contains__('诊断'):
    dignosis_uie = result_0['诊断']
else:
    dignosis_uie = []


print('examination_uie:')
print(examination_uie)

print('cure_uie:')
print(cure_uie)

print('dignosis_uie:')
print(dignosis_uie)

print('operate_uie:')
print(operate_uie)


# 把times_all里的时间加到time_uie里面，防止抽取的时间不足
for i in range(0, len(times_all)):
    times_all_text = times_all[i][0]
    if len(time_uie) == i:
        times_add = {'end': times_all[i][1][1] + 1, 'start': times_all[i][1][0], 'text': times_all_text}
        time_uie.append(times_add)

# print('time_uie:')
# print(time_uie)


# time_uie = result_0['时间']
# operate_uie = result_0['手术']
# examination_uie = result_0['检查']
# cure_uie = result_0['治疗']
# symptom_uie = result_0['症状']
# dignosis_uie = result_0['诊断']


symptom_uie = result_0['症状']
# 时间排序

# time_uie_sort = []
# time_uie_sort.append(time_uie[0])

for i in range(len(time_uie)):
    for j in range(i + 1, (len(time_uie))):
        if time_uie[i]['start'] > time_uie[j]['start']:
            time_uie[i], time_uie[j] = time_uie[j], time_uie[i]


print('time_uie:')
print(time_uie)

# 分割句子（根据时间）找中间的事件（症状提取）
sentence_part = []
for i in range(0, len(time_uie)):
    if i == len(time_uie) - 1:
        part_start = time_uie[i]['end']
        part_end = len(text_list)
        part_text = time_uie[i]['text']
        time_start = time_uie[i]['start']
        time_end = time_uie[i]['end']
    else:
        part_start = time_uie[i]['end']
        part_end = time_uie[i + 1]['start']
        part_text = time_uie[i]['text']
        time_start = time_uie[i]['start']
        time_end = time_uie[i]['end']

    sentence_part.append(((time_start, time_end), (part_start, part_end), part_text))

# 症状提取
sy = []
cu = []
op = []


for i in range(0, len(sentence_part)):
    start = sentence_part[i][1][0]
    end = sentence_part[i][1][1]
    sy_time_start = sentence_part[i][0][0]
    sy_time_end = sentence_part[i][0][1]
    time_text = sentence_part[i][2]
    for k in range(0, len(symptom_uie)):
        if symptom_uie[k]['start'] > start and symptom_uie[k]['end'] < end:
            sy.append(
                ((sy_time_start, sy_time_end), (start, end), time_text,
                 (symptom_uie[k]['start'], symptom_uie[k]['end'], symptom_uie[k]['text'])))

    if len(cure_uie) != 0:
        for u in range(0, len(cure_uie)):
            if cure_uie[u]['start'] > start and cure_uie[u]['end'] < end:
                cu.append(('', time_text, (cure_uie[u]['start'], cure_uie[u]['end']), cure_uie[u]['text']))
    
    if len(operate_uie) != 0:
        for m in range(0, len(operate_uie)):
            if operate_uie[m]['start'] > start and operate_uie[m]['end'] < end:
                op.append(('', time_text, (operate_uie[m]['start'], operate_uie[m]['end']), operate_uie[m]['text']))


# print()

# sy排序（症状、发病实体位置排序）
for i in range(len(sy)):
    for j in range(i + 1, len(sy)):
        if sy[i][0] == sy[j][0] and sy[i][3][0] > sy[j][3][0]:
            sy[i], sy[j] = sy[j], sy[i]

# 检查事件
ex = []

for i in range(0, len(sentence_part)):
    start = sentence_part[i][1][0]
    end = sentence_part[i][1][1]
    ex_time_start = sentence_part[i][0][0]
    ex_time_end = sentence_part[i][0][1]
    time_text = sentence_part[i][2]
    for k in range(0, len(examination_uie)):
        if examination_uie[k]['start'] > start and examination_uie[k]['end'] < end:
            ex.append(
                ((ex_time_start, ex_time_end), (start, end), time_text,
                 (examination_uie[k]['start'], examination_uie[k]['end'], examination_uie[k]['text'])))

# print()


# items 优化一下
for i in range(0, len(items)):
    if items[i]['item'] == '报告':
        items[i]['wordtag_label'] = '场景事件'



# 检查排序

for i in range(len(examination_uie)):
    for j in range(i + 1, (len(examination_uie))):
        if examination_uie[i]['start'] > examination_uie[j]['start']:
            examination_uie[i], examination_uie[j] = examination_uie[j], examination_uie[i]

# 症状排序

for i in range(len(cure_uie)):
    for j in range(i + 1, (len(cure_uie))):
        if cure_uie[i]['start'] > cure_uie[j]['start']:
            cure_uie[i], cure_uie[j] = cure_uie[j], cure_uie[i]


# 治疗排序

for i in range(len(symptom_uie)):
    for j in range(i + 1, (len(symptom_uie))):
        if symptom_uie[i]['start'] > symptom_uie[j]['start']:
            symptom_uie[i], symptom_uie[j] = symptom_uie[j], symptom_uie[i]

# 诊断排序

for i in range(len(dignosis_uie)):
    for j in range(i + 1, (len(dignosis_uie))):
        if dignosis_uie[i]['start'] > dignosis_uie[j]['start']:
            dignosis_uie[i], dignosis_uie[j] = dignosis_uie[j], dignosis_uie[i]


# 手术排序

for i in range(len(operate_uie)):
    for j in range(i + 1, (len(operate_uie))):
        if operate_uie[i]['start'] > operate_uie[j]['start']:
            operate_uie[i], operate_uie[j] = operate_uie[j], operate_uie[i]

event_stream = []




# 根据实体和句法关系整合事件（发病、疾病）

# time_pos_in_items = 0
event_pos_in_items = 0
before_event_pos_in_items = 0

for i in range(0, len(sy)):
    sy_i = sy[i]
    # sy_time_get = sy_i[0][0]
    sy_e_get = sy_i[3][0]
    for j in range(0, len(items)):
        # if items[j]['offset'] == sy_time_get:
        #     time_pos_in_items = j
        if items[j]['offset'] == sy_e_get:
            event_pos_in_items = j

    if i == 0:
        for k in range(event_pos_in_items - 1, 0, -1):
            if items[k]['wordtag_label'] == '场景事件' and abs(event_pos_in_items - k) <= 2:
                event_stream.append((sy_i[0], sy_i[2], items[k]['item'], sy_i[3][2], (sy_i[3][0], sy_i[3][1]), '症状'))
                before_event_pos_in_items = event_pos_in_items
                event_pos_in_items = 0
                break
    else:
        for m in range(event_pos_in_items - 1, before_event_pos_in_items, -1):
            if items[m]['wordtag_label'] == '场景事件' and abs(event_pos_in_items - m) <= 2:
                event_stream.append((sy_i[0], sy_i[2], items[m]['item'], sy_i[3][2], (sy_i[3][0], sy_i[3][1]), '症状'))
                before_event_pos_in_items = event_pos_in_items
                event_pos_in_items = 0
                # before_event_pos_in_items = 0
                break

            else:
                if items[event_pos_in_items + 1]['wordtag_label'] == '场景事件':
                    event_stream.append((sy_i[0], sy_i[2], sy_i[3][2], items[event_pos_in_items + 1]['item'], (sy_i[3][0], sy_i[3][1]), '症状'))
                    break

    for p in range(0, len(examination_uie)):
        if abs(examination_uie[p]['end'] - symptom_uie[i]['start']) <= 4:
            event_stream.append(
                (sy_i[0], sy_i[2], sy_i[3][2], examination_uie[p]['text'], (sy_i[3][0], sy_i[3][1]), '检查'))
            break

    # for y in range(0, len(dignosis_uie)):
    #     if abs(dignosis_uie[y]['end'] - symptom_uie[i]['start']) <= 3:
    #         event_stream.append(
    #             (sy_i[0], sy_i[2], sy_i[3][2], dignosis_uie[y]['text'], (sy_i[3][0], sy_i[3][1]), '诊断'))
    #         break


# 插入治疗事件
for i in range(0, len(cu)):
    event_stream.append(('', cu[i][1], '', cu[i][3], cu[i][2], '治疗'))


# 插入手术事件
for i in range(0, len(op)):
    event_stream.append(('', op[i][1], '', op[i][3], op[i][2], '手术'))


# 判断诊断事件
judge_dig = ['诊断', '判断', '确诊']
for i in range(0, len(event_stream)):
    if event_stream[i][2] in judge_dig:
        event_stream_i_list = list(event_stream[i])
        event_stream_i_list[5] = '诊断'
        event_stream[i] = tuple(event_stream_i_list)


judge_exa = ['提示', '检查']
for i in range(0, len(event_stream)):
    if event_stream[i][2] in judge_exa:
        event_stream_i_list = list(event_stream[i])
        event_stream_i_list[5] = '检查'
        event_stream[i] = tuple(event_stream_i_list)


# 重组event_stream


for i in range(0, len(event_stream)):
    for j in range(i, len(event_stream)):
        if event_stream[i][1] == event_stream[j][1] and event_stream[i][3] == event_stream[j][3] and event_stream[i][5] != event_stream[j][5]:
            result1_list_i = list(event_stream[i])
            result1_list_i[5] = event_stream[i][5] + ', ' + event_stream[j][5]
            event_stream[i] = tuple(result1_list_i)

            event_stream.remove(event_stream[j])
            break



# for i in range(len(event_stream)):
#     for j in range(i + 1, (len(event_stream))):
#         if event_stream[i][1] == event_stream[j][1] and event_stream[i][4][0] > event_stream[j][4][0]:
#             event_stream[i], event_stream[j] = event_stream[j], event_stream[i]



for i in range(0, len(dignosis_uie)):
    event_stream.append(('诊断:', dignosis_uie[i]['text']))



print('event_stream:')
print(event_stream)




