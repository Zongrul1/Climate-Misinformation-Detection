import json

filepath = "dev.json"
with open(filepath, 'r', encoding='UTF-8') as js:
    label_list1 = []
    data = json.load(js)
    for k, v in data.items():
        label_list1.append(str(v['label']))

filepath = "dev-output.json"
with open(filepath, 'r', encoding='UTF-8') as js:
    label_list2 = []
    data = json.load(js)
    for k, v in data.items():
        label_list2.append(v['label'])

print(label_list1)
print(label_list2)
i = 0
result_list1 = []
result_list2 = []
for i in range(len(label_list1)):
    if label_list1[i] == '0' and label_list2[i] == '1':
        result_list1.append(i)
    if label_list1[i] == '1' and label_list2[i] == '0':
        result_list2.append(i)
print(result_list1)
print(result_list2)