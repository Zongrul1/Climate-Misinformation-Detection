import json

filepath = "dev.json"
number = [5,72,75,97]
with open(filepath, 'r', encoding='UTF-8') as js:
    data_list = []
    data = json.load(js)
    i = 0
    for k, v in data.items():
        if i in number:
            print(i)
            print(str(v['text']))
            print()
        i += 1
