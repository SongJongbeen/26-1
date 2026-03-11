import json

with open('sos507/wk02_0310/assignment/full_abstracts.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(len(data))

# 2706
