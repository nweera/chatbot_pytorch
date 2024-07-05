import json
from ntlk_utils import tokenize, stem, bag_of_words

with open('intents.json','r') as f:
    intents = json.load(f)
#print(intents)
all_w = list()
tags = list()
xy = list()
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_w.extend(w)
        xy.append((w, tag))

del_w =['!', '?', ',', '.']
all_w =[stem(w) for w in all_w if w not in del_w]
#print(all_w)
all_w = sorted(set(all_w))
tags = sorted(set(tags))
print(all_w)
