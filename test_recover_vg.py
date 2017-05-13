import json

import re, math
from collections import Counter, defaultdict
WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text, stops):
    words = WORD.findall(text)
    goods = [w for w in words if w not in stops]
    return Counter(goods)



cap_dict={}
print'reading input file'
data=json.load(open('/users/ud2017/hoavt/data/VG100k/region_descriptions.json'))
count=0
for image in data:
    #count+=1
    #if count > 100: break
    id_=image.get('id')
    phrases = []
    for region in image.get('regions'):
        phrase=region['phrase'].lower()
        phrases.append(phrase) #[x['phrase'] for x in image.get('regions')]
    key = " ".join(phrases)
    cap_dict[key] = id_

import operator
stops=set(['to', 'has', 'are', 'for', 'by', 'on', 'of', 'a', 'with', 'this', 'and', 'an', 'at', 'in', 'the'])
#convert to vector
print 'converting to vectors'
vector_dict={}
for key in cap_dict:
    id_ = cap_dict[key]
    vector_dict[id_] = text_to_vector(key, stops)

caption1='a green refrigerator white refrigerator and brown chair are in a room with an open window'
vector1=text_to_vector(caption1, stops)
caption2='the brilliant white snow serves as a back drop as five men wearing snow boards and winter attire plan their snow boarding adventures'
vector2=text_to_vector(caption2, stops)
caption3=' women sitting at a fancy table looking at her food'
vector3=text_to_vector(caption3, stops)
scores={key: get_cosine(vector1, vector_dict[key]) for key in vector_dict}
scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
print scores[:3]

scores={key: get_cosine(vector2, vector_dict[key]) for key in vector_dict}
scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
print scores[:3]

scores={key: get_cosine(vector3, vector_dict[key]) for key in vector_dict}
scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
print scores[:3]

