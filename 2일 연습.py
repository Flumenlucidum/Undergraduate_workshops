# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 23:30:14 2018

@author: main
"""

True
False
10
int(11.36)
float(67)
str(1987)

elec =[1948,1960,1980,1988,1993]
elec[1:4]
elec[1:]
elec[-2]
elec[0::2]
1979 in elec
elec_new = [1998,2003,2008,2013,2017]
elec + elec_new
elec.append(1998)
elec
elec.insert(0,1945)
elec

elec.pop(0)
elec
elec[1]=1961
elec

avengers=('Tony', 'Steve', 'Thor', 'Natasha', 'Bruce')
for name in avengers:
    print (name, end=' ')

avspec = [['Tony','M',40, True],['Thor','M',900,False],['Natasha','F',30,True]]
for spec in avspec:
    if spec[2]>=40:
        print(spec)
        
for spec in avspec:
    if spec[1] == 'F':
        print(spec)
'''동등성 확인을 위한 연산자들은 == 
=는 할당을 의미 '''
#unpacking 
for spec in avspec:
    (name, sex, age, human) = spec
    print (age)

a = [1,2,3,4,5]
b = ['Kor', 'Jap', 'Chn', 'Ger', 'USA']

for index, number in enumerate(a):
    print(index, number, b[index])
    #인덱스
c=zip(a,b)
print(c)
print(list(c))
# zip
for d,e in zip(a,b):
    print (d,e)
elec
elec_one = [year+1 for year in elec]
elec_one
elec
elec
elec_demo=[]    
for year in elec:
    if year >1987:
        elec_demo.append(year)
        print(elec_demo)


av=dict()
av['Tony']=40
av['steve']=80
av['Thor']=900
av
av['Natasha']
av.get('Natasha', "unknown")

av2={'QS':30, 'SW':30, 'Vision':5}
av.update(av2)
av
'Thor' in av2
av.setdefault('Natasha', 30)
av
print(av['Natasha']==30)
av.pop('QS')
del av['SW']
''' pop 과 del의 괄호가 다르다는점 '''

print(av)
print (av.keys())


print(av.values())
print(av.items())
for name in av.keys():
    print(name)

for name, age in av.items():
    print ('name={}, age={}'.format(name,age))
    
jiphap = set()    
jiphap.add(23)
print (jiphap)
23 in jiphap



'Tony' + ' Iron Man'
'Thor, Son of Odin'.split('o')
'Th' in "Thor son of odin"
'Thor\nLokey'
print('Thor\nLokey')
# print 웬만하면 써야할듯 
'Thor, Son of Odin'.find('Son')
'Thor, Son of Odin'[7]
'Thor, Son of Odin'[5:8]
#끝에는 제외하는 듯 
len('Thor, Son of Odin')
'Thor, Son of Odin'.startswith('LO') is False
'    Thor, Son of Odin    '.strip()
'Thor, Son of Odin'.lower()
'Thor, Son of Odin'.upper()
'{}, Son of {}'.format('Thor','Odin')
'{}, Son of {}'.format('Lokey','Laufei')

' '.join(['I', 'am', 'your', 'father'])

'사랑해'.encode('utf8')
print('Thor, Son of Odin''\t''Lockey')
print('\'Thor, Son of Odin\'')
#따옴표쓰는거는 딱히 그대로 근데 개행이나 탭은 따옴표 따로 

text = '프로그램 언어를 익히기 위해 책이나 글만 보면서 따라해서는 중간에 막히는 부분들이 발생합니다. 그리고 막연히 어렵게 느껴지기도 하고요. 또 어떤 경우에는 눈으로만 읽는 분들이 있는데, 눈으로만 봐서는 실제로 프로그램을 작성하기가 어렵습니다. 본 과정은 실습을 중심으로 진행합니다. 그래서, 따라할 수 있는 형태의 강의 자료가 제공됩니다. 온라인에 공개되기 때문에 수업을 듣지 않은 분들도 자료를 열람할 수 있지만, 실습을 진행하면서 발생하는 Q&A나 개별 1:1 지도, 각 개인의 프로젝트 목표에 대한 피드백 등은 제한된 메일링 리스트를 사용하여 진행합니다.'
new_text = text.split(' ')
print(new_text)
new_text =text.replace(',', ' ').replace('.',' ')
print(new_text)
word_list=new_text.split(' ')
print(word_list)
text = '프로그램 언어를 익히기 위해 책이나 글만 보면서 따라해서는 중간에 막히는 부분들이 발생합니다. 그리고 막연히 어렵게 느껴지기도 하고요. 또 어떤 경우에는 눈으로만 읽는 분들이 있는데, 눈으로만 봐서는 실제로 프로그램을 작성하기가 어렵습니다. 본 과정은 실습을 중심으로 진행합니다. 그래서, 따라할 수 있는 형태의 강의 자료가 제공됩니다. 온라인에 공개되기 때문에 수업을 듣지 않은 분들도 자료를 열람할 수 있지만, 실습을 진행하면서 발생하는 Q&A나 개별 1:1 지도, 각 개인의 프로젝트 목표에 대한 피드백 등은 제한된 메일링 리스트를 사용하여 진행합니다.'

new_text = text.replace(',', '').replace('.', '')
word_list = new_text.split()
word_unique_set = set()

for word in word_list:
    word_unique_set.add(word)

print('Total words: {}'.format(len(word_unique_set)

word_freq_dict=dict()
if word in word_list:
    word_freq_dict = dict()
for word in word_list:
    if word not in word_freq_dict:
        word_freq_dict[word] = 0
    word_freq_dict[word] = word_freq_dict[word] + 1
print('Total words: {}'.format(len(word_freq_dict)))


text ='웁살라 대학교에서는 언어강좌를 우선 듣게 되지만 더욱 중요한 것은 제 전공인 사회학과 경제학입니다. 언어는 그 사회와 뗄 수 없는 관계를 맺고 사회의 경제 정치 역사적 상황에 따라 변하게 되기 때문입니다. 따라서 사회의 변화와 발전에 대한 강좌를 수강할 계획입니다. 여름에 Basic Swedish 강좌를 들으며 스웨덴어의 문법 통사 구조와 발음들을 집중적으로 공부할 예정입니다. 강좌 과정에서 다양한 교환학생들과 영어로 대화를 하며 그들의 사회 문화적 분위기와 언어에 대해 대화를 구체적으로 나눠볼 생각입니다. 한국에서 배운 독일어와 스웨덴어의 동사변화나 발음구조의 유사점과 차이점이 무엇이고, 왜 그런지에 대해서도 개별적으로 공부해볼 생각입니다.'
new_text = text.replace(',','').replace('.','')
new_text
word_list = new_text.split()
word_list
bindo=dict()
for word in word_list:
    if word not in bindo:
        bindo[word]=0
    bindo[word]=bindo[word]+1
print ('chong bindosu:{}'.format(len(bindo)))
bindo

top_five_words = sorted(bindo.items(), key=lambda x: x[1], reverse=True)[:5]
for word, freq in top_five_words:
      print(','.join([word, str(freq)])
      
topfive=sorted (bindo.items(),key=lambda x: x[1], reverse=True)[:10]
print(topfive)
for word, freq in topfive:
    print(','.join([word,str(freq)]))
    
for i in [0, 1, 2]:
      for j in [0, 1, 2]:
          print(i, j)