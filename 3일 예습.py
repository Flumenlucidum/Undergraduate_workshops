# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:07:40 2018

@author: main
"""
C:\Users\main\.spyder-py3\python-basic-lecture-master\python-basic-lecture-master
#r 을 앞에 눌러 
fin = open(r'C:\Users\main\.spyder-py3\python-basic-lecture-master\assets\day1-example-read.txt')
''' \ \ 이렇게 두번쓰거나 앞에 r escape 문자 안쓰겠다, 역슬래시 대신 슬래시 써도 괜찮'''

import os
os.listdir('.')   #이런 파일이 있는 디렉토리가 기본 
help(open)  # 도움말
fin = open('assets/day1-example-read.txt')
content =fin.read()
print(content, end='')
fin.close()

#file객체에게 with와 write 가능 
with open(r'C:\Users\main\.spyder-py3\python-basic-lecture-master\assets\day1-example-read.txt') as fin:
    content = fin.read()
print (content,end='')

fin = open(r'C:\Users\main\.spyder-py3\python-basic-lecture-master\assets\day1-example-read.txt')
for line in fin:
    print (line, end='')
fin.close()

#파일 기록
with open('outputs\myoutput.txt', 'w', encoding='utf8') as fout:
    fout.write('안녕 텍스트')
print (fout)
'''기본은 read 모드
read()  readline()  write(str)
'''
import os
import csv
#엑셀에 csv저장할때  " " 감싸주는 등 
with open(os.path.join('outputs','basic-csv writer.txt'),'w',encoding='utf8')as fout:
    writer = csv.writer(fout)
    writer.writerow(['안녕 텍스트', 'https://www.wikipedia.org'])
    writer.writerow(['안녕, 파이썬','https://python.org'])
print(csv.writer)
''' append 하는 모드는 'w' 대신 'a'
'''
import os
os.path.join('outputs', 'basic-1-csv-writer.txt')
fin = open('outputs\\basic-1-csv-writer.txt')

#인코딩하는 습관

fine=open('plan.txt')
content=fine.read()
print(content,end='')
#함수
def multi(a,b):
    return a*b
multi(56,67)

def hello(name):
    print('hello,{}'.format(name))
    
hello('hey')
import networkx as nx

G = nx.Graph()
G.add_node(1)
G.add_node(2)
G.add_edge(1, 2)

pos = nx.spring_layout(G)
nx.draw_networkx_edges(G, pos, alpha=0.2);
nx.draw_networkx_nodes(G, pos, node_size=50);

def draw(graph):
    pos = nx.spring_layout(graph)
    nx.draw_networkx_edges(graph, pos, alpha=0.2);
    ax = nx.draw_networkx_nodes(graph, pos, node_size=50);
    return ax

gra =nx.Graph()
gra.add_node(1)
gra.add_node(2)
gra.add_edge(1,2)
draw (gra)

gra =nx.Graph()
gra.add_node(1)
gra.add_node(2)
gra.add_edge(1,2)
draw (gra)

def positional(a,b):
    print(a,b)
positional('Thor', 'Lockey')

def keyword(c=None,d='hero'):
    print('Hey',c,d)
keyword('tony', 'stark')
keyword('thor')

a_list =3.141592
a_func=round
print (a_func(a_list))

a_list=[1.6,1.5,2.7]
def clean(lista, func=round):
    return list([func(element) for element in lista])
clean(a_list)

'''엑셀 파일다루는 라이브러리로 판다스 사용 '''
import pandas as pd
df =pd.DataFrame([1.536,1.845,2.376])
df
#데이터 프레임 변환도구 중 하나인 apply
rounded_df = df.apply(round)
rounded_df
round_two_decimal_point= lambda val : round(val,2)
df =pd.DataFrame([1.536,1.845,2.376])
rounded_df = df.apply(round_two_decimal_point)
rounded_df

import pandas as pd
df1=pd.DataFrame([2.345,1.456,1.789])
rounded_df1 = df1.apply(round)
rounded_df1

r2dp=lambda val : round(val,1)
rounded_df2p= df1.apply(r2dp)



'''함수화 코드정리용 '''
import networkx as nx
import matplotlib.pyplot as plt

text ='웁살라 대학교에서는 언어강좌를 우선 듣게 되지만 더욱 중요한 것은 제 전공인 사회학과 경제학입니다. 언어는 그 사회와 뗄 수 없는 관계를 맺고 사회의 경제 정치 역사적 상황에 따라 변하게 되기 때문입니다. 따라서 사회의 변화와 발전에 대한 강좌를 수강할 계획입니다. 여름에 Basic Swedish 강좌를 들으며 스웨덴어의 문법 통사 구조와 발음들을 집중적으로 공부할 예정입니다. 강좌 과정에서 다양한 교환학생들과 영어로 대화를 하며 그들의 사회 문화적 분위기와 언어에 대해 대화를 구체적으로 나눠볼 생각입니다. 한국에서 배운 독일어와 스웨덴어의 동사변화나 발음구조의 유사점과 차이점이 무엇이고, 왜 그런지에 대해서도 개별적으로 공부해볼 생각입니다.'
def read_file(path):
    with open(path) as fin:
        return fin.read()

def construct_wordnet(text):
    lines = text.split('\n')      # 줄 단위로 자른다

    word_edges = {}

    for line in lines:
        _line = line.strip()
        if not _line:             # 빈줄이면 건너뛴다
            continue
        statements = _line.split('.') # 문장 단위로 자른다
        for statement in statements: # 빈 문장이면 건너뛴다
            if not statement:
                continue
            words = statement.split(' ') # 단어 단위로 자른다
            cleansed_words = [w.replace('.', '').replace(',', '').strip() for w in words] # 단어에서 구두점이나 공백을 없앤다
            cleansed_words_2 = [w for w in cleansed_words if len(w) > 1] # 구두점 및 공백 제거로 인해 빈 문자열이 되어버린 원소, 그리고 한글자 단어를 제거한다
            num_words = len(cleansed_words_2)
            for index_i in range(num_words): # 한 문장에 등장한 단어들을 서로 연결한다
                word_i = cleansed_words_2[index_i]
                for index_j in range(index_i+1, num_words):
                    word_j = cleansed_words_2[index_j]
                    word_to_word = (word_i, word_j)
                    word_to_word = tuple(sorted(word_to_word))
                    word_edges[word_to_word] = word_edges.setdefault(word_to_word, 0) + 1
    return word_edges

def remove_low_frequency(word_edges, cutoff=2):
    # 등장 빈도가 1회인 edge는 제거한다
    keys = list(word_edges.keys())
    for key in keys:
        if word_edges[key] < cutoff:
            del word_edges[key]
    return

def draw_graph(word_edges):
    G = nx.Graph()
    for (word_1, word_2), freq in word_edges.items():
        G.add_edge(word_1, word_2, weight=freq)

    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(12, 12))    # 결과 이미지 크기를 크게 지정 (12inch * 12inch)
    widths = [G[node1][node2]['weight'] for node1, node2 in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.1)
    nx.draw_networkx_labels(G, pos, font_family='Noto Sans CJK KR') # 각자 시스템에 따라 적절한 폰트 이름으로 변경
    return


'''결과물 공유해야'''
text = read_file('assets/moon_speech.txt')
wordnet = construct_wordnet(text)
remove_low_frequency(wordnet)
draw_graph(wordnet)
plt.show()

'''클래스

함수 여러개를 묶어놓은 것 
클래스에 소속된 함수는 메소드라고 부름 
공용 변수들 존재 
'''
from datetime import datetime
from random import randint
from random import random

class Person:
   def __init__(self, name, weight=3.0, height=0.20):

       self.name = name
      self.weight = weight
      self.height = height

   def eat(self):
      self.weight = self.weight + round(random(), 2) * 10

   @property
   def bmi(self):
      return round(self.weight / pow(self.height, 2), 1)

p1 = Person('Tom', weight=75.0, height=1.83)
print(p1.bmi)
p1.eat()
p1.eat()
print(p1.bmi)
'''
객체한테는 p1. 이런식으로 메소드 진행 '''
class Person:
    pass
p=Person()
p  #클래스가 객체화된다고 표현 
'''
()에서 무슨일이 일어나는지 _init_함수로 정의
'''

g = nx.Graph()
path = 'assets/moon_speech_euckr.txt'
    print (path)
    n = 7
# -*- coding: utf-8 -*-
def count_words(path):
    
    # 파일로부터 내용을 읽어들인다
    fin=open(path)
    text= fin.read()
    text
    fin.close()
    # 한 행씩 하나의 리스트가 되도록 나눈다(split)
    lines = text.split('\n')
    lines
    # 단어별 빈도를 저장할 빈 dict를 만든다.
    word_freq_dict = dict()
    # 한 행씩 순회하면서 수행한다 (for)
    for line in lines:
        #  해당 행에서 부호(, . ! 등)를 없앤다 (replace)
        striped_line = line.replace(',','').replace('.','').replace('!','')
        print(striped_line)
    
    #   빈 칸을 기준으로 어절 단위로 분리하여 리스트로 만든다 (split)
        words = striped_line.split()
        print(words)
    #   단어 리스트를 순회하면서 수행한다 (for)
        for word in words:
    #     먄악 단어의 길이가 2보다 작다면 해당 순회를 건너뛴다 (if, continue)
            if len(word)<2:
                continue
            print(word)
            #     단어 빈도 dict에, 해당 단어의 빈도를 하나 증가시킨다
            word_freq_dict[word]=word_freq_dict.get(word,0)
            print(word_freq_dict)
    # 빈도순으로 내림차순 정렬하고, 상위 7개를 잘라서 반환한다 (sorted, list slicing)
    '''sorted([5,3,2,4,1])
    sorted([5,3,2,4,1],reverse=True)
    sorted([('빈도순',5),('내림차순',3),('오름차순',4)], key=lambda x: x[1])'''
    
    '''lambda x : x[1]
    -> 
    def sort(x):
        return x[1] (두번째원소반환)'''
    sorted(word_freq_dict.items(),key= lambda x: x[1], reverse=True)
    
count_words(path)      
        