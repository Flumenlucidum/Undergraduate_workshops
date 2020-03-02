# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 14:45:08 2018

@author: main
"""

for i in range(10):
   print(i, end='')
   print(',', end=' ')
   
   
import networkx as nx
import random

G = nx.Graph()
for i in range(49):
    G.add_node(i)

for i in range(49):
    G.add_edge(i, random.randint(0,49))


pos = nx.spring_layout(G)
nx.draw_networkx_edges(G, pos, alpha=0.2);
nx.draw_networkx_nodes(G, pos, node_size=50);

import networkx as nx

G = nx.Graph()
G.add_node(1)
G.add_node(2)
G.add_edge(1, 3)

pos = nx.spring_layout(G)
nx.draw_networkx_edges(G, pos, alpha=0.2);
nx.draw_networkx_nodes(G, pos, node_size=50);

#외부 라이브러리 사용시 
# networkx 라이브러리  
#pos -> 그리는 방법 중 하나 spring 레이아웃 기법 
#G pos 정보 사용   알파- 색상값 RGB  투명도 값의 정도 
 
 #해결
import networkx as nx
import random
 
#for aVariable in aCandidate:
    #statements
G = nx.Graph()

#50만큼 반복하면서 노드를 추가   #psudo code #range 50 ->50 개
for node_id in range(50):
    G.add_node(node_id)

for i in range(100):
    node_a = random.randint(0,50)
    node_b = random.randint(0,50)
    G.add_edge(node_a, node_b)


pos = nx.spring_layout(G)
nx.draw_networkx_edges(G, pos, alpha=0.2);
nx.draw_networkx_nodes(G, pos, node_size=50);

#raw 참고  add_path

#format(number)-- 

#기초 2 
#정수 int   실수 float 서로 바꾸기  str 문자 
 #tuple - 메모리를 더 효율적으로 하길 원할 때   변경불가
 
 #위치 참조시 0부터 시작 
a_list = [1, 2, 3, 4, 5]
a_list[3]
a_list[-2] 
a_list[2:4]
#[:2] 이 경우 끝의 첨자는 포함되지 않음  
a_list[2]
#컨테이너 함수 1 in a_list
 #리스트 더하기 뒤로 이어붙여짐 
 #리스트의 리스트    
a_list = [['Tom', 'M', 15, True], 
          ['Chris', 'F', 28, True], 
          ['Timothy', 'M', 32, False]]

for row in a_list:
    if row[2] > 20:
        print(row)
        