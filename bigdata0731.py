#!/usr/bin/env python
# coding: utf-8

# In[1]:


a=30


# In[2]:


whos


# In[3]:


# 머신러닝과 딥러닝  
#선형대수 left pseudo inverse matrix 이용 
#기계학습모형 --훈련시켜서 정답 예측하게 y=wx +b 의 w와 b를 갖고있는것
#어떤 과정을 예측할 때 필요한 데이터를 선별하는 과정 -기계학습
#이런 과정 안거칠때 딥러닝  tensorflow 라이브러리 
#인공신경망 알고리즘 -계속 w b 바꿔가면서 시도 그 차이 (절댓값)
#딥러닝- 영상이나 자연어 처리면 사용 그러나 데이터분석은 머신러닝 


# In[4]:


del a


# In[5]:


whos


# In[6]:


for i in range(1,11):
    print('{}번째'.format(i))


# In[7]:


# 중첩루프   반복문 안에 새로운 반복문-- 2차원 구조 
#for row in rows:
#    for data in row: 
        


# In[8]:


list_2d=[[1,2,3],[4,5,6],[7,8,9]]


# In[9]:


for row in list_2d:
    for data in row:
        print(data, end= ' ')


# In[13]:


for row in list_2d:
    for data in row:
        print(data, end= ' ')
    print()   #바깥 반복문의 시행을 구분하기 위함


# In[14]:


#3차원 리스트 인덱싱 
#리스트 안에 2차원 리스트 두개  [면][행][열] 면-> 몇번째 2차원리스트인지(깊이)
list_2d[1][2]


# In[15]:


for i in range(2,10):
    for j in range(1,10):   #컴퓨터는 프린팅할때 항상 행 우선으로
        print(i*j, end=' ')
    print()


# In[16]:


#작업 진행상황 -progress bar   print(..... end='\r')


# In[18]:


import time 
progress=' '
percent=0 
num=100
for i in range(1,num+1):
    percent =(i/num)*100
    progress = int(percent*5/10)*'*'
    time.sleep(1)
    print("{:5.1f}%[{:<50s}] {}".format(percent, progress,i))


# In[21]:


# 함수->return  
def  add(a,b):   #a, b-> 매개변수 parameter variable 
    '''
    두 수를 더하는 함수입니다 
    '''   # 함수의 설명-- docstring 
    print(a+b)


# In[22]:


add(34,56) #함수를 call 한다 -> argument value 


# In[23]:


add.__doc__


# In[26]:


def fibo(n):
    a = 0
    b=1
    while a<n:
        print(a, end=" ")
        a,b=b,a+b
    print()


# In[27]:


fibo(40)


# In[28]:


def func_a():
    num=10  #지역변수 -함수안에서만 쓸수있는 변수  함수 안인지 밖인지 중요 
    print(num)


# In[29]:


func_a(1)


# In[31]:


func_a()


# In[32]:


whos


# In[33]:


del num, percent, progress, data, add


# In[35]:


def cube(x):
    print(x**3)


# In[36]:


cube(4)


# In[37]:


del cube


# In[38]:


#함수안에서 전역변수를 사용하고 싶다면 global a 이런식으로 
a


# In[39]:


def mul(a,b):
    return a*b


# In[40]:


result=mul(50,39)


# In[41]:


result


# In[42]:


def swap(a,b):
    return b,a


# In[43]:


a=10
b=20


# In[44]:


c=swap(a,b)


# In[45]:


x,y= swap(a,b)


# In[46]:


x


# In[47]:


y


# In[48]:


def add(a,b,c):
    return a+b+c 


# In[49]:


add(4,5,6)


# In[55]:


#파이썬에서 이름이 같은 함수 여러개 정의-- 중복 정의 - 파이썬 함수 중복정의 불가능  자바는 가능 
def add(a,b,*c, d='add', **e):   #d 처럼 기본값가지는 매개변수--- 선택 매개변수  **e-> 딕셔너리 매개변수   순서변하면 안돼 
    print(a,b,c,d)
    sum=a+b
    for data in c:
        sum =data +sum
    return sum


# In[52]:


add(3,4)


# In[53]:


add(4,5,6)


# In[54]:


add(1,2,3,4)


# In[56]:


#매개변수에 *붙이면  Tuple parameter 라고 부름 
# a,b는 필수 매개변수 
add(2,3)


# In[57]:


add(4,5,47,d='plus')


# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt   #matplot library 안의 pyplot모듈 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


iris=sns.load_dataset("iris")


# In[3]:


#iris.head()


# In[4]:


grid=sns.FacetGrid(iris)


# In[6]:



iris.head()


# In[89]:


sns.scatterplot(iris.petal_length, iris.petal_width)


# In[70]:


import seaborn as sns


# In[71]:


pip install seaborn


# In[72]:


get_ipython().system('python -m pip install seaborn')


# In[8]:


#딕셔너리 매개변수   f= ' '   g= ' ' 이런식   


# In[12]:


sns.heatmap(iris.corr(), vmax=1.0, vmin=-1.0)


# In[13]:


plt.hist(iris.petal_length)   # histogram 단변량


# In[14]:


sns.scatterplot(iris.petal_length,iris.petal_width)


# In[18]:


grid=sns.FacetGrid(iris, col='species')
grid.map(plt.hist,'sepal_length')
grid.set_axis_labels('Sepal Length', "Count")


# In[19]:


#FacetGrid.map(func, *args, **kwargs)   map 함수는 그 영역 호출해서    args의 그 데이터 종류 인수로 func에 전달 
#변량개수 아직 몰라 *붙인 이유--- 전부다 가져가서  
#scatter plot등의 그래프 그래프 어떻게 보여질지 옵션 붙일 수 있어   **kwargs에 들어가 
#함수에 인수를 전달하는 것이 바로 튜플인수와 딕셔너리 인수 


# In[20]:


grid=sns.FacetGrid(iris, col='species')


# In[23]:


def add(*data, msg="더하기"):
    sum=0
    for item in data:
        sum= sum+item
    print(msg)
    return sum


# In[24]:


add(3,4,45,6,7)


# In[28]:


def op_(func,*args,**kwargs):
    print('추가작업')
    return func(*args,**kwargs)


# In[29]:


op_(add,1,2,3,4,5, msg='덧셈')  # 풀어서 입력해줘야   인수 언패킹 


# In[30]:


#람다식 
#함수와 생김새는 다르지만 똑같은 역할을 하는 식 


# In[31]:


def add (a,b):
    return a+b


# In[32]:


add=lambda a,b : a+b  #자동으로 리턴 


# In[33]:


add(4,5)


# In[34]:


data=[1,2,3,4,5,6,7]


# In[35]:


# filter(함수, data )-> 데이터에서 함수하나하나 뽑아서 트루인 값만 출력
#명령모드 ctrl g   ctrl a   ctrl b


# In[36]:


list(filter(lambda x: x%2==0,data))


# In[37]:


filter(lambda x: x%2==0,data)


# In[38]:


#데이터 프레임  
#데이터탐색 eda 최적화-머신러닝 


# In[8]:


import pandas as pd


# In[40]:


#판다스에는 DataFrame클래스(엑셀의 sheet 한장)와 Series클래스- 시계열데이터(엑셀의 '열' 하나)
#엑셀로 안되는경우는 데이터가 너무 많거나 학습을 해야하는 경우 
#axis=0 행 axis=1 열  


# In[9]:


import seaborn as sns


# In[10]:


iris=sns.load_dataset('iris')


# In[43]:


type(iris)


# In[45]:


iris.head()


# In[47]:


d={'col':[1,2,3],'col2':[3,4,5]}


# In[48]:


pd.DataFrame(data=d)


# In[49]:


d1=[{'col1':1,'col2':5},{'col1':3,'col2':8}]


# In[50]:


pd.DataFrame(d1)


# In[51]:


#NaN자동으로 할당  데이터프레임은 열단위로 같은 타입이어야 -- 여러타입이 섞이면 가장 높은? 정수-> 실수  더 자세히 


# In[52]:


a=[1,2,3,4,5]
b=[6,7,8,9,10]


# In[53]:


pd.DataFrame({'col1':a, 'col2':b})


# In[56]:


pd.DataFrame([a,b],columns=["col1","col2"])


# In[57]:


pd.DataFrame([a,b],columns=["col1","col2",'c', 'd','e'])  


# In[58]:


import numpy as np


# In[5]:


pd.DataFrame(np.c_[a,b],columns=["col1","col2"])    #np.c 열단위로 합쳐주는 객체


# In[6]:


member_df=pd.read_csv('member_data.csv')   #read_excel하면 읽을수있어


# In[65]:


member_df


# In[66]:


pd.read_csv("member_data.csv", comment='#')# 주석행 문자 입력 --이 행은 안읽어


# In[67]:


# 딕셔너리를 데이터프레임으로   리스트를 데이터프레임으로 --그냥하면 행이 되기 때문에-  numpy이용해서 np.c열로 합치기
#파일로부터 불러서 데이터 프레임 
#scikit-learn skilearn
#패키지 모듈 함수 
from sklearn import datasets 


# In[4]:


iris=datasets.load_iris()


# In[69]:


type(iris)


# In[70]:


print(iris)


# In[71]:


iris.data


# In[72]:


iris.target  


# In[73]:


x= pd.DataFrame(iris.data, columns=iris.feature_names)


# In[74]:


x.head


# In[75]:


x.head()


# In[76]:


y=pd.DataFrame(iris.target_names[iris.target], columns=['species'])


# In[77]:


y.head()


# In[82]:


iris_df=pd.concat([x,y],axis=1)  #열기준 합쳐-- 행의 개수가 같아야함 


# In[83]:


iris_df.columns=["sepal_length","sepal_width", 'petal_length','petal_width','species']  #행이름 바꾸기


# In[84]:


iris_df


# In[85]:


iris_df.species


# In[86]:


member_df=pd.read_csv("member_data.csv")


# In[87]:


member_df.index=['a','b','c','d']   #columns- 열바꾸기 index -행바꾸기 


# In[88]:


member_df


# In[91]:


member_df.index=[['LR','LR','TD','TD'],['e','w','s','n']]


# In[92]:


member_df


# In[3]:


#부분데이터 조회하는 방법  
import pandas as pd
member_df =pd.read_csv('member_data.csv')


# In[4]:


member_df.columns=["a",'b','c','d']


# In[5]:


member_df['c']


# In[10]:


member_df.loc[0:3,'a']    #iloc 순서로 찾을때와(3빼) loc-이름으로 찾을때 [행,열]->0-3-- 3까지 다들어가


# In[11]:


member_df.loc[0:3]


# In[12]:


member_df.loc[0:3,'b':'d']


# In[13]:


member_df.loc[[0,1,2],['b','d']]


# In[14]:


iris_df.loc[:,setosa]


# In[15]:


member_df.iloc[0]   #얘도 행으로 


# In[16]:


member_df.iloc[0:3,0:3]  


# In[17]:


member_df.iloc[:,0:3]  


# In[18]:


member_df.iloc[::-1,0:3] 


# In[19]:


import statsmodels.api as sm


# In[20]:


# pandas- datafram  numpy--n 차원 배열 
#sklearn statsmodel-> machine learning
#seaborn matplotlib-> visualization 


# In[21]:


import seaborn as sns
iris=sns.load_dataset('iris')


# In[22]:


iris.head()


# In[23]:


#독립변수 열 만 추출하세요 x = iris.


# In[28]:


x=iris.iloc[:,0:4]


# In[29]:


x


# In[30]:


x=iris.iloc[:,:-1]


# In[31]:


x


# In[32]:


y=iris.iloc[:,-1] #iris.species


# In[33]:


y


# In[35]:


#조건은 loc를 이용  행기준
iris.loc[iris.species=='versicolor'].head(10)


# In[36]:



iris.loc[iris["species"]=='versicolor',['petal_width','petal_length']].head(10)


# In[42]:


iris.loc[(iris['species']=='setosa')&(iris.petal_length.astype(float)>1.6)].head(10)   #astype -특정 데이터 형식으로 바꿔 


# In[43]:


iris.head()


# In[44]:


#drop을 통해 행이 열 지우기   인덱스가 아닌 이름으로 지우는 것  원본 바꾸는게 아니고 삭제한 데이터 프레임을 반환 
iris.drop(0, inplace= True)   #inplace-매개변수   현재 데이터를 변경 시켜 


# In[45]:


iris.head()


# In[46]:


iris.drop(1, axis=1)  #axis=1-- 열에서 찾는것


# In[47]:


iris.drop("species", axis=1)


# In[48]:


iris.drop(labels=['sepal_length','sepal_width'],axis='columns')   #여러 행이나 열 제거할 때 라벨로 


# In[49]:


member_df


# In[50]:


member_df.sort()


# In[51]:


# Title
       *contents
       


# In[53]:


member_df.sort_index()


# In[55]:


member_df.sort_index(axis=1)


# In[56]:


member_df.sort_values(by='b') #열 지정해서 행들을 바꿔줘


# In[58]:


member_df.sort_values(by='d', ascending=False)


# In[61]:


iris.describe()   #기초통계량 


# In[62]:


iris.describe(include='all')  #숫자타입외의 변수들에 대한 설명도 포함 


# In[63]:


member_df.describe(include='all')


# In[65]:


iris.min()   # 각 열 별통계량


# In[66]:


iris.std()


# In[68]:


iris.var()


# In[69]:


iris.corr()


# In[70]:


iris.max()


# In[71]:


iris.std()


# In[72]:


iris.mean()


# In[80]:


iris.loc[iris.species=='setosa'].mean()


# In[11]:


#groupby 
iris_grouped=iris.groupby(iris.species)


# In[12]:


iris_grouped.std()


# In[14]:


for type, group in iris_grouped:
    print(type)
    print(group.head(2))


# In[15]:


member_df


# In[17]:


member_df=pd.read_csv("member_data.csv")


# In[18]:


member_df


# In[19]:


wine=pd.read_csv("http://javaspecialist.co.kr/pds/297",sep=";")


# In[21]:


wine.head()


# In[23]:


#각 등급별 개수는 어떻게 되는가  # 5등급인 와인의 평균  알코올 함량은 어떻게 되는가 
#독립변수를 종속변수와 분리 독립변수 x   y (등급 )
wine.columns


# In[24]:


wine.describe(include='all')


# In[25]:


wine_grouped=wine.groupby(wine.quality)


# In[26]:


wine_grouped.mean()


# In[27]:


#5등급 와인의 알콜합량 평균은 9.899706


# In[28]:


wine_grouped.describe()


# In[29]:


# 위 개수란에 있어 


# In[30]:


wine.columns


# In[31]:


wine[:,:-1].head(10)


# In[32]:


x=wine[:,:-1]


# In[33]:


x=wine.iloc[:,:-1]


# In[34]:


x


# In[35]:


y=wine.iloc[:,-1]


# In[37]:


y.head(10
    )


# In[41]:


wine_grouped.count()


# In[42]:


med=pd.read_excel('신약효과 테스트.xls')


# In[43]:


med.describe(include='all')


# In[44]:


wine.groupby(by='quality').count()


# In[45]:


wine.loc[wine.quality==5].alcohol.mean()


# In[46]:


x=wine.iloc[:,:-1]


# In[47]:


y=wine.quality


# In[48]:


#와이드 포맷    -- 기계학습에서 이런식-- 열단위 데이터 구조  열단위

#롱포맷 -언피봇테이블 -- 기본 가공안된 데이터들 이런식 행단위 --기록들 


# In[49]:


#melt()라는 함수를 통해 언피벗팅
#실제로 많이 하는것은 pivot table 을 통한 피벗팅  index columns values    pivot_table()


# In[50]:


#dataframe -만들기- 보통 불러와
# iloc loc 로 원하는것 뽑기 
# 기초통계 describe group
# apply  -반복문 없이도 같은 효과  dataframe전체에 어떤 효과를 주고 싶을때 


# In[51]:


import statsmodels.api as sm


# In[52]:


iris=sns.load_dataset('iris')


# In[53]:


#iris 데이터의 모든 값을 반올림 하세요.
round(4.6)


# In[54]:


iris_x=iris.iloc[:,:-1]


# In[55]:


iris2=[]
for row in iris_x.iterrows():
        for data in row:
            print(data)


# In[56]:


import numpy as np


# In[58]:


iris_x.apply(np.round)


# In[59]:


#각변수별 평균과 데이터의 차이를 갖는 데이터 프레임을 만들기 


# In[60]:


iris_x.mean()


# In[61]:


iris_x_avg=iris_x.mean()


# In[62]:


iris_x.apply(lambda x: x-iris_x_avg, axis=1 )


# In[64]:


iris_x.apply(lambda x: x**2,axis=1)


# In[66]:


iris_x.apply(np.sum, axis=1)   #집계에는 달라 열을 합계(행단위로 합계함) (axis=1 )


# In[67]:


import numpy as np
data=np.arange(12).reshape(3,4)
data


# In[69]:


data[1][2]


# In[70]:


whos


# In[71]:


np.sum(data)


# In[72]:


np.sum(data, axis=0)


# In[73]:


np.sum(data,axis=1)


# In[74]:


iris_x.applymap(np.sum)  #각 요소별로 작동 


# In[75]:


iris=sns.load_dataset('iris')


# In[78]:


sl=iris.sepal_length


# In[79]:


sl


# In[80]:


#wine 데이터  5, 6 데이터 모든 독립변수값 반올림한 데이터 생성 


# In[83]:


wine_qual_56=wine.loc[wine.quality=5&wine.quality=6]


# In[84]:


x


# In[88]:


x.apply(np.round,if wine.quality=5 & wine.quality=6, inplace=True)


# In[89]:


wine56=wine.loc[(wine.quality==5)|(wine.quality==6)]


# In[90]:


wine56_x=wine56.iloc[:,:-1]


# In[91]:


result=wine56_x.applymap(round)   #applymap은 각요소별로 


# In[92]:


result.head()


# In[ ]:




