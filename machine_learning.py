#!/usr/bin/env python
# coding: utf-8

# In[2]:


#수학적 공식이 안된다면 최적화에서 기계학습이 필요 
#ML studies algorithms that improve with experience
#경험을 통해 performance measure 높여 task 달성 
#지도 학습 과 비지도학습 -> 종속변수가 있으면 지도학습 
#종속변수가 없는것은 군집 나누기, 연관분석  
#지도학습의 두가지 ---1)회귀 데이터 평균수렴경향, 다음 분기 예측  2) 분류  클래스로 구분하기 
#지도학습에서 얻는 것 -- 잘 분류하기(필기체 인식)  회귀는 예측,  군집분석 
#MSE-mean squared error - 분산   
#분류군집회귀에 따라서 알고리즘 달라    분석대상과 분석 방식


# In[3]:


#Scikit learn -기계학습관련 함수들 // statsmodels- 통계 관련 함수들 


# In[5]:


#데이터 분석 - 머신러닝 EDA (탐색적 데이터 분석), 딥러닝-인공신경망 
#일반적 문제해결은 머신러닝   EDA- insight발견   딥러닝- 모든 문제 해결은 할 수 있어 
# ML   DL 
#머신러닝--  특징을 찾는 과정이 있다 -> 변수들 중에서 종속변수에 영향을 주는 독립변수 찾는 과정 이런게 있으면 딥러닝 
#딥러닝은 모든 변수를 다 넣어    분산=0인 열들은 머신러닝 독립변수에서 빼야돼 
#영상처리  영상 화면 화소의 수만큼 변수 존재   딥러닝 (영상처리 신호처리(사진보고 개구리인지 인식), 자연어처리(다음 문장))
1024*798


# In[6]:


#전처리 과정  기계학습을 위해선 숫자여야하는데 문자이거나,  결측치가 있을때
#data frame 여기서 변수를 선택   로그성데이터도 와이드 포맷으로 
#파생변수 추가 
#모형 선택 #모형 생성 -> 수식에 x 값 집어넣어서 y예측함 -> 평가수치 높이게 전처리 다시해봐 


# In[7]:


#알고리즘  지도-decision tree, gaussian process  ensemble method(가중치 따른 배깅, 부스팅)/ 비지도
#전처리 -표준화   여러변수가 있을 때 -- 나이 변수와 연봉변수는 수의 크기 달라--- 표준화 필요 
#표준화와 정규화는 달라 표준화는 0-100  0-1  50->0.5   정규화는 0.5아니고 그 분포따라 달라져 


# In[8]:


import seaborn as sns


# In[9]:


iris=sns.load_dataset('iris')


# In[10]:


x=iris.iloc[:,:-1]


# In[11]:


x.head()


# In[12]:


#표준화 4가지 방법 주로 쓰여   
from sklearn.preprocessing import scale 


# In[13]:


x_scaled=scale(x)


# In[15]:


x_scaled[:5,:]  #평균 0 표준편차 1 


# In[17]:


x_scaled.mean(axis=0)   #0에 가까운 값 


# In[18]:


from sklearn.preprocessing import robust_scale


# In[20]:


iris_rs=robust_scale(x)


# In[23]:


iris_rs[:4,:]


# In[24]:


from sklearn.preprocessing import minmax_scale   #가장 많이 사용돼  min ->0 max->1로 
iris_mm=minmax_scale(x)


# In[25]:


iris_mm[:5,:]


# In[26]:


#데이터가 음수값을 가지면 minmax 보다도 maxabs많이 사용 


# In[27]:


from sklearn.preprocessing import maxabs_scale  #절댓값이 가장 큰걸 1로 mapping  음수면 그 1에 - 붙여 


# In[28]:


iris_ma=maxabs_scale(x)


# In[29]:


iris_ma[:6,:]


# In[30]:


#정규화 Lasso 정규화, Lidge 정규화  정규분포 따르게 만드는 것  -> 정규분포라고 학습이 잘된다는 보장이 없어 


# In[31]:


class Test:
    pass   #데이터 저장하거나 기능을 수행하기 위함 


# In[32]:


class Test:
    def __init__(self, data): #생성자
        self.data=data
    def do_it(self):               #method(클래스 내부에 있는 함수를 지칭하는 말 )
        print(self.data)


# In[33]:


t1=Test('Hello')


# In[35]:


t1.do_it()


# In[36]:


#데이터를 저장하고 기능을 수행하게 해  표준화한것을 원본으로   클래스를 통한 작업은 원래로 돌리는
#inverse_transform기능


# In[37]:


from sklearn.preprocessing import StandardScaler


# In[38]:


sc=StandardScaler()


# In[42]:


sc.fit(x)   #fit- 학습때의 함수
iris_scaled=sc.transform(x)
iris_scaled[:5,:]


# In[43]:


iris_x=sc.inverse_transform(iris_scaled)


# In[44]:


iris_x[:5,:]


# In[45]:


x.head()


# In[47]:


class hero: 
    time=20
    num=0
    def __init__(self,first,age,home):
        self.first=first  
        self.age=age    
        self.home=home    
        self.email=first+'@gmail.com'
        
       
    
    def howold(self):
        return '{} is {} years old'.format(self.first,self.age)
    
    def ageafter(self):
        self.age=int(self.age+self.time)
Thor=hero('Thor',1500,'Asgard')
BW=hero('Nat',30,'Russia')
cap=hero('Steve',100,'Brooklyn')
print(Thor.email)
print(BW.email)
print(cap.email)

print('{} is {} years old'.format(Thor.first,Thor.age))


# In[50]:


cap.howold()


# In[51]:


#stt  음성인식 (문장 단어인식  )-- 지도학습 


# In[52]:


#전처리- 레이블 인코딩 --텍스트를 결국 숫자로 바꿔야돼 데이터는 모두 숫자여야  음성도 디지털 인코딩 필요  영상도 화소 0~255
#수바꾸기 2방법
iris.species


# In[53]:


from sklearn.preprocessing import LabelEncoder


# In[54]:


le=LabelEncoder()
le.fit(iris.species) # 학습시킨다는 뜻 


# In[55]:


species=le.transform(iris.species)


# In[56]:


species


# In[57]:


#0,1,2 따라 가중 달라질 우려 그래서  원-핫 인코딩 사용    클래스의 수 만큼 변수가 만들어져 


# In[60]:


from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
enc.fit(species.reshape(-1,1))  #행 -1데이터 맞게알아서 해주세요 열 1 한개로 해주세요   이경우 변수 세 개인셈 


# In[61]:


iris_onehot=enc.transform(species.reshape(-1,1))


# In[62]:


iris_onehot.toarray()   #이렇게 만들어진 행렬을 희소행렬이라고도 지칭함  안하는 경우도 ,, 딥러닝에서는 대부분 원핫으로 


# In[63]:


#전처리 -결측치처리


# In[64]:


import random 
for col in range(4):
    x.iloc[[random.sample(150,30),col]=float('NaN')           #랜덤하게 0-149 중 30개를 뽑아내 (행)


# In[65]:


x=iris.iloc[:,:-1]


# In[66]:


x.iloc[[4,7,15,26,30],0]=float('NaN')
x.iloc[[5,9,10,23,40],1]=float('NaN')
x.iloc[[4,11,15,26,50],2]=float('NaN')
x.iloc[[3,13,14,40,80],3]=float('NaN')


# In[67]:


x.head()


# In[68]:


from sklearn.preprocessing import Imputer   #imputer는 전처리하는 클래스 


# In[69]:


imp_mean=Imputer(axis=0)  #행기준으로


# In[71]:


x_mean=imp_mean.fit_transform(x)


# In[76]:


x.mean(axis=0)


# In[74]:


x_mean[:5,:]


# In[77]:


#python으로 머신러닝하는 것은 전처리 등 표준화 인코딩 굉장히 쉬운 편 
#평균 중위수 디테일 보다 더 자세히 처리하는것은 그 데이터의 다른 성질을 고려하여 채우기 


# In[78]:


#imp_median=Imputer(strategy='median', axis=0)   이런식으로 하면 중위값으로 대체 


# In[79]:


x.median(axis=0)


# In[80]:


imp_median=Imputer(strategy='median', axis=0)


# In[81]:


imp_median.fit_transform(x)[:5,:]


# In[82]:


#최빈값은 strategy='most_frequnet'
x.mode(axis=0)


# In[83]:


#변수 선택   -> 분류모형  만들기 rf나 디시젼 트리   변수들의 영향도 알아내기 
import pandas as pd 
redwine=pd.read_csv('http://javaspecialist.co.kr/pds/297',delimiter=';')


# In[86]:


redwine.tail()


# In[85]:


redwine.describe()


# In[87]:


#와인 등급에 영향을 많이 주는 변수가 무엇인지 찾아보자 
x=redwine.iloc[:,:-1]
y=redwine.iloc[:,-1]


# In[88]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
scaler.fit(x)


# In[89]:


x_scaled=scaler.transform(x)


# In[90]:


x_scaled


# In[91]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=10,random_state=0)  #random forest classifier (분류  )  random forest regressor (회귀)
rf.fit(x_scaled,y)


# In[94]:


#모형을 위해 rfc 알고리즘 사용 fit으로 학습 -- rf객체는 수식을 가지고 있어 
rf.feature_importances_   #각 변수별로 열별로의 중요도 종속변수에 영향을 크게 주는가 


# In[95]:


import numpy as np
imp_df=pd.DataFrame(data=np.c_[x.columns.values,rf.feature_importances_],columns=['name','importance'])


# In[97]:


imp_df.sort_values('importance',ascending=False, inplace=True)


# In[98]:


imp_df


# In[99]:


imp_df.iloc[:5,:]


# In[100]:


#변수 하나가 빠지면 나머지 변수들의 중요도 순서가 달라진다는 것을 인식해야 
#꼴찌인 변수 삭제해가는 과정해서 5개 남기기가 더 좋을 수도 있어 
#독립변수들끼리 관련성  다중공선성 있다고 이야기해  
#VIF 분산 팽창계수 이를 통해 다중공선성을 판단  변수별로 10이넘으면 다중공선성 있다 생각 
#제일 큰 애 지우고 그다음 가장큰애 지우고... 10보다 큰 애 없을 때까지 


# In[103]:


from sklearn.feature_selection import RFE #Recursive Feature Elimination
select=RFE(rf, n_features_to_select=5)
select.fit(x_scaled,y)


# In[104]:


select.get_support()


# In[105]:


x.head()


# In[112]:


rfe_df=pd.DataFrame(data=np.c_[x.columns.values,select.get_support()],columns=['variables','importance'])


# In[114]:


rfe_df.sort_values('importance', ascending=False, inplace=True)


# In[115]:


rfe_df


# In[116]:


#이제 회귀모형에서 importance  -> .coef_ 회귀계수    분류모형은 RFE   
#파생변수 추가-- 가설따라서 넣어보거나 빼보기 


# In[117]:


#모형 생성과 평가 
#평가-train data set   test data set 나눠서 훈련 모형 후 테스트-- y 와 y' 차이  
#so we need to divide data into sets  #모형만들기전


# In[120]:


from sklearn.model_selection import train_test_split


# In[122]:


train_x, test_x, train_y,test_y =(train_test_split(x,y,test_size=0.3))


# In[123]:


train_x


# In[124]:


#이제 모형을 만들어 학습시키자 
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(50,30))
mlp.fit(train_x,train_y)


# In[126]:


#이제 mlp가 모형을 가지고 있다 
pred=mlp.predict(test_x)


# In[127]:


test_y


# In[128]:


pred


# In[129]:


#모형 평가 -> 분류 회귀인지 따라 다름 
#분류모형 -맞나 안 맞나 --accuracy 
#회귀모형- MSE-mean squared error  RMSE(MSE의 근호 )


# In[130]:


#예측한것중 진짜의 비율 precision  진짜중 예측해서 맞춘 것 비율 Recall  -> 조화평균 -> F score, F measure 


# In[131]:


#모델들은 모두 score 함수 있어서 평가 가능   
#분류 모형: 정확도 ,f1 score(f measure)(precision과 recall의 비율 )
#회귀모형 : RMSE or MSE, R2(결정계수)


# In[132]:


#  hjk7902@gmail.com


# In[135]:


import pandas as pd
import numpy as np
redwine=pd.read_csv('http://javaspecialist.co.kr/pds/297',delimiter=';')


# In[137]:


redwine


# In[138]:


train=redwine.sample(frac=0.7)
test=redwine.loc[~redwine.index.isin(train.index)]


# In[141]:


train.head()
print(train.shape, test.shape)
train_x=train.iloc[:,:-1]
train_y=train.iloc[:,-1]
test_x=test.iloc[:,:-1]
test_y=test.iloc[:,-1]


# In[142]:


from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(50,30))
mlp.fit(train_x,train_y)


# In[143]:


print("Training score: %s" %mlp.score(train_x,train_y))


# In[144]:


pred=mlp.predict(test_x)


# In[145]:


confusion_matrix=pd.crosstab(test_y, pred, rownames=['Ture'],colnames=['Predicted'],margins=True)


# In[147]:


type(confusion_matrix)


# In[148]:


print(confusion_matrix)


# In[149]:


confusion_matrix.iloc[2:5,0:3]


# In[150]:


cm= confusion_matrix.as_matrix()


# In[151]:


cm


# In[152]:


cm_row=confusion_matrix.shape[0]
cm_col=confusion_matrix.shape[1]
print(cm_row,cm_col)


# In[153]:


accuracy= (cm[2][0]+cm[3][1]+cm[4][2])/float(cm[cm_row-1][cm_col-1])


# In[154]:


print(accuracy)


# In[155]:


#선형대수 SVD -null 값을 채울때 특이값  잡음제거 등 
#np.sum(axis=0,1,2 ) => 2,3,4 일때 axis=0이면 열두개 나와   1 8개 2 6개 
#명령문  csv파일 data frame 판다스
#기계학습 모형 생성 및 예측-> MLPClassifier    fit 함수 
#fit.(x,y) 학습  predict(x)-> 예측 
#모형 평가   socre() 분류는 accuracy 회귀는 결정 계수
#metrics()-> 함수 평가하는 것들 들어있어


# In[ ]:




