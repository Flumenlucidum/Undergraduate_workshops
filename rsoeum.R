noise
filter(noise,)
noise2=read.csv('noise2.csv')
variable.names(noise2)
quart1=filter(noise2, 분기==1)
quart1
library(gdata)
install.packages('gdata')
library(gdata)
noi=read.xlsx('noise2.xlsx',fileEncoding='euc-kr')
variable.names(noi)
filter(noi, '분기'==1)
head(noi)
summary(noi)
filter(noi,'23시'==60)
soeum=read.csv('noise2.csv',fileEncoding='euc-kr')
soeum
variable.names(soeum)
so1=filter(soeum, 분기==1)
head(soeum)
filter(soeum,용도구분=='상업지역')
library(dplyr)
variable.names(soeum)
see =soeum %>%
  filter(분기==4, 야간평균>65)%>%
  group_by(측정지역)%>%
  summarise(num=n())
pur= soeum%>%
  filter(분기==4, 야간평균>65)%>%
  group_by(용도구분)%>%
  summarize(mean(야간평균),num=n())

soeum%>% filter