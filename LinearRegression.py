#coding=utf-8

from numpy import *
from pylab import *
from matplotlib import *

#######加载数据######
def loadData():
	data=loadtxt('C:\Users\DELL\Desktop\linear\data1.txt', delimiter=',')
	dataMat=data[:,0:1]  #取出x值
	labelMat=data[:,1:]  #取出y值
	return dataMat,labelMat
	
######绘出数据分布######	
def plotData(dataMat,labelMat):
	plot(dataMat, labelMat, 'x')
	xlabel('poopulation in 10,000s')
	ylabel('profit in $10,000s')
	show()

######代价函数######
def costFunction(theta,X,y):
	m=X.shape[0]
	h=X.dot(theta)     #计算预测值
	e=h-y              #预测值与实际值的差
	J=e.T.dot(e)/m/2   #代价
	return J

######梯度下降######
def gradDescent(theta,X,y,alpha,iter):
	m=X.shape[0]
	J_history=zeros([iter,1])

	for i in range(iter):	
		h=X.dot(theta)
		e=(h-y)
		theta = theta-alpha*(1.0/m)*X.T.dot(e)  #同时更新theta
		J_history[i]=costFunction(theta,X,y)
	return theta,J_history

######预测函数######
def predict(theta,x):
	h=x.dot(theta)       #计算h预测值
	return h	
	
#######拟合曲线######	
def linearFit(theta,x,y):
	h=x.dot(theta)
	plot(x[:,1],y,'x')
	plot(x[:,1],h,'r--')
	xlabel('poopulation in 10,000s')
	ylabel('profit in $10,000s')
	title('univariable linear regression')
	show()


#########功能测试#######
datax,datay=loadData()            #加载数据
plotData(datax,datay)             #画出分布图
init_theta=zeros([2,1])           #初始化theta为0
x0=ones([datax.shape[0],1])       #增加一列x0=1
X=column_stack([x0,datax])
J=costFunction(init_theta,X,datay)#计算初始代价
print('initial cost: ',J)

#####设置参数，运行梯度下降
alpha=0.01
iter=1500
theta,J_history=gradDescent(init_theta,X,datay,alpha,iter)
print('Theta found by gradient descent: ',theta)
#####预测两组数据
t1=array([1,3.5])
print('prediction of t1: ',predict(theta,t1)) 
t2=array([1,7])
print('prediction of t2: ',predict(theta,t2))
#####绘制拟合曲线
linearFit(theta,X,datay)