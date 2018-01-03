#coding=utf-8

from numpy import *
from pylab import *
from matplotlib import *

#######加载数据######
def loadData():
	data=loadtxt('C:\Users\DELL\Desktop\linear\data2.txt', delimiter=',')
	dataMat=data[:,0:2]  #取出x值
	labelMat=data[:,2:]  #取出y值
	return dataMat,labelMat

#######特征缩放######
def featureNomalize(X):
	m,n=X.shape
	u=zeros([1,n])
	s=zeros([1,n])
	x=zeros([m,n])
	for i in range(n-1):            #对除去第一列的各列求均值和标准差
		u[0,i+1]=mean(X[:,i+1])
		s[0,i+1]=std(X[:,i+1])
	for i in range(n):              #对除去第一列的各列进行特征缩放
		for j in range(m):
			if i==0:
				x[j,i]=1
			else:
				x[j,i]=(X[j,i]-u[0,i])/s[0,i]
	return x,u,s

######代价函数######
def costFunction(theta,X,y):
	m=X.shape[0]       #获取样本个数
	h=X.dot(theta)     #计算预测值
	e=h-y              #预测值与实际值的差
	J=e.T.dot(e)/m/2   #代价
	return J

######梯度下降######
def gradDescent(theta,X,y,alpha,iter):
	m=X.shape[0]                     #获取样本个数
	J_history=zeros([iter,1])        #存放迭代过程中的代价J
	for i in range(iter):	
		h=X.dot(theta)               #计算预测值
		e=(h-y)                      #计算与实际值的偏差
		theta = theta-(1.0/m)*alpha*X.T.dot(e)  #同时更新theta
		J_history[i]=costFunction(theta,X,y)    #计算theta更新后的代价J
	return theta,J_history

######预测######
def predict(theta,x,u,s):
	m,n=x.shape
	x_scal=zeros([m,n])
	x_scal[:,0]=x[:,0]
	x_scal[:,1:]=(x[:,1:]-u[:,1:])/s[:,1:]
	print x_scal
	h=x_scal.dot(theta)
	return h

#======================================================功能测试============================#
datax,datay=loadData()            #加载数据   
print datax.shape
print datay.shape
    
x0=ones([datax.shape[0],1])       #增加一列x0=1
X=column_stack([x0,datax])

init_theta=zeros([X.shape[1],1])  #初始化theta为0
print init_theta.shape

J=costFunction(init_theta,X,datay)#计算初始代价
print('initial cost: ',J)

x,u,s=featureNomalize(X)          #特征缩放

#####设置参数，运行梯度下降
alpha=0.1
iter=1500
theta,J_history=gradDescent(init_theta,x,datay,alpha,iter)
print('Theta found by gradient descent: ',theta)

#####预测######
t1=mat(array([1,1650,3]))
h=predict(theta,t1,u,s)
print h
