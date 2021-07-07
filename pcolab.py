import numpy as np
import pandas as pd
import random as rn
import time
from sklearn.metrics.cluster import rand_score , mutual_info_score

class myn:
    def __init__(self, x,k ,n0 ,Y , strdata):
        self.x_array_with_class=pd.DataFrame(x)
        self.x_array_with_class=np.array(self.x_array_with_class)
        
        
        if strdata=="wbc": #preprocessing for wbc dataset (ignore ? tuple(16 tuple  from 698 =0.02%) & change str col to int)
            b=self.x_array_with_class.copy()
            i=0
            while i< len(b):
                if '?' in b[i] :
                    # print(i)
                    b=np.delete(b,i,0)
                i+=1 
            b[:,5]=pd.to_numeric(b[:,5])
            self.x_array_with_class=b
        self.x_array=self.x_array_with_class.copy()
        self.x_array=np.delete(self.x_array,-1,1)# (array,number of col or row, axi=0 for row 1 for col)
        # self.x_array=np.array([[1,1],[2,1],[1,2],[2,2],[-3,-1],[-4,-1],[-3,-2],[-4,-2],[5,-14],[-5,14]])
        # print(x[3] , type(x))
        
        self.n=len(self.x_array)
        self.k=k
        self.Y=Y
        self.n0=int (n0 * self.n)
        self.n_max=100
        self.sigma=10 ** -6
        self.z=[]
        # self.s=0
        # self.p0=0
        self.u=np.zeros([self.n,self.k+1],dtype=int)
        self.u_befor=np.zeros([self.n,self.k+1],dtype=int)
        self.mi= np.zeros(self.n,dtype=int)
        self.dimi= []
        self.ii=np.zeros(self.n,dtype=int)
        self.o=[]
        self.true_u=np.zeros([self.n,self.k+1],dtype=int)
    
    def create_z0(self):    
        random_data_index=[int(rn.uniform(0 , len(self.x_array))) for i in range(self.k) ]
        
        # for i in random_data_index:
        #     z.append(list(x_array[i]))
        self.z=[list(self.x_array[i]) for i in random_data_index]
    
    def zigma_norm2(self,a,b):
        l2 = np.sum(np.power((a-b),2))
        return l2
    
    def update0(self):
        self.oreder_dimi()

    def oreder_dimi(self):
        self.u=np.zeros([self.n,self.k+1],dtype=int)
        self.dimi=[]
        for i in range(self.n):
            min1=self.zigma_norm2(self.x_array[i],self.z[0])
            zj=0
            for j in range(1,len(self.z)):
                candid=self.zigma_norm2(self.x_array[i], self.z[j])
                if candid < min1:
                    min1 , zj=candid,j
            self.u[i][zj]=1 #create new u that col k+1 are 0
            self.mi[i]=zj
            self.dimi.append([min1,i, self.mi[i]])
        
        self.dimi.sort()
        self.dimi.reverse()
        # print(self.dimi)
        for j in range(self.n):
            self.ii[j]=self.dimi[j][1]
        # print('u in update' ,self.u)
        # print(self.ii)
  

    def zigma_zigma(self, u1,z1):
        st1=0
        for l in range(self.k):  
            for j in range(self.n):
                st1+=u1[j][l] * self.zigma_norm2(self.x_array[j],z1[l])
        return st1        

    def D(self,u1,z1):
        st1=self.zigma_zigma(u1,z1)
        s2=0
        for j in range(self.n):
            s2+=u1[j][self.k ]
        return (self.Y / (self.n - s2)) * st1

    def P(self):
        st1=self.zigma_zigma(self.u , self.z)
        s2=0
        for j in range(self.n):
            s2+=self.u[j][self.k ]
        D=(self.Y / (self.n - s2)) * st1
        ll=0
        for i in range(self.n):
            ll+= self.u[i][-1 ] * D
        
        ll+= st1
        return ll

   
    def Theorem1(self):
        # for i in range(self.n):
        #     min1=np.sum(np.power((self.x_array[i]-self.z[0]),2))
        #     for l in range(self.k):
        #         dil=np.sum(np.power((self.x_array[i]-self.z[l]),2))
        #         min1= dil if dil < min1 else min1
        # dimi=min1  
        self.oreder_dimi()
        d=self.D(self.u_befor,self.z)
        dimi_grater=[]
        for j in range(len(self.dimi)):
            if self.dimi[j][0]> d:
                dimi_grater.append(self.dimi[j][1])
        self.o=list(set.intersection(set(self.ii[: self.n0]), set(dimi_grater)))  
        # print('o: ' , self.o)
        for i in range(self.n):
            for l in range(self.k):
                if i in self.o:
                    self.u[i][l]=0
                if i not in self.o and l==self.mi[i]:
                    self.u[i][l]=1 
            l=self.u[i]
            self.u[i][self.k ]=1- sum(l[:-1])
        # print('u in theor' ,self.u)
    
    def Theorem3(self):
        dimenstion_xi=len(self.x_array[0])
        # for l in range(self.k):
        l=0
        
        while True:
            sorat=0
            makhraj=0
            for i in range(self.n):
                makhraj+=self.u[i][l]
            for s in range(dimenstion_xi):
                for i in range(self.n):               
                    sorat +=self.u[i][l] * self.x_array[i][s]                   
                self.z[l][s]=sorat/makhraj
                sorat=0
                
            l+=1
            
            if l>= self.k:
                break
        # print(self.z)
    
    def rand_index_wbc(self):
        true_cluster_wbc=[]
        for i in range(len (self.x_array_with_class)):
            if self.x_array_with_class[i][-1]==2:
                true_cluster_wbc.append(0)
            if self.x_array_with_class[i][-1]==4:
                true_cluster_wbc.append(1)
        pred_cluster_wbc=[]
        num_outlier=0
        for i in range(len (self.u)):
            if self.u[i][0]==1:
                pred_cluster_wbc.append(0)
            if self.u[i][1]==1:
                pred_cluster_wbc.append(1)
                num_outlier+=1
        
         
        m1=mutual_info_score(true_cluster_wbc, pred_cluster_wbc, contingency=None) 
        
        return rand_score(true_cluster_wbc, pred_cluster_wbc) , num_outlier , m1
        
    def rand_index_shuttle(self):
        true_cluster_shuttle=[]
        outlier_list=[2,3,6,7]
        normal_list=[1,4,5]
        for i in range(len (self.x_array_with_class)):
            if self.x_array_with_class[i][-1] in normal_list:
                true_cluster_shuttle.append(0)
                # print(self.x_array_with_class[i][-1])
            if self.x_array_with_class[i][-1] in outlier_list:
                true_cluster_shuttle.append(1)
        pred_cluster_shuttle=[]
        num_outlier=0
        for i in range(len (self.u)):
            if self.u[i][0] ==1 :
                pred_cluster_shuttle.append(0)
            if self.u[i][1] ==1 :
                pred_cluster_shuttle.append(1)
                num_outlier+=1
        
        mutual_info_score1=mutual_info_score(true_cluster_shuttle, pred_cluster_shuttle, contingency=None)

        return rand_score(true_cluster_shuttle, pred_cluster_shuttle) , num_outlier , mutual_info_score1      


for y1 in range(1,3):
    print(y1)
    if y1==1:
        print('wbc:')
        x= pd.read_csv("wbc.csv")
        # x= pd.read_csv("https://raw.githubusercontent.com/ft46/kdd/main/wbc.csv")
    if y1==2:
        print('shuttle:')
        x= x= pd.read_csv("shuttle.csv")
        # x= pd.read_csv("https://raw.githubusercontent.com/ft46/kdd/main/shuttle.csv")
    # if y1==3:
    #   x= pd.read_csv("https://github.com/ft46/kdd/blob/main/mydata.csv")  
    time_execute_avg=0
    rand_index_avg=0
    num_outlier_avg=0
    mutual_info_score_avg=0
    for i in range(100):
        
        if y1==1:
        # for wbc k=1 , n0=0.5 ,Y=3
            myob=myn(x , 1 , 0.5, 3,"wbc")
        if y1==2:
        # for shuttle k=1 , n0=0.1 ,Y=9
            myob=myn(x , 1 , 0.1 ,9 ,"shuttle")
        if y1==3:  
            myob=myn(x , 2 , 0.1, 3,"shuttle")
        start_time=time.time()
        myob.create_z0()
        s=0
        p_old=0
        myob.update0()
        myob.u_befor=myob.u.copy()
        while True :
            myob.Theorem1()
            myob.Theorem3()
            myob.u_befor=myob.u.copy()
            s+=1
            p_new=myob.P()
            if abs(p_new - p_old) < myob.sigma or s> myob.n_max:
                break
            p_old=p_new
        if y1==1:
            rand_index , num_outlier  , m1 =myob.rand_index_wbc()
        if y1==2:
            rand_index , num_outlier,m1 =myob.rand_index_shuttle()
        # print(rand_index ,'\n', num_outlier)
        # print(myob.u)
        # print(myob.z)
        del myob
        time_execute=time.time()-start_time

        time_execute_avg += time_execute
        rand_index_avg += rand_index
        num_outlier_avg += num_outlier
        mutual_info_score_avg += m1


    time_execute_avg /=100
    rand_index_avg /= 100
    num_outlier_avg /=100
    mutual_info_score_avg /=100
    print('time_execute_avg= ',time_execute_avg )
    print('rand_index_avg= ' ,rand_index_avg)
    print('num_outlier_avg= ' ,num_outlier_avg )
    print('mutual_info_score_avg= ' ,mutual_info_score_avg )


