
#%
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB,ComplementNB,MultinomialNB,BernoulliNB
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
import time
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
import time
from sklearn import metrics


def preprocess():
    missing_values = ["n/a", "na", "--","NA","?",""," ",-1,"NAN","NaN"]
    df=pd.read_csv('kick.csv',na_values=missing_values)
    num_rows=len(df)

    drop_cols=list()
    print(df.isnull().sum().sum())
    print("Dtypes Before Preprocessing",df.dtypes)
    null_stats=df.isnull().sum()



    #%
    for index,val in enumerate(null_stats):
        if val > num_rows/2:
            
            drop_cols.append(df.columns[index])
    print("####################################")
    print(drop_cols)
    drop_cols.append('SubModel')
    drop_cols.append('Trim')
    drop_cols.append('Color')
    drop_cols.append('VNST')
    drop_cols.append('VehYear')
    drop_cols.append('WheelTypeID')
    drop_cols.append('VNZIP1')
    drop_cols.append('BYRNO')

    df=df.drop(drop_cols,axis=1)
    #replace missing values of numeric columns by median/mean
    df=df.fillna(df.median())
    #drop rows with missing values in column with datatype as object/string
    df=df.dropna(axis=0)
    len(df.columns)
    df_new=pd.get_dummies(df)
    return df_new


def train_model(model,df=None,n=5,upsample=False):
    print("****************************")
    print(model)
    print("****************************")
    metrics={}
    if upsample:
        df_majority = df[df['IsBadBuy']==0]
        df_minority = df[df['IsBadBuy']==1]
        df_minority_upsampled = resample(df_minority, 
                                    replace=True,
                                    n_samples=len(df_majority),
                                    random_state=123)     # sample with replacement
                                                            
                                            # to match majority class
                                    # reproducible results
        df=pd.concat([df_majority, df_minority_upsampled])

    # print("before stratified sampling")
    # print("0 stats = {}".format(len(df[df['IsBadBuy']==0])))
    # print("1 stats = {}".format(len(df[df['IsBadBuy']==1])))
    # print("ratio stats = {}".format(len(df[df['IsBadBuy']==0])/len(df[df['IsBadBuy']==1])))
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df['IsBadBuy'] ,stratify=df['IsBadBuy'],test_size=0.25,shuffle=True)
    model.fit(X_train,y_train)
    
    # print("After stratified sampling")
    # print("Train Data")
    # print("0 stats = {}".format(y_train[y_train==0].size))
    # print("1 stats = {}".format(y_train[y_train==1].size))
    # print("ratio stats = {}".format(y_train[y_train==0].size/y_train[y_train==1].size))
    
    print("Test Data")
    print("0 stats = {}".format(y_test[y_test==0].size))
    print("1 stats = {}".format(y_test[y_test==1].size))
    print("ratio stats = {}".format(y_test[y_test==0].size/y_test[y_test==1].size))

    print("Total Dataset Size={} x {}".format(num_rows,len(df.columns)))
    start=time.time()
    score=cross_val_score(model,X_test,y_test,cv=n)#model.score(X_test,y_test)
    end=time.time()
    y_pred=model.predict(X_test)

    print(np.unique(y_test))
    print(np.unique(y_pred))
    prec_recall=precision_recall_fscore_support(y_test,y_pred,average='weighted')
    p,r,f,sp=prec_recall
    t=(end-start)
    s=score.mean()
    metrics['Accuracy']=s
    metrics['Error']=1-s
    metrics['Precision']=p
    metrics['Recall']=r
    metrics['FScore']=f
    metrics['Training_Time_in_s']=t
    print("Score= {}".format(s))
    print("Error= {}".format(1-s))
    print("Training Time={}".format(t))
    print("Precision,Recall,F_beta,Support {}".format(prec_recall))
    
    #y_pred=model.predict(X_test)
    
    return metrics






def make_models():
    models={}
    ##  @Wen & @Uzair refer below commented line as example to add your models here
    #For eg: models.update({'SVC':[SVC(kernel='poly',gamma='auto',degree=4),'hyper':{'kernel':'poly','gamma':'auto','degree':4}}]})
    ##
    
    models.update({'GaussianNB':[GaussianNB(),{'hyper':{'fit_prior':'default'}}]})
    models.update({'BernoulliNB':[BernoulliNB(),{'hyper':{'fit_prior':'default'}}]})
    models.update({'MultinomialNB':[MultinomialNB(),{'hyper':{'fit_prior':'default'}}]})
    models.update({'ComplementNB':[ComplementNB(),{'hyper':{'fit_prior':'default'}}]})
    
    # This while loop is only for Logistic
    C=0.001
    while (C<=1000):
            print(C)
            models.update({'LogisticRegression':[LogisticRegression(penalty='l2',solver='lbfgs',C=C,max_iter=1000),{'hyper':{'penalty':'l2','solver':'lbfgs','C':C,'max_iter':1000}}]})
            C=C*10

    print("*****************")
    print(models)
    print("******************")
    return models

def run_my_models():
    models=make_models()
    df=preprocess()
    #This was the loop I was talking about in morning
    upsampling=False
    i=0
    for item in models.items():
        i=i+1
        model=item[1][0]
        if item[0]== 'LogisticRegression': # This condition is True only for Logistic
            upsampling=True
        results=train_model(model=model,df=df,n=5,upsample=upsampling)
    
        #json code  part will begin from here on We can do on saturday
    


#run_my_models()  