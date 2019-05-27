
#%
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.utils import resample
#from sklearn.naive_bayes import GaussianNB,ComplementNB,MultinomialNB,BernoulliNB
import numpy as np
# pd.options.display.max_cols=1000
# pd.options.display.max_rows=100
missing_values = ["n/a", "na", "--","NA","?",""," ",-1,"NAN","NaN"]
df=pd.read_csv('kick.csv',na_values=missing_values)
num_rows=len(df)

drop_cols=list()
print(df.isnull().sum().sum())
print("Dtypes Before Preprocessing",df.dtypes)
null_stats=df.isnull().sum()
#df=df.dropna(thresh=100,axis='rows')
######################################categorically encoding remaining columns###############
##################### & perform one-hot encoding  ###########################################
#enc=OneHotEncoder(handle_unknown='ignore')
# for col in df.columns:
#     if df[col].dtypes=='object':
#         print(col)
#         df[col]=df[col].fillna(df[col].value_counts().index[0])
#         df[col]=df[col].astype('category')
        
#         cat_df=df[col].cat.codes
#         zip_df=pd.DataFrame(list(zip(df[col],cat_df)))
#         df[col]=enc.fit_transform(zip_df)
#         print(zip_df)
#         print("@@@@@@@@@@@@@@@@@@")
#         print(cat_df)
#         print("******************")
#print(df['Make'])


#%
for index,val in enumerate(null_stats):
    if val > num_rows/2:
        #print(val)
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


#%
# df_new=enc.fit_transform(df)
len(df.columns)


#%
# import qgrid


#%

#%
# grid=qgrid.QGridWidget(df=df)
# display(grid)











#%
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
import time
# def train_model(model,df=None,n=5):
#     X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df['IsBadBuy'] ,stratify=df['IsBadBuy'],test_size=0.25,shuffle=True)
#     model.fit(X_train,y_train)


#     print("Total Dataset Size={} x {}".format(num_rows,len(df.columns)))
#     start=time.time()
#     score=cross_val_score(model,X_test,y_test,cv=n)#model.score(X_test,y_test)
#     end=time.time()
#     y_pred=model.predict(X_test)
#     prec_recall=precision_recall_fscore_support(y_test,y_pred,average='weighted')
#     print("Score= {}".format(score.mean()))
#     print("Precision,Recall,F_beta,Support {}".format(prec_recall))
#     from sklearn import metrics
#     y_pred=model.predict(X_test)
#     metrics.confusion_matrix(y_test,y_pred)
#     return (1-score.mean()),(end-start),prec_recall
#%


#%
import matplotlib.pyplot as plt
import seaborn as sns
# print("Errors=",scores)
# precision=[row[0] for row in quad_metrics]
# recall=[row[1] for row in quad_metrics]
# print("Time in Seconds {}" .format(time_taken))
# model_names=['Gaussian','Bernoulli','Multinomial','ComplementNB']
# data=pd.DataFrame(list(zip(model_names,scores)),columns=['Naive Bayes Variants','error=1-accuracy'])
# data1=pd.DataFrame(list(zip(model_names,time_taken)),columns=['Naive Bayes Variants','time_taken in seconds'])
# data2=pd.DataFrame(list(zip(model_names,precision)),columns=['Naive Bayes Variants','precision'])
# data3=pd.DataFrame(list(zip(model_names,recall)),columns=['Naive Bayes Variants','recall'])
# sns.barplot(x="Naive Bayes Variants",y="error=1-accuracy",data=data)
# plt.show()
# sns.barplot(x="Naive Bayes Variants",y="time_taken in seconds",data=data1)
# plt.show()
# sns.barplot(x="Naive Bayes Variants",y="precision",data=data2)
# plt.show()
# sns.barplot(x="Naive Bayes Variants",y="recall",data=data3)
# plt.show()


   
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time
df_new=pd.get_dummies(df)
len(df_new.columns)
models=[]
scores=[]
time_taken=[]
quad_metrics=[]

#X_train, X_test, y_train, y_test = train_test_split(df_new.iloc[:,1:], df_new['IsBadBuy'] ,stratify=df_new['IsBadBuy'],test_size=0.25,shuffle=True)
#model=SVC(kernel='poly',degree=2,gamma='auto')

# start=time.time()
# model.fit(X_train,y_train)
# end=time.time()
# print("end of training")
# print(model.score(X_test,y_test))
# n=5 #cv folds
# score=cross_val_score(model,X_test,y_test,cv=n)
# y_pred=model.predict(X_test)
# prec_recall=precision_recall_fscore_support(y_test,y_pred,average='weighted')
# print(end-start)
# print(score.mean())
# print(prec_recall)
# for model in models:
#     s,t,m=train_model(model=model,df=df_new,n=5)
#     scores.append(s)
#     quad_metrics.append(m)
#     time_taken.append(t)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
import time
def train_model(model,df=None,n=5):
    print("****************************")
    print(model)
    print("****************************")
    df_majority = df[df['IsBadBuy']==0]
    df_minority = df[df['IsBadBuy']==1]
    df_minority_upsampled = resample(df_minority, 
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=123)     # sample with replacement
                                                           
                                         # to match majority class
                                  # reproducible results
    df=pd.concat([df_majority, df_minority_upsampled])
    print("before stratified sampling")
    print("0 stats = {}".format(len(df[df['IsBadBuy']==0])))
    print("1 stats = {}".format(len(df[df['IsBadBuy']==1])))
    print("ratio stats = {}".format(len(df[df['IsBadBuy']==0])/len(df[df['IsBadBuy']==1])))
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df['IsBadBuy'] ,stratify=df['IsBadBuy'],test_size=0.25,shuffle=True)
    model.fit(X_train,y_train)
    
    print("After stratified sampling")
    print("Train Data")
    print("0 stats = {}".format(y_train[y_train==0].size))
    print("1 stats = {}".format(y_train[y_train==1].size))
    print("ratio stats = {}".format(y_train[y_train==0].size/y_train[y_train==1].size))
    
    print("Test Data")
    print("0 stats = {}".format(y_test[y_test==0].size))
    print("1 stats = {}".format(y_test[y_test==1].size))
    print("ratio stats = {}".format(y_test[y_test==0].size/y_test[y_test==1].size))
    # y_predicted=model.predict(X_test)
    # y_var=np.var(y_predicted,axis=1)
    # y_noise=np.var(y_test,axis=1)
    print("Total Dataset Size={} x {}".format(num_rows,len(df.columns)))
    start=time.time()
    score=cross_val_score(model,X_test,y_test,cv=n)#model.score(X_test,y_test)
    
    end=time.time()
    y_pred=model.predict(X_test)
    dat=pd.DataFrame(data=y_test)
    print(dat)
    print(len(dat[dat['IsBadBuy']==0]))
    print(len(dat[dat['IsBadBuy']==1]))
    print(np.unique(y_test))
    print(np.unique(y_pred))
    prec_recall=precision_recall_fscore_support(y_test,y_pred,average='weighted')
    print("Score= {}".format(score.mean()))
    print("Error= {}".format(1-score.mean()))
    print("Training Time={}".format((end-start)))
    print("Precision,Recall,F_beta,Support {}".format(prec_recall))
    from sklearn import metrics
    y_pred=model.predict(X_test)
    print(metrics.confusion_matrix(y_test,y_pred))
    return (1-score.mean()),(end-start),prec_recall
# models.append(GaussianNB())

# models.append(BernoulliNB())
# models.append(MultinomialNB())
# models.append(ComplementNB())
# quad_metrics=[]

C=0.001

# models.append(LogisticRegression(penalty='l2',solver='newton-cg',C=C,max_iter=1000))
# models.append(LogisticRegression(penalty='l2',solver='lbfgs',C=C,max_iter=1000))
# models.append(LogisticRegression(penalty='l2',solver='liblinear',C=C,max_iter=1000))
# models.append(LogisticRegression(penalty='l2',solver='saga',C=C,max_iter=1000))
# models.append(LogisticRegression(penalty='l2',solver='sag',C=C,max_iter=1000))

while (C<=1000):
        print(C)
        models.append(LogisticRegression(penalty='l2',solver='lbfgs',C=C,max_iter=1000))
        C=C*10

print("*****************")
print(models)
print("******************")
for model in models:
    s,t,m=train_model(model=model,df=df_new,n=5)
    scores.append(s)
    quad_metrics.append(m)
    time_taken.append(t)
    

print(scores)
print(quad_metrics)
print(time_taken)
# import matplotlib.pyplot as plt
# import seaborn as sns
# print("Errors=",scores)
# precision=[row[0] for row in quad_metrics]
# recall=[row[1] for row in quad_metrics]
# print("Time in Seconds {}" .format(time_taken))
# model_names=['lbfgs','Bernoulli','Multinomial','ComplementNB']
# data=pd.DataFrame(list(zip(model_names,scores)),columns=['Naive Bayes Variants','error=1-accuracy'])
# data1=pd.DataFrame(list(zip(model_names,time_taken)),columns=['Naive Bayes Variants','time_taken in seconds'])
# data2=pd.DataFrame(list(zip(model_names,precision)),columns=['Naive Bayes Variants','precision'])
# data3=pd.DataFrame(list(zip(model_names,recall)),columns=['Naive Bayes Variants','recall'])
# sns.barplot(x="Naive Bayes Variants",y="error=1-accuracy",data=data)
# plt.show()
# sns.barplot(x="Naive Bayes Variants",y="time_taken in seconds",data=data1)
# plt.show()
# sns.barplot(x="Naive Bayes Variants",y="precision",data=data2)
# plt.show()
# sns.barplot(x="Naive Bayes Variants",y="recall",data=data3)
# plt.show()
#print(df1)
# correlation on original data and not pre-processed data
def get_redundant_pairs(data):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = data.columns
    print(cols)
    for i in range(0, data.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(data, n=5):
    au_corr = data.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(data)
    print(au_corr)
    #au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]




#kde=KernelDensity(kernel='gaussian').fit(df_new.iloc[:,1:])
#kde.score_samples(df_new.iloc[:,1:])
#df.iloc[:,1:]


#%

# 