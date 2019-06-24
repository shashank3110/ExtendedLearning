#%%
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import json
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support,mean_squared_error
import time
import sqlite3
import nltk
nltk.download('punkt')
nltk.download('stopwords')
#nltk.download('vader')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
#from textblob.sentiments import NaiveBayesAnalyzer
from functools import reduce
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import sprint6
#%%
# con=sqlite3.connect('database.sqlite')
# #df=pd.read_sql('Select * from Reviews',con=con,chunksize=10000)
# df=pd.read_sql('Select * from Reviews',con=con)
# #data=[]
# # for chunk in df:
# #     data.append(chunk)

# s=df['Score']
# #s=df['ProductId']
# sample=resample(df,n_samples=25000,replace=False,stratify=s)
#%%
# sample=data[0]
# print(type(df))
#%%
def get_sentiments_from_polarity(polarity=0,threshold=0):
    if np.abs(polarity) < threshold :
        return "Neutral"
    elif np.sign(polarity)>0:
        return "Positive"
    else:
        return "Negative"


#%%
def get_reviews():
    con=sqlite3.connect('database.sqlite')
    df=pd.read_sql('Select * from Reviews',con=con)
    s=df['Score']
    sample=resample(df,n_samples=10000,replace=False,stratify=s)
    
    cleaned_review=pd.DataFrame(columns=['Summary','Helpful','Score','Sentiment','Sentiment_Polarity'])
    return cleaned_review,sample,df

#%%    
def preprocess_and_get_sentiments(cleaned_review=None,sample=None,threshold=0):
    cleaned_review['Score']=sample['Score']
    cleaned_review['Summary']=sample['Summary']
    ######################Preprocessing###########################
    stop_words=set(stopwords.words("english"))
    punc = list(string.punctuation)
    punc.extend(["`","``","''","..."])
    ##################################################
    i=0
    sentiment_result=[]
    sentiment_polarity=[]

    for review in sample['Summary']:
        #print(review)
        ##################Preprocessing########################
        # review=word_tokenize(review)
        # review=[word for word in review if word not in stop_words] #Removing stop words
        # review=[word for word in review if word not in punc] #Removing punctuations
        # if len(review)>0:
        #     filtered_review=reduce(lambda a,b:a+' '+b,review)
        # else:
        #     filtered_review=''
        if len(review)>0:
            filtered_review=review
        else:
            filtered_review=''
        #############################################
        sentiment = TextBlob(filtered_review).sentiment
        print(f'Sentiment={sentiment}')
        s=get_sentiments_from_polarity(sentiment.polarity,threshold)
        print(s)
        sentiment_result.append(s)
        print(sentiment.polarity)
        sentiment_polarity.append(sentiment.polarity)
        i+=1

    deno=sample['HelpfulnessDenominator']
    nume=sample['HelpfulnessNumerator']
    sample.index=cleaned_review.index
    cleaned_review['Helpful']=np.where(deno>0,nume/deno,0)
    cleaned_review['Sentiment']=sentiment_result
    cleaned_review['Sentiment_Polarity']=sentiment_polarity
    print(cleaned_review['Sentiment'])
    sample['Sentiment']=cleaned_review['Sentiment']
    sample['Sentiment_Polarity']=cleaned_review['Sentiment_Polarity']
    #helpful_df=predict_helpfulness(sample)
    sample['Helpful_Score']= cleaned_review['Helpful'] #pd.cut(cleaned_review['Helpful'], bins = [1,2,3,4,5], include_lowest = True)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(sample.head())
    #print(sample.Sentiment_Polarity)
    return cleaned_review,sample
    
#%%
# s='nn ! 8,l'
# s=s.translate(dict.fromkeys(string.punctuation))
# print(s)
# l=list(string.punctuation)
# cleaned_review.tokenized_reviews[i]=review
# print(cleaned_review.tokenized_reviews)
# reviews=pd.DataFrame(data=rlist,columns=['tokenized_reviews'])
# print(reviews)
#%%
def concat_sentiments(cleaned_review=None,sample=None):
    
    #%%
    review_df=sample

    review_df=review_df.drop(columns=['HelpfulnessDenominator','HelpfulnessNumerator','Summary','Text'],axis=1)
    #%%
    review_df['Sentiment']=cleaned_review['Sentiment']
    #print("review df .sentiment")
    #print(review_df['Sentiment'])
    review_df['Sentiment']=review_df['Sentiment'].astype('category')
    review_df['Sentiment']=review_df['Sentiment'].cat.codes
    review_df=review_df.drop(columns=['Sentiment_Polarity'],axis=1)
    review_df['ProductId']=review_df['ProductId'].astype('category')
    review_df['ProductId']=review_df['ProductId'].cat.codes
    #%%
    review_df.dtypes
    #one-hot encoding the sentiments
    sentiment_df=pd.get_dummies(review_df['Sentiment'],columns=['Sentiment_Neutral','Sentiment_Negative','Sentiment_Positive'])
    #print("Sentiemnt DF")
    #print(sentiment_df.head())
    sentiment_df.columns=['Sentiment_Neutral','Sentiment_Negative','Sentiment_Positive']#(index=str,columns={0:'Sentiment_Neutral',1:'Sentiment_Negative',2:'Sentiment_Positive'})

    #%%
    review_df=pd.concat([review_df,sentiment_df],axis=1)
    review_df=review_df.drop(columns=['Sentiment'])
    #print(len(review_df['UserId'].unique()))
    #print(len(review_df['ProductId'].unique()))
    return review_df
    
#%%

def train_model(model,X,y,n=5,upsample=False,kernelTransform=False,gamma = 0.1,kernel='rbf',sampleForSVC=False):
    print("****************************")
    num_rows=len(X)
    metrics={}
    print("****************************")
    print(model)
    print("****************************")
    
    print("Total Dataset Size={} x {}".format(num_rows,10))
    
    
    print(f"before split {len(X)}")
    #df_train=df.drop(['Class'],axis=1)
    stratify=y
    y_test,y_pred,t,_=train_core(model,X,y,stratify,n)
    cnf=confusion_matrix(y_test,y_pred)
    # This is the Final test accuracy
    s=accuracy_score(y_test,y_pred)
    prec_recall=precision_recall_fscore_support(y_test,y_pred,average='weighted')
    p,r,f,_=prec_recall
    
    #prec_recall=precision_recall_fscore_support(y_test,y_pred,average='weighted')
    
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
    
    return metrics,cnf
#%%
def train_core(model=None,X=None,y=None,stratify=None,n=5):
    
    print(y.head())
    #stratify=X['ProductId']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y ,stratify=stratify,test_size=0.49,shuffle=True)
    print(f" after split {len(X_train)}") 
    kmodels=[]
    kscores=[]
    #nfold or kfold  crossvaidation
    kf=KFold(n_splits=n)
    t=0
    for train_index,test_index in kf.split(X_train,y_train):
        #print(f"Train Index start={train_index[0]} & end= {train_index[-1]}")
        #print(f"Test Index start= {test_index[0]} & end= {test_index[-1]}")
        Xk_train, Xk_test = X_train[train_index[0]:], X_train[test_index[0]:]
        yk_train, yk_test = y_train[train_index[0]:], y_train[test_index[0]:]
        start=time.time()
        m=model.fit(Xk_train,yk_train)
        end=time.time()
        kmodels.append(m)
        yk_pred=m.predict(Xk_test)
        # Note this is the cross validation accuracy and not the final test accuracy
        if stratify is not None:
            ks=accuracy_score(yk_test,yk_pred) #this function can be used only for multiclass classification
            kscores.append(ks)
            best_model_index=kscores.index(max(kscores))
        else:
            ks=mean_squared_error(yk_test,yk_pred)
            kscores.append(ks)
            best_model_index=kscores.index(min(kscores))

        t=t+(end-start)
    t=t/n
    print(f"Cross validation scores ={kscores}")
    
    print(f"Best model index ={best_model_index}")
    best_model=kmodels[best_model_index]
    print(f"Best model ={best_model}")
    y_pred=best_model.predict(X_test)

    return y_test,y_pred,t,best_model
#%%
def run():
    cleaned_review,sample,df=get_reviews()
    sample_helpful=resample(df,n_samples=10000,replace=False,stratify=df['Score'])
    #print(sample.head())
    #print(sample_helpful.head())
    cleaned_review_helpful=cleaned_review
    cleaned_review,sample=preprocess_and_get_sentiments(cleaned_review=cleaned_review,sample=sample,threshold=0.5)
    review_df=concat_sentiments(cleaned_review,sample)
    y=review_df['Score']
    X=review_df
    X=X.drop(columns=['UserId','Score','ProfileName','Time'],axis=1)
    ###########################################################
    cleaned_review_helpful,sample_helpful=preprocess_and_get_sentiments(cleaned_review=cleaned_review_helpful,sample=sample_helpful,threshold=0.5)
    review_df_helpful=concat_sentiments(cleaned_review_helpful,sample_helpful)
    #y_h=review_df_helpful['Score']
    #X_h=review_df_helpful
    X_helpful=review_df_helpful.drop(columns=['UserId','Score','ProfileName','Time'],axis=1)
    y_helpful=X_helpful['Helpful_Score']
    X_helpful=X_helpful.drop(columns=['Helpful_Score'],axis=1)

    helpful_df=predict_helpfulness(X_helpful,y_helpful,X)
    regression_error=mean_squared_error(y_helpful,helpful_df)
    ###########################################################
    #print(X.head)
    helpful_df=pd.DataFrame(data=helpful_df,index=X.index,columns=['Helpful_Score'])
    
    X['Helpful_Score']=helpful_df
    print("After adding predicted helpful scores")
    print(X.head)
    print("************* Training Final Reviews Model*****************")
    
    models={}
    model=RandomForestClassifier(criterion='gini',n_estimators=10)
    models.update({'Random Forest':[model,{'hyper':{'criterion': 'gini','n_estimators':'10'}}]}) #,'Max_depth':'None','Min_samples_split':'2'
    metrics,cnf=train_model(model,X,y)
    print(metrics)
    print(cnf)
    # return X
    return X,models,metrics,cnf,regression_error

#%%
from sklearn.linear_model import LinearRegression
def predict_helpfulness(X=None,y=None,train_data=None):
    model=LinearRegression()
    print("**************Training Helpful score model*************")
    _,_,_,best_model=train_core(model=model,X=X,y=y,n=5)
    train_data=train_data.drop(columns=['Helpful_Score'],axis=1)
    helpful_df=best_model.predict(train_data) 
    # get artificial features for our unkown train_data
    return helpful_df
# X_backup=X
# dummy_df=pd.get_dummies(X)

#%%
df,models,metrics,cnf,regression_error=run()
for i in models.items():
    m=i
#%%
def make_json(df=None,model_data=None,upsample=False,stratified_column=None,metrics=None,test_size=0.49,nfold=5):
    ''' Preparation of Metadata in Dictionary Format'''
    if upsample:
        sampling='Upsampling'
    else:
        sampling='Stratified Sampling stratified on '+ stratified_column

    encoding={'encoding_used':'One-Hot Encoding'}
    data_meta_data={'Name':'Amazon Fine Food Reviews','Rows':len(df),'Columns Before Preprocessing':10,'Columns After Preprocessing & one hot encoding':len(df.columns),'Encoding':encoding,'Classification Type':"Multi-Class","Class Variable":'Score'}
    training_charc={'Hyper Parameters':m[1][1]['hyper'],'Test_size':test_size,'No. of Cross Validation Folds Used':nfold,'Sampling':sampling}
    meta_data={m[0]:{'Data_Meta_Data':data_meta_data,'Training Characteristics':training_charc,"Metrics":metrics}}

    jfile='meta_data_'+str(np.random.randint(1,10000,1))+'.json'
    return meta_data,jfile

#%%
meta_data,jfile=make_json(df=df,model_data=m,stratified_column='Score',metrics=metrics)
sprint6.write_json(jfile,meta_data)


#%%
