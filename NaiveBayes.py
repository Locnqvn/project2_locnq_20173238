import numpy as np
from sklearn.naive_bayes import MultinomialNB


class Data:
    def __init__(self, matrixfeature, label):
        self.X = matrixfeature
        self.label = label


def BinarySearch(word, wordslist):
    l = 0
    r = len(wordslist)-1
    while(l <= r):
        m = int((r+l)/2)
        if(word == wordslist[m]):
            return m
        else:
            if(word < wordslist[m]):
                r = m-1
            else:
                l = m+1
    return -1

# Tim wordslist voi nguong df


def find_wordslist(df):
    # doc du lieu
    with open("C:\\Users\\nql\\Desktop\\20192\\project2\\textpreprocessing\\text_train", 'r') as f:
        train = f.read().splitlines()
    # tim wordslist
    words = []
    for document in train:
        text_data = document.split("<<>>")
        text = text_data[-1].split()
        words += list(set(text))
    words = list(set(words))
    words.sort()
    words_df = dict.fromkeys(words, 0)
    for document in train:
        text_data = document.split("<<>>")
        text = set(text_data[-1].split())
        for word in text:
            words_df[word] += 1
    words_list = [word for word in words if words_df[word] > df]
    return words_list


def feature_vector(words_list, pathin):
    # doc du lieu
    with open(pathin, 'r') as f:
        data = f.read().splitlines()
    # tim ma tran chua cac vector dac trung
    dim_of_vector = len(words_list)
    num_of_document = len(data)
    label = []
    matrix_feature_vector = []
    for i in range(num_of_document):
        text_data = data[i].split("<<>>")
        label.append(text_data[0])
        words_df=dict.fromkeys(words_list,0)
        text = text_data[-1].split()
        for word in text:
            if(BinarySearch(word, words_list) != -1):
                words_df[word] += 1
        matrix_feature_vector.append(list(words_df.values()))
    return Data(matrix_feature_vector, label)

class MultiNB:
    def fit(self,X_train,label):
        num_of_document=len(label)
        dim_fea_vector=len(X_train[0])
        setlabel=list(set(label))
        num_of_labels=len(setlabel)
        numdoc_of_each_class=dict.fromkeys(setlabel,0)
        num_words_in_class=[]
        for i in setlabel:
            num_words_in_class.append(np.zeros(dim_fea_vector))
        for i in range(num_of_document):
            numdoc_of_each_class[label[i]]+=1
            num_words_in_class[setlabel.index(label[i])]+=X_train[i]

        for i in range(num_of_labels):
            total_words=sum(num_words_in_class[i])+dim_fea_vector
            num_words_in_class[i]=(num_words_in_class[i]+1)/total_words
            num_words_in_class[i]=np.log10(num_words_in_class[i])
        
        log_pc=np.array(list(numdoc_of_each_class.values()))
        log_pc=np.log10(log_pc/num_of_document)
        self.log_lamda_class=num_words_in_class
        self.log_pc=log_pc
        self.dim_fea_vector=dim_fea_vector
        self.labels=setlabel
        self.num_of_labels=len(setlabel)

    def predict(self,X_test):
        predict=[]
        num_doc_of_test=len(X_test)
        log_lamda_class=self.log_lamda_class
        log_pc=self.log_pc
        dim_fea_vector=self.dim_fea_vector
        labels=self.labels
        for i in range(num_doc_of_test):
            pretmp=np.zeros(self.num_of_labels)
            for j in range(self.num_of_labels):
                pretmp[j]+=np.dot(log_lamda_class[j],X_test[i])
            pretmp+=log_pc
            pos=np.array(pretmp).argmax()
            predict.append(labels[pos])
        return predict

predict_me=[]
predict_sklearn=[]
list_len=[]


for i in range(1):
    words_list = find_wordslist(6)
    path_train = "C:\\Users\\nql\\Desktop\\20192\\project2\\textpreprocessing\\text_train"
    path_test = "C:\\Users\\nql\\Desktop\\20192\\project2\\textpreprocessing\\text_test"
    train = feature_vector(words_list, path_train)
    test = feature_vector(words_list, path_test)
    len_words_list=len(words_list)
    print(len_words_list)
    list_len.append(len_words_list)
    n_test=len(test.label)
    # kiem tra lai voi sklearn
    '''
    model=MultinomialNB()
    model.fit(train.X,train.label)
    pre_sklearn=[]
    for i in range(n_test):
        pre_sklearn.append(model.predict([test.X[i]])[0])
    '''
    #
    a=MultiNB()
    a.fit(train.X,train.label)
    pre_me=a.predict(test.X)

    # ans 
    cnt_me=0
    cnt_sklearn=0
    for i in range(n_test):
        if(int(pre_me[i])==int(test.label[i])):
            cnt_me+=1
        '''
        if(int(pre_sklearn[i])==int(test.label[i])):
            cnt_sklearn+=1
        '''
    predict_me.append(cnt_me)
    predict_sklearn.append(cnt_sklearn)

print(list_len)
print(predict_me)
#print(predict_sklearn)
