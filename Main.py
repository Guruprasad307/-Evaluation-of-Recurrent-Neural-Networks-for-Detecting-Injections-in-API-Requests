from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from nltk.corpus import stopwords
import nltk
import pickle
import seaborn as sns

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import Dropout, LSTM, GRU, SimpleRNN, Bidirectional

main = tkinter.Tk()
main.title("Evaluation of Recurrent Neural Networks for Detecting Injections in API Requests")
main.geometry("1300x1200")

global filename
global accuracy, precision, recall, fscore
global dataset
global X, Y
global X_train, X_test, y_train, y_test, vectorizer, labels, gru_model
stop_words = set(stopwords.words('english'))

def uploadDataset():
    global filename, dataset, labels
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset))
    label, count = np.unique(dataset['Label'].values.ravel(), return_counts = True)
    labels = ['Normal', 'SQL Injection', 'XML/JSON Injection']
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Dataset Class Label Graph")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
            
def processDataset():
    global dataset, X, Y
    global X_train, X_test, y_train, y_test, vectorizer
    text.delete('1.0', END)

    dataset['Label'] = dataset['Label'].astype(str).astype(int)
    vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=300)
    X = vectorizer.fit_transform(dataset['Sentence'].astype('U')).toarray()
    temp = pd.DataFrame(X, columns=vectorizer.get_feature_names())
    text.insert(END,"SQL, JSON & XML statemenets to TF-IDF vector\n\n")
    text.insert(END, str(temp)+"\n\n")
    Y = dataset['Label'].ravel()
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], 30, 10))
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"Dataset Train & Test Split Details\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing  : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, y_test):
    global labels
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    text.update_idletasks()

    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()       

def runVanilla():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    vanilla_model = Sequential()
    vanilla_model.add(Bidirectional(SimpleRNN(100, input_shape=(X_train.shape[1], X_train.shape[2]))))
    vanilla_model.add(Dropout(0.5))
    vanilla_model.add(Dense(100, activation='relu'))
    vanilla_model.add(Dense(y_train.shape[1], activation='softmax'))
    vanilla_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if os.path.exists("model/vanilla_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/vanilla_weights.hdf5', verbose = 1, save_best_only = True)
        hist = vanilla_model.fit(X_train, y_train, batch_size = 16, epochs = 25, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/vanilla_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        vanilla_model = load_model("model/vanilla_weights.hdf5")
    #perform prediction on test data   
    predict = vanilla_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    #calling this function to calculate accuracy and other metrics
    calculateMetrics("Vanilla RNN", predict, y_test1)
    
def runLSTM():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore
    lstm_model = Sequential()
    lstm_model.add(Bidirectional(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]))))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(100, activation='relu'))
    lstm_model.add(Dense(y_train.shape[1], activation='softmax'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if os.path.exists("model/lstm_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
        hist = lstm_model.fit(X_train, y_train, batch_size = 16, epochs = 25, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/lstm_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        lstm_model = load_model("model/lstm_weights.hdf5")
    #perform prediction on test data   
    predict = lstm_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    #calling this function to calculate accuracy and other metrics
    calculateMetrics("LSTM RNN", predict, y_test1)

def runGRU():
    global X_train, y_train, X_test, y_test
    global accuracy, precision, recall, fscore, gru_model
    gru_model = Sequential()
    gru_model.add(Bidirectional(GRU(100, input_shape=(X_train.shape[1], X_train.shape[2]))))
    gru_model.add(Dropout(0.5))
    gru_model.add(Dense(100, activation='relu'))
    gru_model.add(Dense(y_train.shape[1], activation='softmax'))
    gru_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    if os.path.exists("model/gru_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/gru_weights.hdf5', verbose = 1, save_best_only = True)
        hist = gru_model.fit(X_train, y_train, batch_size = 16, epochs = 25, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/gru_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        gru_model = load_model("model/gru_weights.hdf5")
    #perform prediction on test data   
    predict = gru_model.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    #calling this function to calculate accuracy and other metrics
    calculateMetrics("GRU RNN", predict, y_test1)

def graph():
    #comparison graph between all algorithms
    df = pd.DataFrame([['Vanilla RNN','Accuracy',accuracy[0]],['Vanilla RNN','Precision',precision[0]],['Vanilla RNN','Recall',recall[0]],['Vanilla RNN','FSCORE',fscore[0]],
                       ['LSTM RNN','Accuracy',accuracy[1]],['LSTM RNN','Precision',precision[1]],['LSTM RNN','Recall',recall[1]],['LSTM RNN','FSCORE',fscore[1]],
                       ['GRU RNN','Accuracy',accuracy[2]],['GRU RNN','Precision',precision[2]],['GRU RNN','Recall',recall[2]],['GRU RNN','FSCORE',fscore[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', figsize=(6, 3))
    plt.title("All Algorithms Performance Graph")
    plt.show()    
    
def predictInjection():
    vectorizer, labels, gru_model
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    df = pd.read_csv(filename)
    temp = df.values
    X = vectorizer.transform(df['Test_data'].astype('U')).toarray()
    X = np.reshape(X, (X.shape[0], 30, 10))
    predict = gru_model.predict(X)
    predict = np.argmax(predict, axis=1)
    print(predict)
    for i in range(len(predict)):
        text.insert(END,"Test Data = "+str(temp[i])+"\n")
        text.insert(END,"Predicted Injection = "+labels[predict[i]]+"\n\n")


font = ('times', 16, 'bold')
title = Label(main, text='Evaluation of Recurrent Neural Networks for Detecting Injections in API Requests')
title.config(bg='chocolate', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Injection Dataset", command=uploadDataset)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='lawn green', fg='dodger blue')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=700,y=200)
processButton.config(font=font1)

vanillaButton = Button(main, text="Run Vanilla RNN Algorithm", command=runVanilla)
vanillaButton.place(x=700,y=250)
vanillaButton.config(font=font1) 

lstmButton = Button(main, text="Run LSTM Algorithm", command=runLSTM)
lstmButton.place(x=700,y=300)
lstmButton.config(font=font1)

gruButton = Button(main, text="Run GRU Algorithm", command=runGRU)
gruButton.place(x=700,y=350)
gruButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=700,y=400)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Injection from Test Data", command=predictInjection)
predictButton.place(x=700,y=450)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='light salmon')
main.mainloop()
