# -*- coding: utf-8 -*-
"""//check the visibility of math
Spyder Editor

This is a temporary script file.
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from gensim.models import word2vec 
from sklearn.manifold import TSNE
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

#reading the preprocessed data and extracting each column from the data
df = pd.read_csv('777_QTR_after_data_preprocessing.csv')
FL =  list(map(str, df[:214429]['FLIGHTLEG_ID']))
date = list(df[:214429]['PERIOD_ENDDATE'])
mmsgs = list(df[:214429]['MESSAGE_CODE'])
fde =  list(map(str,df[:214429]['FDE_CODE']))


#FLEET OF 1000 SENTENCES OF FLIGHT LEGS
FL_dict={}
for i in range(10000,18000):
    if FL[i] not in FL_dict:
        FL_dict[FL[i]]=[]
    FL_dict[FL[i]].append(mmsgs[i])
    if(fde[i]=='nan'):
        continue
    FL_dict[FL[i]].append(fde[i][:-2])
print(FL_dict)
print(" ")

#corpus contains only the list of mmsgs and fdes for the word2vec process
corpus= []
for i in range(5000,19000):
    corpus.append(mmsgs[i])
    if(fde[i]=='nan'):
        continue
    corpus.append(fde[i][:-2])
print(corpus)
print(" ")


def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens

tokenized_corpus = tokenize_corpus(corpus)
print(" ")
#print(tokenized_corpus)

import torch
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F
print("VOCABULARY OF FDES AND MMSGS")
FL = [] #vocabulary of unique mmsgs and fdes
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in FL:
            FL.append(token)

word2idx = {w: idx for (idx, w) in enumerate(FL)}
print(word2idx)
idx2word = {idx: w for (idx, w) in enumerate(FL)}

pair_size = len(FL)
print(" ")

def get_word_frequencies(tokenized_corpus):
  frequencies = Counter()
  for sentence in corpus:
    for word in sentence:
      frequencies[word] += 1
  freq = frequencies.most_common()
  return freq

print(get_word_frequencies(tokenized_corpus))
# let mmsg= context and let fde=centre-word
#We can now generate pairs fde and mmsg. Let’s assume context window to be symmetric and equal to 2.
window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
    indices = [word2idx[word] for word in sentence]
    # for each word, threated as center word
    for center_word_pos in range(len(indices)):
        # for each window position
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make soure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array

def get_input_layer(word_idx):
    x = torch.zeros(pair_size).float()
    x[word_idx] = 1.0
    return x

#compiling the training loop

#hidden layer
#w1-weight matrix
embedding_dims = 300
W1 = Variable(torch.randn(embedding_dims, pair_size).float(), requires_grad=True)
#output layer
W2 = Variable(torch.randn(pair_size, embedding_dims).float(), requires_grad=True)

embedding_dims = 300

num_epochs = 100
learning_rate = 0.001

for epo in range(num_epochs):
    loss_val = 0
    for data, target in idx_pairs:
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data[0]
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
    
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')
        
        

num_features = 300    # Word vector dimensionality                      
min_word_count = 3    # 50% of the corpus                    
num_workers = 4       # Number of CPUs
window_size = 500
negative_sampling=5

print(" ")

model = word2vec.Word2Vec(tokenized_corpus, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = window_size, sample = negative_sampling)

print(" ")


# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
#print(model[])
# save model
model.save('model.bin')
# load model
#new_model = Word2Vec.load('model.bin')
#print(new_model)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
print (tfidf_matrix.shape)
#vector representation of AR1


print(" ")
print("vector representation of AR1")
skip_pairs = []
i = 1
word_length = len(words)
while i < word_length:
    for j in range(i+1, word_length):
        if (int(words[j][:2]) == int(words[i-1][:2]) or int(words[j][:2]) == int(words[i][:2])) and words[j][2] == '-' and words[i-1][2] != '-' and words[i][2] != '-':
            skip_pairs.append([words[i-1],words[i],words[j]])
            del words[j]
            word_length -= 1
            break
    i+=1
        
print("Prediction of mmsgs from fdes")  
print("[fi+fj-mj=mi]")      
print(skip_pairs)

count=0
for i in range(len(skip_pairs)):
    b = skip_pairs[i][2]
    
    if int(skip_pairs[i][0][:2]) == int(b[:2]):
        a = skip_pairs[i][0]
        x = skip_pairs[i][1]
    else:
        a = skip_pairs[i][1]
        x = skip_pairs[i][0]
    predicted = model.most_similar([x, b], [a])[0][0]
    print(" {} is to  {} as {} is to {} ".format(a, b, x, predicted))
    if int(x[:2]) == int(predicted[:2]):
         count+=1
        
#to find the success rate
success_rate = (count/len(skip_pairs))*100
print("The success rate is {0} %".format(success_rate))        

#finding the success rate for each chapter list
print(" ")
print(" SUCCESS RATES FOR CHAPTER LISTS")
labels = []
success=[]
for k in range (20,80):
    count=0
    success_rate=0
    for i in range (len(skip_pairs)):
        b = skip_pairs[i][2]
    
        if int(skip_pairs[i][0][:2]) == int(b[:2]):
            a = skip_pairs[i][0]
            x = skip_pairs[i][1]
        else:
            a = skip_pairs[i][1]
            x = skip_pairs[i][0]
            predicted = model.most_similar([x, b], [a])[0][0]
            
        if(int(a[:2]) == k or int(x[:2]) == int(predicted[:2]) == k):
            count+=1
            success_rate = (count/len(skip_pairs))*100
            
        else:
           continue
    if(success_rate == 0):
        continue
    else:
        print("The success rate for CHAPTER-LIST {0}---->{1}%".format(k,success_rate))
        labels.append(k)
        success.append(success_rate)



import random

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color


colors=[]
for i in range (len(labels)) :
    colors.append(generate_new_color(colors,pastel_factor=0.9))
fig1, ax1 = plt.subplots()
patches, texts = plt.pie(success, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.title("SUCCESS RATES FOR THE CHAPTER LISTS")
plt.axis('equal')
plt.tight_layout()
plt.show()





#scatter plot

import seaborn as sns 

corpus = success #not sure the exact api
emb_tuple = tuple([model[v] for v in corpus])
X = np.vstack(emb_tuple)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

colors=[]
for i in range (len(labels)) :
    colors.append(generate_new_color(colors,pastel_factor=0.9))        
sns.set_context("poster")
fig, ax = plt.subplots(figsize=(10,10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=.6, s=60)
plt.legend(patches, labels, loc="best")





"""

print("________________________________________________________________________")

print("")
print("vector representation of AR2")
cbow_pairs = []
i = 1
word_length = len(words)
while i < word_length:
    for j in range(i+1, word_length):
        if (int(words[j][:2]) == int(words[i-1][:2]) or int(words[j][:2]) == int(words[i][:2])) and words[j][2] != '-'and words[i][2] == '-' and words[i-1][2] ==  '-':
            cbow_pairs.append([words[i-1],words[i],words[j]])
            del words[j]
            word_length -= 1
            break
    i+=1
        
print("Prediction of fde from mmsgs")        
print(cbow_pairs)

count=0
for i in range(len(cbow_pairs)):
    b = cbow_pairs[i][2]
    if int(cbow_pairs[i][0][:2]) == int(b[:2]):
        a = cbow_pairs[i][0]
        x = cbow_pairs[i][1]
    else:
        a = cbow_pairs[i][1]
        x = cbow_pairs[i][0]
    predicted = model.most_similar([x, b], [a])[0][0]
    print(" {} is to  {} as {} is to {} ".format(a, b, x, predicted))
    if int(x[:2]) == int(predicted[:2]):
         count+=1
 #to find the success rate   
success_rate = (count/len(cbow_pairs))*100

print("The success rate is {0} %".format(success_rate))           


#finding success rate for all chapter lists
#finding the success rate for each chapter list
print(" ")
print(" SUCCESS RATES FOR CHAPTER LISTS")
labels=[]
success=[]
for k in range (20,80):
    count=0
    success_rate=0
    for i in range (len(cbow_pairs)):
        b = cbow_pairs[i][2]
    
        if int(cbow_pairs[i][0][:2]) == int(b[:2]):
            a = cbow_pairs[i][0]
            x = cbow_pairs[i][1]
        else:
            a = cbow_pairs[i][1]
            x = cbow_pairs[i][0]
            predicted = model.most_similar([x, b], [a])[0][0]
            
        if(int(a[:2]) == k or int(x[:2]) == int(predicted[:2]) == k):
            count+=1
            success_rate = (count/len(cbow_pairs))*100
        else:
           continue
    if(success_rate == 0):
        continue
    else:
        print("The success rate for CHAPTER-LIST {0}---->{1}%".format(k,success_rate))
        labels.append(k)
        success.append(str(success_rate))        
colors=[]
for i in range (len(labels)) :
    colors.append(generate_new_color(colors,pastel_factor=0.9))
fig1, ax1 = plt.subplots()
patches, texts = plt.pie(success, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.title("SUCCESS RATES FOR THE CHAPTER LISTS")
plt.axis('equal')
plt.tight_layout()
plt.show()


"""


