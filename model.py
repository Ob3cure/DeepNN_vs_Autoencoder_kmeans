import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
import re
import torch.nn.functional as F
from sklearn.cluster import KMeans
import os
import time
import streamlit as st
import kagglehub

@st.cache_data
def Read_Training_Dataset():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        #imdb = pd.read_csv(current_dir + "\\Training_Dataset\\imdb_master.csv", encoding='latin-1')
        path = kagglehub.dataset_download("utathya/imdb-review-dataset",force_download=True)
        imdb = pd.read_csv(path + "\\imdb_master.csv", encoding='latin-1')

        y = imdb["label"].map({"neg":0,"pos":1})
        y = y.loc[~(y.isna())]
        y = y.to_numpy()
        _X = imdb.loc[~(imdb["label"].map({"neg":0,"pos":1}).isna()),"review"]

        tokenizer = SimpleTokenizer(max_length=128)
        input_ids, _ = tokenizer.fit_transform(_X)
        return imdb,input_ids,y
    except Exception as e:
        print(f"Error on reading dataset:{e}")
        return None,e,None


class SimpleTokenizer:
    def __init__(self, vocab_size=10000, max_length=1000, oov_token='<UNK>', pad_token='<PAD>'):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.word2idx = {}
        self.idx2word = {}
        self._fitted = False

    def _tokenize_text(self, text):
        return re.findall(r"[A-Za-z']+", str(text).lower())

    def fit(self, texts):
        word_freq = Counter()
        for text in texts:
            words = self._tokenize_text(str(text))
            word_freq.update(words)
        most_common = word_freq.most_common(self.vocab_size - 2) 
        vocab = [self.pad_token, self.oov_token] + [word for word, _ in most_common]
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self._fitted = True
        return self

    def transform(self, texts):
        if not self._fitted:
            raise ValueError("Call fit before apply transform is required.")
        sequences = []
        for text in texts:
            words = self._tokenize_text(str(text))
            ids = [self.word2idx.get(w, self.word2idx[self.oov_token]) for w in words]
            ids = ids[:self.max_length]
            pad_len = self.max_length - len(ids)
            input_ids = ids + [self.word2idx[self.pad_token]] * pad_len
            attention_mask = [1] * len(ids) + [0] * pad_len
            sequences.append((input_ids, attention_mask))
        input_ids = np.array([seq[0] for seq in sequences], dtype=np.int64)
        attention_mask = np.array([seq[1] for seq in sequences], dtype=np.int64)
        return input_ids, attention_mask

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

def training_text():
    str = np.array(["Working very hard...",
                        "Training...",
                        "Progressing...",
                        "The model is learning...",
                        "Walking down hill...",
                        "Training data delivering"])
    rand_str = np.random.choice(str)
    return rand_str

def train_xentropy(model,optimizer,train_loader,output_container,progress_bar=None,log=[],epoch=30):
    start = time.perf_counter()
    for i in range(epoch):
        loss_pr = 0
        for X,y in train_loader:
            loss = F.cross_entropy(model(X),y)
            loss.backward()
            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad()
                loss_pr += loss.item()
        output_container.write("loss in epoch"+str(i)+":"+str(loss_pr))
        log.append("loss in epoch"+str(i)+":"+str(loss_pr))
        if(progress_bar is not None):
            progress_bar.progress((i + 1)/epoch,training_text())
    end = time.perf_counter()
    output_container.write("Training terminated.")
    log.append("Training terminated.")
    output_container.write(f"Time spent:{(end - start) * 1000} ms")
    log.append(f"Time spent:{(end - start) * 1000} ms")

def train_mse(model, optimizer, train_loader, output_container, progress_bar=None, log=[], epoch=30):
    start = time.perf_counter()
    for i in range(epoch):
        loss_pr = 0
        for X, y in train_loader:
            loss = -F.cosine_similarity(model(X), y).mean()
            loss.backward()
            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad()
                loss_pr += loss.item()
        msg = f"loss in epoch {i}: {loss_pr}"
        output_container.write(msg)
        log.append(msg)
        if progress_bar is not None:
            progress_bar.progress((i + 1) / epoch, training_text())
    end = time.perf_counter()
    end_msg = "Training terminated."
    output_container.write(end_msg)
    log.append(end_msg)
    time_msg = f"Time spent: {(end - start) * 1000} ms"
    output_container.write(time_msg)
    log.append(time_msg)

def autoencoder_kmeans_eval(autoencoder,X,y,epoch=10):
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    mse = 0
    
    for i in range(epoch):
        layer = autoencoder[:3] #embedding -> flattening -> linear ->...
        encoded = layer(X)
        labels = KMeans(2).fit(encoded).predict(encoded)
        mse += ((labels - y) ** 2).mean()

    mse = mse / epoch
    for param in autoencoder.parameters():
        param.requires_grad = True
    return labels,mse

def kmeans_eval(X,y,epoch=10):
    mse = 0

    for i in range(epoch):
        mse += ((KMeans(2).fit(X).predict(X) - y) ** 2).mean()

    return mse / epoch

def Evaluate_NN(data,label,nn_hidden,_output_container,embed_dim=32,max_len=128,_progress_bar=None,voc_size=10000,train_to_test_data_ratio=0.8,training_epoch=1):
    output_container = _output_container
    progress_bar = _progress_bar
    log = []
    dict = st.session_state

    key = tuple(["nn"]) + tuple(nn_hidden)+tuple([embed_dim,max_len,train_to_test_data_ratio,training_epoch])
    if key not in dict:
        dict[key] = ""
    else:
        v,T = dict[key]
        return v
    
    split = round(train_to_test_data_ratio * 50000)
    output_container.write("There are "+str(split)+" training data and "+str(50000-split)+" testing data")
    log.append("There are "+str(split)+" training data and "+str(50000-split)+" testing data")
    
    train_data = data[:split] 
    train_label = label[:split]
    test_data = data[split:] 
    test_label = label[split:]
    
    l = len(nn_hidden)
    deep_nn = nn.Sequential()
    deep_nn.append(nn.Embedding(voc_size,embed_dim))
    deep_nn.append(nn.Flatten())
    if(l > 0):
        deep_nn.append(nn.Linear(max_len*embed_dim,nn_hidden[0]))
        deep_nn.append(nn.ReLU())
        for i in range(l-1):
            deep_nn.append(nn.Linear(nn_hidden[i],nn_hidden[i+1]))
            deep_nn.append(nn.ReLU())
        deep_nn.append(nn.Linear(nn_hidden[-1],2))
    else:
        deep_nn.append(nn.ReLU())
        deep_nn.append(nn.Linear(max_len*embed_dim,2))      

    optimizer = torch.optim.NAdam(deep_nn.parameters())
    dataset = torch.utils.data.TensorDataset(torch.tensor(train_data),torch.tensor(train_label,dtype=torch.long))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    train_xentropy(deep_nn,optimizer,dataloader,output_container,progress_bar,log=log,epoch=training_epoch)

    accuracy_nn = ((torch.argmax(deep_nn(torch.tensor(test_data)),dim=-1) - torch.tensor(test_label,dtype=torch.float32)) ** 2).mean().item()
    output_container.write("--------------------------------------")
    log.append("--------------------------------------")

    output_container.write("nn has error rate:"+str(accuracy_nn))
    log.append("Deep nn has error rate:"+str(accuracy_nn))

    output_container.write("With hidden dim:")
    log.append("With hidden dim:")

    if(l > 0):
        output_container.write(str(max_len*embed_dim) + " -> "+str(nn_hidden[0]))
        log.append(str(max_len*embed_dim) + " -> "+str(nn_hidden[0]))

        for i in range(len(nn_hidden) - 1):
            output_container.write((nn_hidden[i])+" -> "+str(nn_hidden[i+1]))
            log.append((nn_hidden[i])+" -> "+str(nn_hidden[i+1]))

        output_container.write(str(nn_hidden[-1])+" -> 2")
        log.append(str(nn_hidden[-1])+" -> 2")
    else:
        output_container.write(str(max_len*embed_dim) + " -> 2")
        log.append(str(max_len*embed_dim) + " -> 2")

    output_container.write("With training epoch "+str(training_epoch))
    log.append("With training epoch "+str(training_epoch))

    output_container.write("--------------------------------------")
    log.append("--------------------------------------")

    dict[key] = accuracy_nn,log
    return accuracy_nn

def Evaluate_AC(data, label, ac_hidden, _output_container,
                embed_dim=32, kmeans_apply_epoch=10, voc_size=10000, max_len=128,
                _progress_bar=None, train_to_test_data_ratio=0.8, training_epoch=1):
    output_container = _output_container
    progress_bar = _progress_bar
    log = []

    cache = st.session_state
    key = ("ac",) + (ac_hidden,) + (embed_dim, max_len, kmeans_apply_epoch, train_to_test_data_ratio, training_epoch)
    if key in cache:
        accuracy_ac, cached_log = cache[key]
        for line in cached_log:
            output_container.write(line)
        if progress_bar is not None:
            progress_bar.progress(1.0, "Using cached result")
        return accuracy_ac

    split = round(train_to_test_data_ratio * 50000)
    msg = f"There are {split} training data and {50000-split} testing data"
    output_container.write(msg)
    log.append(msg)

    train_data = data[:split]
    train_label = label[:split]
    test_data = data[split:]
    test_label = label[split:]

    autoencoder = nn.Sequential(
        nn.Embedding(voc_size, embed_dim),
        nn.Flatten(),
        nn.Linear(max_len * embed_dim, ac_hidden),
        nn.ReLU(),
        nn.Linear(ac_hidden, max_len)
    )

    optimizer = torch.optim.NAdam(autoencoder.parameters())
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_data), torch.tensor(train_data, requires_grad=False)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    train_mse(autoencoder, optimizer, dataloader, output_container,
              progress_bar, log=log, epoch=training_epoch)

    _, accuracy_ac = autoencoder_kmeans_eval(autoencoder, torch.tensor(test_data),
                                             test_label, epoch=kmeans_apply_epoch)

    # 输出最终结果
    sep = "--------------------------------------"
    output_container.write(sep)
    log.append(sep)
    result_msg = f"Autoencoder_kmeans has average error rate: {accuracy_ac}"
    output_container.write(result_msg)
    log.append(result_msg)
    output_container.write(f"With hidden dim {ac_hidden}")
    log.append(f"With hidden dim {ac_hidden}")
    output_container.write(f"With kmeans apply epoch {kmeans_apply_epoch}")
    log.append(f"With kmeans apply epoch {kmeans_apply_epoch}")
    output_container.write(f"With training epoch {training_epoch}")
    log.append(f"With training epoch {training_epoch}")
    output_container.write(sep)
    log.append(sep)

    # 存入缓存
    cache[key] = (accuracy_ac, log)
    return accuracy_ac

def Evaluate_KM(data,label,output_container,kmeans_apply_epoch=10,train_to_test_data_ratio=0.8):
    log = []
    if "kmeans" in st.session_state:
        v,T = st.session_state["kmeans"]
        return v
    else:
        st.session_state["kmeans"] = ""

    split = round(train_to_test_data_ratio * 50000)
    output_container.write("There are "+str(split)+" training data and "+str(50000-split)+" testing data")
    log.append("There are "+str(split)+" training data and "+str(50000-split)+" testing data")

    test_data = data[split:] 
    test_label = label[split:]
    
    accuracy_km = kmeans_eval(test_data,test_label,epoch=kmeans_apply_epoch)

    output_container.write("--------------------------------------")
    log.append("--------------------------------------")

    output_container.write("KMeans has average error rate:"+str(accuracy_km))
    log.append("KMeans has average error rate:"+str(accuracy_km))

    output_container.write("With kmeans apply epoch "+str(kmeans_apply_epoch))
    log.append("With kmeans apply epoch "+str(kmeans_apply_epoch))

    output_container.write("--------------------------------------")
    log.append("--------------------------------------")

    st.session_state["kmeans"] = accuracy_km,log
    return accuracy_km