import streamlit as st
import time
import model

st.set_page_config(layout="wide")

if "init" not in st.session_state:
    st.session_state["init"] = False

if "training" not in st.session_state:
    st.session_state["training"] = "None"

if "nn_result" not in st.session_state:
    st.session_state["nn_result"] = "None"

if "ac_result" not in st.session_state:
    st.session_state["ac_result"] = "None"

if "km_result" not in st.session_state:
    st.session_state["km_result"] = "None"

if not(st.session_state.init):
    st.write("It may takes a while to load the dataset.")

DataFrame,X,y = model.Read_Training_Dataset()
if(DataFrame is None):
    st.error("Error on handling read csv:" + str(X))

if not(st.session_state.init):
    st.session_state.init = True
    st.rerun()

max_len = 128
nn_hidden = [2]
#SideBar
Sidebar = st.sidebar
with Sidebar:
    if st.session_state.training == "None":
        embed_dim = st.selectbox("Embedding Dim",[16,32,64,128,256,512,1024])
        train_data_ratio = st.selectbox("Train Data Ratio",[0.5,0.6,0.7,0.8,0.9,0.95])
        training_epoch = st.selectbox("Training epoch",[1,5,30,50,80,100,300,500,1000])
        kmeans_apply_epoch = st.selectbox("KMeans apply epoch",[1,5,10,12,20])
    else:
        embed_dim = st.selectbox("Embedding Dim",[16,32,64,128,256,512,1024],disabled=True)
        train_data_ratio = st.selectbox("Train Data Ratio",[0.5,0.6,0.7,0.8,0.9,0.95],disabled=True)
        training_epoch = st.selectbox("Training epoch",[1,5,30,50,80,100,300,500,1000],disabled=True)
        kmeans_apply_epoch = st.selectbox("KMeans apply epoch",[1,5,10,12,20],disabled=True)

st.title("Train the classification model and view the performance.")

st.header("Deep nn")
left,right = st.columns([1,6])
nn_hp = left.container(border=True,height=500)
_list_nn_hp = [0,0,0]

with nn_hp:
    st.write("Deep nn")
    st.write("Hyperparameters")
    if st.session_state.training == "None":
        _list_nn_hp[0] = st.selectbox("First Layer",[None,2,4,8,16,32,64,128,256,512])
        _list_nn_hp[1] = st.selectbox("Second Layer",[None,2,4,8,16,32,64,128,256,512])
        _list_nn_hp[2] = st.selectbox("Third Layer",[None,2,4,8,16,32,64,128,256,512])
    else:
        _list_nn_hp[0] = st.selectbox("First Layer",[None,2,4,8,16,32,64,128,256,512],disabled=True)
        _list_nn_hp[1] = st.selectbox("Second Layer",[None,2,4,8,16,32,64,128,256,512],disabled=True)
        _list_nn_hp[2] = st.selectbox("Third Layer",[None,2,4,8,16,32,64,128,256,512],disabled=True)

list_nn_hp = []
for i in range(3):
    if _list_nn_hp[i] is not None and _list_nn_hp[i] != 0:
        list_nn_hp.append(_list_nn_hp[i])
nn_hidden = list_nn_hp

nn_struct = f"Structure: Embedding{embed_dim} -> Flatten -> {max_len * embed_dim} -> "
for i in list_nn_hp:
    nn_struct += str(i) + " -> "
nn_struct += "2"

nn_container = right.container(border=True,height=500)
with nn_container:
    st.header("Input the hyperparameters of deep nn below and click start training button.")
    st.write(nn_struct)
    st.divider()
    nn_console = st.container(border=True,height=100)
    nn_progress = st.progress(0,"Training Progress")
    col1,col2,col3 = st.columns(3)

    if st.session_state.training == "None":
        nn_train_button = col1.button("Start Training",key="nn_1")
    elif st.session_state.training == "nn_training":
        nn_train_button = col1.button("Training",key="nn_2",disabled=True)
    else:
        nn_train_button = col1.button("Waiting",key="nn_3",disabled=True)

    nn_train_status = col2.empty()

    if st.session_state.nn_result == "None":
        nn_train_result = col3.badge("Test Error:"+str(st.session_state.nn_result),color="gray")
    else:
        nn_train_result = col3.badge("Test Error:"+str(st.session_state.nn_result),color="green")

nn_console.write("Output during Training:")
key = tuple(["nn"]) + tuple(nn_hidden)+tuple([embed_dim,max_len,train_data_ratio,training_epoch])
if key in st.session_state:
    nn_progress.progress(1.,"Cache Found")
    v,T = st.session_state[key]
    for t in T:
        nn_console.write(t)

if nn_train_button:
    st.session_state.training = "nn_training"
    st.rerun()

if st.session_state.training == "nn_training":
    nn_train_status.badge("Training Status:Training",color="yellow")
    st.session_state.nn_result = model.Evaluate_NN(X,y,nn_hidden,nn_console,embed_dim,max_len,_progress_bar=nn_progress,train_to_test_data_ratio=train_data_ratio,training_epoch=training_epoch)
    st.session_state.training = "None"
    st.rerun()
else:
    nn_train_status.badge("Training Status:Waiting",color="gray")

st.header("Autoencoder + kmeans")
left,right = st.columns([1,6])
ac_hp = left.container(border=True,height=500)
hidden_dim = 0

with ac_hp:
    st.write("Autoencoder")
    st.write("Hyperparameters")
    if st.session_state.training == "None":
        hidden_dim = st.selectbox("Hidden dim",[2,4,8,16,32,64,128,256,512])
    else:
        hidden_dim = st.selectbox("Hidden dim",[2,4,8,16,32,64,128,256,512],disabled=True)

ac_struct = f"Structure: Embedding{embed_dim} -> Flatten -> {max_len * embed_dim} -> {hidden_dim} -> {max_len}" 

ac_container = right.container(border=True,height=500)
with ac_container:
    st.header("Input the hyperparameters of autoencoder below and click start training button.")
    st.write(ac_struct)
    st.divider()
    ac_console = st.container(border=True,height=100)
    ac_progress = st.progress(0,"Training Progress")
    col1,col2,col3 = st.columns(3)

    if st.session_state.training == "None":
        ac_train_button = col1.button("Start Training",key="ac_1")
    elif st.session_state.training == "ac_training":
        ac_train_button = col1.button("Training",key="ac_2",disabled=True)
    else:
        ac_train_button = col1.button("Waiting",key="ac_3",disabled=True)

    ac_train_status = col2.empty()

    if st.session_state.ac_result == "None":
        ac_train_result = col3.badge("Test Error:"+str(st.session_state.ac_result),color="gray")
    else:
        ac_train_result = col3.badge("Test Error:"+str(st.session_state.ac_result),color="green")

ac_console.write("Output during Training:")
key = ("ac",) + (hidden_dim,) + (embed_dim, max_len, kmeans_apply_epoch, train_data_ratio, training_epoch)
if key in st.session_state:
    ac_progress.progress(1.,"Cache Found")
    v,T = st.session_state[key]
    for t in T:
        ac_console.write(t)

if ac_train_button:
    st.session_state.training = "ac_training"
    st.rerun()

if st.session_state.training == "ac_training":
    ac_train_status.badge("Training Status:Training",color="yellow")
    st.session_state.ac_result = model.Evaluate_AC(X,y,hidden_dim,ac_console,embed_dim,kmeans_apply_epoch,_progress_bar=ac_progress,train_to_test_data_ratio=train_data_ratio,training_epoch=training_epoch)
    st.session_state.training = "None"
    st.rerun()
else:
    ac_train_status.badge("Training Status:Waiting",color="gray")

st.header("KMeans directly")
km_container = st.container(border=True,height=200)

with km_container:
    km_console = st.container(border=True,height=100)
    col1,col2,col3 = st.columns(3)
    if st.session_state.training == "None":
        km_train_button = col1.button("Start Fit_Transform",key="km_1")
    elif st.session_state.training == "km_training":
        km_train_button = col1.button("Working",key="km_2",disabled=True)
    else:
        km_train_button = col1.button("Waiting",key="km_3",disabled=True)

    km_train_status = col2.empty()

    if st.session_state.km_result == "None":
        km_train_result = col3.badge("Test Error:"+str(st.session_state.km_result),color="gray")
    else:
        km_train_result = col3.badge("Test Error:"+str(st.session_state.km_result),color="green")

if "kmeans" in st.session_state:
    v,T = st.session_state["kmeans"]
    for t in T:
        km_console.write(t)
    
if km_train_button:
    st.session_state.training = "km_training"
    st.rerun()

if st.session_state.training == "km_training":
    km_train_status.badge("Training Status:Training",color="yellow")
    st.session_state.km_result = model.Evaluate_KM(X,y,km_console,kmeans_apply_epoch,train_to_test_data_ratio=train_data_ratio)
    st.session_state.training = "None"
    st.rerun()
else:
    km_train_status.badge("Training Status:Waiting",color="gray")    

st.divider()
st.write("Models are trained using imdb review dataset. You may check the dataset on https://www.kaggle.com/datasets/utathya/imdb-review-dataset")