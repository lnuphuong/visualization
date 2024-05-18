import streamlit as st
import os
import matplotlib.pyplot as plt
import re
import pandas as pd

##########PATH#################### 
@st.cache_data
def load_path():
    return os.getcwd()

path = load_path()
##############LOG##################
def load_logdata(ml_model, security_model, fl_algorithm, dataset, num_clients, straggler_proximal_pairs, path):
    try:
        
        input_mapping = {
        "FHE": "fhe",
        "RSA-AES": "sym",
        "Pascal": "pascal",
        "CIFAR10": "cifar10",
        "FedProx": "fedprox",
        "FedAvg": "fedavg",
        "MobileNet": "mobilenet",
        "ResNet": "resnet"
        }
        selected_options = [input_mapping[security_model], input_mapping[dataset], input_mapping[fl_algorithm], input_mapping[ml_model]]
        selected_options = '_'.join(selected_options)
        #path = os.path.join(os.getcwd(), 'result')
        folder_path = os.path.join(path, selected_options)
        #st.write(folder_path)
        os.chdir(folder_path)
        if fl_algorithm == 'FedProx':
            straggler, proximal = str(straggler_proximal_pairs.split(', ')[0]), str(straggler_proximal_pairs.split(', ')[1])
            log = str(num_clients)+'client_'+straggler.replace('.', '')+ 'prob_'+proximal.replace('.','')+'mu_log.txt'
            log = log.replace('(', '').replace(')','')
            log_path = os.path.join(os.getcwd(), log)  
        else: 
            log = str(num_clients)+'client_'+'log.txt'
            log = log.replace('(', '').replace(')','')
            log_path = os.path.join(os.getcwd(), log)
        #os.chdir(os.path.dirname(os.getcwd()))
        #os.chdir(os.path.dirname(os.getcwd()))
        #st.write(os.getcwd())
        #st.write(path)
        return log_getdata(log_path)
    except FileNotFoundError:
        st.write("Error: No data for this combination generated")
        return None

def log_getdata(log_path):
    with open(log_path, 'r') as f:
        content = f.read()
        #print(content)
    
    distributed_loss = re.findall(r'round \d+: (\d+\.\d+)', content)
    centralized_loss = re.findall(r'round \d+: (\d+\.\d+)', content.split("History (loss, centralized):")[1])
    accuracy_data = re.findall(r'\((\d+), (\d+\.\d+)\)', content.split("History (metrics, centralized):")[1])
    rounds, accuracy_values = zip(*map(lambda x: (int(x[0]), float(x[1])), accuracy_data))
    return distributed_loss, centralized_loss, rounds, accuracy_values

def plot_log(data, Round):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(1, len(data) + 1), data, marker='o')
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Loss")
    ax.grid(True)
    st.pyplot(fig)

###############CLIENT###############
def load_clientdata(ml_model, security_model, fl_algorithm, dataset, num_clients, straggler_proximal_pairs, path):
    try:
        input_mapping = {
        "FHE": "fhe",
        "RSA-AES": "sym",
        "Pascal": "pascal",
        "CIFAR10": "cifar10",
        "FedProx": "fedprox",
        "FedAvg": "fedavg",
        "MobileNet": "mobilenet",
        "ResNet": "resnet"
        }
        selected_options = [input_mapping[security_model], input_mapping[dataset], input_mapping[fl_algorithm], input_mapping[ml_model]]
        selected_options = '_'.join(selected_options)
        #path = os.path.join(os.getcwd(), 'result')
        folder_path = os.path.join(path, selected_options)
        #st.write(folder_path)
        os.chdir(folder_path)
        list_df =[]
        if fl_algorithm == 'FedProx':
            straggler, proximal = str(straggler_proximal_pairs.split(', ')[0]), str(straggler_proximal_pairs.split(', ')[1])
            
            for n in range(num_clients):
                file_name = str(num_clients)+'client_'+straggler.replace('.', '')+ 'prob_'+proximal.replace('.','')+'mu_C' +str(n)
                file_name = file_name.replace('(', '').replace(')','')
                df = client_getdata(file_name)
                list_df.append(df)
          
            st.markdown(f"<h1 style='font-size:20px;'>Memory Usage over Rounds </h1>", unsafe_allow_html=True)
            plot_Allclient(list_df)
        else: 
            for n in range(num_clients):
                file_name = str(num_clients)+'client_C'+str(n)
                file_name = file_name.replace('(', '').replace(')','')
                df = client_getdata(file_name)
                list_df.append(df)
                
            st.markdown(f"<h1 style='font-size:20px;'>All clients Memory Usage over Rounds </h1>", unsafe_allow_html=True)
            plot_Allclient(list_df)
        #os.chdir(os.path.dirname(os.getcwd()))
        #os.chdir(os.path.dirname(os.getcwd()))
        #st.write(os.getcwd())
        #st.write(path)
        return list_df
    except FileNotFoundError:
        st.write("Error: No data for this combination generated")
        return None

def client_getdata(client_path):
    try:
        df = pd.DataFrame(columns = ['TimeStamp', 'MemoryUsage', 'NetSent','NetRecv'])
    
        with open(client_path, 'r') as f:
            content = f.readlines()
            
        for l in content[1:]:
            temp = l.split(',')
            temp = [float(i) for i in temp]
            df.loc[len(df)] = temp
        #st.write(df)
        return df
    except FileNotFoundError:
        st.write("Error: No data for this combination generated")
        return None

def plot_Allclient(list_df):
    fig, ax = plt.subplots(figsize = (20, 10))

    for idx, df in enumerate(list_df):
        ax.plot(df['TimeStamp'], df['MemoryUsage'], label=f'Client {idx}')
    
    ax.set_xlabel('TimeStamp')
    ax.set_ylabel('MemoryUsage')
    ax.set_title('MemoryUsage vs. TimeStamp')
    ax.legend()
    st.pyplot(fig)


def plot_client(n, df):
    st.markdown(f"<h1 style='font-size:20px; Performance of client {n} </h1>", unsafe_allow_html = True)

    fig, ax = plt.subplots(figsize = (20,10))
    ax.plot(df['TimeStamp'], df['MemoryUsage'], label='MemoryUsage')
    ax.plot(df['TimeStamp'], df['NetSent'], label='NetworkUsage')
    
    ax.set_xlabel('TimeStamp')
    ax.set_ylabel('Values')
    ax.set_title('MemoryUsage and NetworkUsage over Time')    
    plt.legend()
    st.pyplot(fig)

 

################################
#################################    
# Sidebar for user inputs
path = os.path.join(path, 'result')
st.write(path)
st.sidebar.title('Choose Parameters')

st.title('Project visualization dashboard')

num_clients = st.sidebar.selectbox('Number of Clients', [2, 4, 8, 16])
dataset = st.sidebar.selectbox('Dataset', ['Pascal', 'CIFAR10'])
ml_model = st.sidebar.selectbox('ML model', ['MobileNet', 'ResNet'])
fl_algorithm = st.sidebar.selectbox('FL Algorithm', ['FedAvg', 'FedProx'])
security = st.sidebar.selectbox('Security', ['FHE','RSA-AES'])

straggler_proximal_pairs = {
    "(0.1, 0.5)": (0.1, 0.5),
    "(0.1, 0.9)": (0.1, 0.9),
    "(0.2, 0.5)": (0.2, 0.5),
    "(0.2, 0.9)": (0.2, 0.9),
    "(0.3, 0.5)": (0.3, 0.9)
}
selected_pair = ('.', '.')
if fl_algorithm == 'FedProx':
    selected_pair = st.sidebar.selectbox('Straggler probability, Proximal mu', list(straggler_proximal_pairs.keys()))

if st.sidebar.button("Generate Results"):
    tab1, tab2 = st.tabs(["Overall measure","Client measure"])
    
    with tab1:
        distributed_loss, centralized_loss, rounds, accuracy_values= load_logdata(ml_model=ml_model, security_model=security, fl_algorithm=fl_algorithm, 
                                   dataset=dataset, num_clients=num_clients, straggler_proximal_pairs=selected_pair, path = path)
        st.markdown(f"<h1 style='font-size:20px;'>Distributed loss over Rounds </h1>", unsafe_allow_html=True)
        plot_log(distributed_loss, rounds)
        st.markdown(f"<h1 style='font-size:20px;'>Centralized loss over Rounds </h1>", unsafe_allow_html=True) 
        plot_log(centralized_loss, rounds)
        st.markdown(f"<h1 style='font-size:20px;'>Accuracy over Rounds </h1>", unsafe_allow_html=True) 
        plot_log(accuracy_values, rounds)
    with tab2:
        list_df  = load_clientdata(ml_model=ml_model, security_model=security, fl_algorithm=fl_algorithm, 
                                   dataset=dataset, num_clients=num_clients, straggler_proximal_pairs=selected_pair, path = path)
        for idx, df in enumerate(list_df):
            st.markdown(f"<h1 style='font-size:20px;'> Client {idx} performance </h1>", unsafe_allow_html=True)
            plot_client(idx, df)

        



