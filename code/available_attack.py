import networkx as nx
import numpy as np
import pickle
import random
from line import *
import os
import math
os.environ["CUDA_VISIBLE_DEVICES"]="3"
# g = nx.read_gpickle('co-authorship_graph.pkl')
# edges_raw = g.edges(data=True)
# print(edges_raw[1])

# file=open('release-youtube-links.txt','r')
# count=0
# for line in file:
#     print(line)
#     count+=1
#     if count==5:
#         break



def get_adj_mat():
    g = nx.read_gpickle('facebook_graph.pkl')
    num_of_nodes = g.number_of_nodes()
    edges_raw = g.edges(data=True)
    nodes_raw = g.nodes(data=True)
    node_index = {}
    node_index_reversed = {}
    for index, (node, _) in enumerate(nodes_raw):
        node_index[node] = index
        node_index_reversed[index] = node
    A=np.zeros([num_of_nodes,num_of_nodes])
    
    for u, v, w in edges_raw:
        i=node_index[u]
        j=node_index[v]
        weight=w['weight']
        A[i][j]=weight
    return A

            

def get_embedding(file_name):
    #file=open('embedding_second-order.pkl','rb')
    #x=pickle.load(file)   
    #file.close()
    #file=open('context_embedding_second-order.pkl','rb')
    #y=pickle.load(file)
    #file.close()
    x,y=get_line()
    #print(type(x))
    g = nx.read_gpickle(file_name)
    nodes_raw = g.nodes(data=True)
    node_index = {}
    node_index_reversed = {}
    for index, (node, _) in enumerate(nodes_raw):
        node_index[node] = index
        node_index_reversed[index] = node
    X=[]
    Y=[]
    for index in range(0,len(node_index_reversed)):
        X.append(x[node_index_reversed[index]])
        Y.append(y[node_index_reversed[index]])
    X=np.array(X)
    Y=np.array(Y)
    return X,Y


def draw_new_graph(A,node_modify,file_name):
    g=nx.read_gpickle('facebook_graph.pkl')
    nodes_raw = g.nodes(data=True)
    node_index = {}
    node_index_reversed = {}
    for index, (node, _) in enumerate(nodes_raw):
        node_index[node] = index
        node_index_reversed[index] = node
    for pair in node_modify:
        i=pair[0]
        j=pair[1]
        g[node_index_reversed[i]][node_index_reversed[j]]['weight']=A[i][j]
    nx.write_gpickle(g,file_name )

def get_control(A,number):
    control=[]
    rs=random.sample(range(1,len(A)),number)
    for i in rs:
        for j in range(len(A)):
            if A[i][j]!=0:
                control.append((i,j))
    return control    
  




def  attack_target(number_of_node,original_value,X,Y):
    file_name='facebook_graph.pkl'
    A=get_adj_mat()
    control=get_control(A,number_of_node)

    #find two nodes without edge between them, index aim_i aim_j
    aim_i=28
    aim_j=19
    st=0.1
    for loop in range(10):
        symbol=str(number_of_node)
        file_name='facebook_graph%s.pkl'%symbol
        for pair in control:
            i=pair[0]
            j=pair[1]

            D_L_x_i=X[j].reshape(X[j].shape[0],1)  # L dui X[i] qiu dao 
            non_zero_in_line_i=[]
            for k in range(len(A)):
                if A[i][k] !=0:
                    non_zero_in_line_i.append(k)
            count=0
            for k in non_zero_in_line_i:
                if count==0:
                    vector_sum=np.matmul(Y[k].reshape(Y[k].shape[0],1),Y[k].reshape(1,Y[k].shape[0]))
                    count+=1
                else:
                    vector_sum+=np.matmul(Y[k].reshape(Y[k].shape[0],1),Y[k].reshape(1,Y[k].shape[0]))
            vector_sum=np.linalg.inv(vector_sum)
            D_X_i_Z_ij= np.matmul(Y[j].reshape(1,Y[j].shape[0]),vector_sum) # X[i] dui Z[i][j] qiu dao
            D_Z_ij_A_ij=1/A[i][j]  #
            D_L_A = D_Z_ij_A_ij * D_X_i_Z_ij * D_L_x_i
            A[i][j]=A[i][j]-st*(D_L_A[0][0])
        draw_new_graph(A,control,file_name) 
        X,Y=get_embedding(file_name) #get new embedding
    final_value=np.matmul(X,X.T)[aim_i][aim_j]
    f=open('target_result.txt','a')
    f.write(str(original_value)+' '+str(final_value)+' '+str(number_of_node)+' '+str(len(control))+'\n')
    f.close()


def attack_available(number_of_node,original_value):
    A=get_adj_mat()
    contorl=get_control(A,number_of_node)
    aim_i=28
    aim_j=19
    seta=10
    st=0.1
    for pair in contorl:
        i=pair[0]
        j=pair[1]
        A[i][j]=0.5
    symbol=str(number_of_node)
    file_name='facebook_graph%s.pkl'%symbol
    draw_new_graph(A,contorl,file_name)
    for loop in range(10):
        X,Y=get_embedding(file_name)
        tmp=0
        for pair in contorl:
            tmp+=1
            i=pair[0]
            j=pair[1] 
            non_zero_in_line_i=[]
            for k in range(len(A)):
                if A[i][k] !=0:
                    non_zero_in_line_i.append(k)           
            O=np.matmul(X,X.T)
            O_ij=O[i][0]
            L_xij=(-1*X[i][0]*(0.5-O_ij)*seta*math.exp(-1*seta*(original_value-0.5)*(O_ij-0.5)))/((1+math.exp(-1*seta*(original_value-0.5)*(O_ij-0.5)))**2)
            count=0
            for k in non_zero_in_line_i:
                if count==0:
                   vector_sum=Y[k][0]*Y[k]
                   count=1
                else:
                   vector_sum+=Y[k][0]*Y[k]
            X_ij_Z_ij=Y[j][0]/vector_sum[0]
            D_Z_ij_A_ij=1/A[i][j]
            D_L_A=L_xij*X_ij_Z_ij*D_Z_ij_A_ij
            A[i][j]=A[i][j]-st*(D_L_A)
        draw_new_graph(A,contorl,file_name)
    final_value=np.matmul(X,X.T)[aim_i][aim_j]
    f=open('available_result.txt','a')
    f.write(str(original_value)+' '+str(final_value)+' '+str(number_of_node)+' '+str(len(contorl))+'\n')
    f.close()

X,Y=get_embedding('facebook_graph.pkl')

#test=[20,60]
original_value=np.matmul(X,X.T)[28][19]
test=[121]
for i in test:
  #attack_target(i,original_value,X,Y)
  attack_available(i,original_value)
