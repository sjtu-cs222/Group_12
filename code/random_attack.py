import networkx as nx
import numpy as np
import pickle
import random
from line import *
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
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
  




def  attack_one(number_of_node):
    A=get_adj_mat()
    control=get_control(A,number_of_node)
    symbol=str(number_of_node)
    file_name='facebook_graph%s.pkl'%symbol
    for pair in control:
        i=pair[0]
        j=pair[1]
        A[i][j]=random.randint(1,35)
    draw_new_graph(A,control,file_name)
    X,Y=get_embedding(file_name)
    aim_i=28
    aim_j=19
    final_value=np.matmul(X,X.T)[aim_i][aim_j]
    f=open('result.txt','a')
    f.write(str(final_value)+' '+str(number_of_node)+' '+str(len(control))+'\n')
    f.close()



test=[121, 363, 605, 848, 1090]
for i in test:
  attack_one(i)
