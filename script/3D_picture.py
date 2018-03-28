#coding:utf-8
import networkx as nx
import matplotlib.pyplot as plt
#此段代码解决 1.matplotlib中文显示问题 2 '-'显示为方块问题
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
def show(G,pos,title=None,photo_name='picture'):
    e_1 =[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] ==1] # 普通边
    e_2 =[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] ==0] # 利用的边
    # Draw nodes
    nx.draw_networkx_nodes(G,pos,node_size=300, node_color='orange')
    # Draw Edges
    nx.draw_networkx_edges(G,pos,edgelist=e_1,width=1, alpha = 1,edge_color='g',style='dashed')
    nx.draw_networkx_edges(G,pos,edgelist=e_2, width=3,alpha=0.6,edge_color='b')
    edge_labels =dict([((u, v), d['weight']) for u, v, d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(G,pos,font_size=10)
    plt.title(title)
    plt.axis('off')
    plt.savefig(photo_name)
    plt.show()
    
#%% read in data - use a pandas dataframe just for convenience
import pandas as pd
data = pd.read_table("~/Desktop/Project/data/melbourne_graph.txt",
                     sep = " ",
                     header = None, 
                     names = ['vx', 'vy', 'weight'])

# %% use network x to prepare dictionary structure which can be fed in to the 
# dijkstra function
import networkx as nx
graph = nx.from_pandas_edgelist(data, 'vx', 'vy', 'weight')
# graph_nodes = graph.nodes()
graph_dict = nx.to_dict_of_dicts(graph)
G = nx.Graph(graph_dict)
pos = nx.shell_layout(G) 
#show(G,pos)
#print data

# Generate 3D scatter pitcture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = Axes3D(fig)

#ax.scatter(data['vx'], data['vy'], data['weight'],'b.')
#ax.scatter(class1[:,0], class1[:,1], class1[:,2],'r.')
#plt.show()

import pandas as pd

sort_data = data.sort_values(by=['weight'])
print sort_data
temp = sort_data['weight']
plt.scatter(range(len(temp)),temp)
plt.show()