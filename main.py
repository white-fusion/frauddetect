import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from som import SelfMap
from sklearn.preprocessing import MinMaxScaler

#Data Preprocessing
dataset = pd.read_csv('credit_card_applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Initialization and training
som = SelfMap(x = 12,y = 12, input_len = 15, sigma=1.0, learning_rate= 0.5)
som.random_weights_init(X)
som.train(data = X, num_iteration = 1000)

#Visualizing results
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
from matplotlib.lines import Line2D

f = plt.figure(figsize=(12,12))
ax = f.add_subplot(111)

xx, yy = som.get_euclidean_coordinates()
umatrix = som.distance_map()
weights = som.get_weights()

for i in range(weights.shape[0]):
    for j in range(weights.shape[1]):
        
        wy = yy[(i, j)]*2/np.sqrt(3)*3/4
        
        hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                      facecolor=cm.Greys(umatrix[i, j]), alpha=.4, edgecolor='gray')
        ax.add_patch(hex)

markers = ['o', 's']
colors = ['r', 'g']
for cnt, x in enumerate(X):
    w = som.winner(x)  # getting the winner
    # palce a marker on the winning position for the sample xx
    wx, wy = som.convert_map_to_euclidean(w) 
    wy = wy*2/np.sqrt(3)*3/4
    plt.plot(wx, wy, markers[y[cnt]], markerfacecolor='None',
             markeredgecolor=colors[y[cnt]], markersize=12, markeredgewidth=2)

xrange = np.arange(weights.shape[0])
yrange = np.arange(weights.shape[1])
plt.xticks(xrange-.5, xrange)
plt.yticks(yrange*2/np.sqrt(3)*3/4, yrange)

divider = make_axes_locatable(plt.gca())
ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Greys, 
                            orientation='vertical', alpha=.4)
cb1.ax.get_yaxis().labelpad = 16
cb1.ax.set_ylabel('distance from neurons in the neighbourhood',
                  rotation=270, fontsize=16)
plt.gcf().add_axes(ax_cb)

legend_elements = [Line2D([0], [0], marker='o', color='r', label='approved',
                   markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2),
                   Line2D([0], [0], marker='s', color='g', label='non-approved',
                   markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2)]
ax.legend(handles=legend_elements, bbox_to_anchor=(0.1, 1.08), loc='upper left', 
          borderaxespad=0., ncol=2, fontsize=14)

plt.savefig('resulting_image.png')
plt.show()

#Finding the frauds
#mappings = som.win_map(X)
# frauds = np.concatenate((mappings[(2,6)], mappings[(2,4)]), axis = 0)
# frauds = sc.inverse_transform(frauds)