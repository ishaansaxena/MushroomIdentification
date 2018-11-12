import data
import numpy as np
import project
from sklearn.decomposition import PCA
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
   [X,y] = data.load();
   print("X:",X);
   print("y:",y);
   X = X.values;
   pca = PCA(n_components = 2);
   X_reduced = pca.fit_transform(X);
   
   poisonous = list(np.where(y==1)[0]);
   edible = list(np.where(y==0)[0]);

   print(X_reduced);
   #2d figure
   pp.figure();
   pp.plot(X_reduced[poisonous,0], X_reduced[poisonous,1], 'bo',markersize = 1) # b=blue, o=circle
   pp.plot(X_reduced[edible,0], X_reduced[edible,1], 'ro',markersize = 1) # r=red, o=circle
   pp.xlabel('PCA feature 0')
   pp.ylabel('PCA feature 1')
   pp.show() # This command will open the figure, and wait

   #3d figure (not really helpful)
   fig = pp.figure();
   ax = fig.add_subplot(111, projection='3d')
   ax.scatter(X_reduced[poisonous,0], X_reduced[poisonous,1],y[poisonous],'bo');
   ax.scatter(X_reduced[edible,0], X_reduced[edible,1],y[edible], 'ro');

   ax.set_xlabel('PCA feature 0');
   ax.set_ylabel('PCA feature 1');
   ax.set_zlabel('Class');

   pp.show();
