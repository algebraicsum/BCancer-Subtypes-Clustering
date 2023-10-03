from sklearn.manifold import TSNE
import pickle 
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
data = pickle.load(open(r"D:\Kuliah\S7\Internship\fitur extracted\feat.pkl", 'rb'))

filenames = np.array(list(data.keys()))

feat = np.array(list(data.values()))
feat = feat.reshape(-1,128)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(feat)
tsne.kl_divergence_


plt.scatter(x=X_tsne[:,0],y=X_tsne[:,1])