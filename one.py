import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

data = {
    'Product': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E', 'Product F'],
    'Price': [500, 550, 1200, 1100, 300, 350],
    'Rating': [4.5, 4.3, 4.8, 4.7, 3.9, 4.0],
    'Sales': [1200, 1100, 800, 850, 1500, 1400]
}

df = pd.DataFrame(data)

X = df[['Price', 'Rating', 'Sales']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(Z, labels=df['Product'].values, leaf_rotation=45)
plt.title('Dendrogram of Products')
plt.xlabel('Products')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()