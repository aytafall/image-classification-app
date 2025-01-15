
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from PIL import Image
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# --- Étape 1 : Prétraitement et Modélisation des Données (Graphes Dynamiques) ---
st.title("Application de Classification sur des Graphes Dynamiques")
st.write("Cette application permet de charger des données, de modéliser un graphe dynamique, et de classifier automatiquement les objets.")

# Création d'un graphe orienté
G = nx.DiGraph()
G.add_edge("A", "B", weight=2)
G.add_edge("A", "C", weight=3)
G.add_edge("B", "D", weight=4)
G.add_edge("C", "D", weight=1)
G.add_edge("D", "E", weight=5)

# Visualisation du graphe
st.write("### Graphe Dynamique")
fig, ax = plt.subplots()
nx.draw(G, with_labels=True, ax=ax)
st.pyplot(fig)

# --- Étape 2 : Algorithme Dynamique de Plus Courts Chemins ---
st.write("### Calcul des Plus Courts Chemins")
shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G))
st.write("Chemins calculés :", shortest_paths)

# --- Étape 3 : Extraction des Caractéristiques des Nœuds ---
def calculate_node_features(graph):
    centrality = nx.degree_centrality(graph)
    clustering = nx.clustering(graph)
    return centrality, clustering

centrality, clustering = calculate_node_features(G)
st.write("### Caractéristiques des Nœuds")
st.write("Centralité :", centrality)
st.write("Clustering :", clustering)

# --- Étape 4 : Classification avec Différents Algorithmes ---
# Simulation de données de classification
X = np.array([[centrality[node], clustering[node]] for node in G.nodes()])
y = np.array([0, 1, 1, 0, 1])  # Labels fictifs

# K-Moyennes
st.write("### Classification avec K-Moyennes")
kmeans = KMeans(n_clusters=2)
labels_kmeans = kmeans.fit_predict(X)
st.write("Labels prédits :", labels_kmeans)

# Naive Bayes
st.write("### Classification avec Naive Bayes")
nb = GaussianNB()
nb.fit(X, y)
pred_nb = nb.predict(X)
st.write("Prédictions :", pred_nb)
st.write("Précision :", accuracy_score(y, pred_nb))
st.write("Score F1 :", f1_score(y, pred_nb))

# Réseau de Neurones
st.write("### Classification avec Réseau de Neurones")
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
mlp.fit(X, y)
pred_mlp = mlp.predict(X)
st.write("Prédictions :", pred_mlp)
st.write("Précision :", accuracy_score(y, pred_mlp))
st.write("Score F1 :", f1_score(y, pred_mlp))

# --- Étape 5 : Visualisation avec PCA ---
st.write("### Réduction de Dimension avec PCA")
pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)

fig2, ax2 = plt.subplots()
ax2.scatter(reduced_X[:, 0], reduced_X[:, 1], c=y, cmap='viridis')
ax2.set_title("Projection PCA des Nœuds Classifiés")
st.pyplot(fig2)

st.write("Merci d'avoir utilisé cette application !")
