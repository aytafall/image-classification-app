
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from PIL import Image, ImageOps
import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# --- Étape 1 : Prétraitement et Modélisation des Données (Graphes Dynamiques) ---
st.title("Application de Classification sur des Graphes Dynamiques")
st.write("Bienvenue dans l'application de classification d'images et de graphes dynamiques. Chargez vos données et laissez l'algorithme classifier automatiquement les objets.")

# --- Chargement des Images ---
st.sidebar.title("Chargement des Images")
uploaded_files = st.sidebar.file_uploader("Chargez une ou plusieurs images (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.write("### Images Chargées")
    images = []
    for file in uploaded_files:
        img = Image.open(file)
        img_gray = ImageOps.grayscale(img)  # Conversion en niveaux de gris
        img_np = np.array(img_gray)  # Conversion en tableau numpy

        # Prétraitement : binarisation
        _, img_binary = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        images.append(img_binary)
        st.image(img_binary, caption=f"Image prétraitée - {file.name}", use_container_width=True, channels="GRAY")

    # --- Prétraitement des Images ---
    st.write("### Prétraitement des Images")
    features = [img.flatten() for img in images]
    features = np.array(features)

    # Vérifier que les données sont valides
    if len(features) > 0:
        # Générer dynamiquement les étiquettes fictives
        y = np.random.randint(0, 2, size=len(features))  # Deux classes : 0 ou 1

        # --- Classification avec K-Moyennes ---
        st.write("### Classification avec K-Moyennes")
        n_clusters = min(len(features), 2)  # Ajuster dynamiquement le nombre de clusters
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(features)
        for i, label in enumerate(labels):
            st.write(f"Image {i+1} - Classe {label}")

        # --- Réduction de Dimension avec PCA ---
        st.write("### Visualisation des Classes avec PCA")
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)

        fig, ax = plt.subplots()
        ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
        ax.set_title("Projection PCA des Images Classifiées")
        st.pyplot(fig)

        # --- Classification avec Naive Bayes ---
        st.write("### Classification avec Naive Bayes")
        nb = GaussianNB()
        nb.fit(features, y)
        pred_nb = nb.predict(features)
        st.write("Prédictions :", pred_nb)
        st.write("Précision :", accuracy_score(y, pred_nb))
        st.write("Score F1 :", f1_score(y, pred_nb))

        # --- Classification avec Réseau de Neurones ---
        st.write("### Classification avec Réseau de Neurones")
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
        mlp.fit(features, y)
        pred_mlp = mlp.predict(features)
        st.write("Prédictions :", pred_mlp)
        st.write("Précision :", accuracy_score(y, pred_mlp))
        st.write("Score F1 :", f1_score(y, pred_mlp))
    else:
        st.error("Les données sont invalides ou vides. Veuillez charger des images valides.")
else:
    st.sidebar.info("Veuillez charger au moins une image pour commencer.")

# --- Étape 2 : Création et Visualisation du Graphe Dynamique ---
G = nx.DiGraph()
G.add_edge("A", "B", weight=2)
G.add_edge("A", "C", weight=3)
G.add_edge("B", "D", weight=4)
G.add_edge("C", "D", weight=1)
G.add_edge("D", "E", weight=5)

st.write("### Graphe Dynamique")
fig, ax = plt.subplots()
nx.draw(G, with_labels=True, ax=ax)
st.pyplot(fig)

# --- Calcul des Plus Courts Chemins ---
st.write("### Calcul des Plus Courts Chemins")
shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G))
st.write("Chemins calculés :", shortest_paths)

# --- Extraction des Caractéristiques des Nœuds ---
def calculate_node_features(graph):
    centrality = nx.degree_centrality(graph)
    clustering = nx.clustering(graph)
    return centrality, clustering

centrality, clustering = calculate_node_features(G)
st.write("### Caractéristiques des Nœuds")
st.write("Centralité :", centrality)
st.write("Clustering :", clustering)

st.write("Merci d'avoir utilisé cette application !")
