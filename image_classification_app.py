
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
        try:
            img = Image.open(file).convert("RGB")
            img_gray = ImageOps.grayscale(img)  # Conversion en niveaux de gris
            img_np = np.array(img_gray)  # Conversion en tableau numpy

            # Prétraitement : redimensionnement et binarisation
            img_resized = cv2.resize(img_np, (100, 100), interpolation=cv2.INTER_AREA)
            _, img_binary = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            images.append(img_binary)
            st.image(img_binary, caption=f"Image prétraitée - {file.name}", use_container_width=True, channels="GRAY")
        except Exception as e:
            st.error(f"Erreur lors du traitement de l'image {file.name}: {str(e)}")

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
        n_components = min(len(features), len(features[0]), 2)  # Ajuster le nombre de composants
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features)

        fig, ax = plt.subplots()
        if n_components == 2:
            ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
            ax.set_title("Projection PCA des Images Classifiées (2D)")
            st.write("Cette visualisation montre les images projetées dans un espace à deux dimensions après une réduction PCA. Chaque point représente une image classifiée en fonction de ses caractéristiques principales.")
        elif n_components == 1:
            ax.scatter(reduced_features[:, 0], np.zeros_like(reduced_features[:, 0]), c=labels, cmap='viridis')
            ax.set_title("Projection PCA des Images Classifiées (1D)")
            st.write("Cette visualisation montre les images projetées sur une seule dimension après une réduction PCA. Chaque point correspond à une image classifiée en fonction de sa caractéristique principale.")
        st.pyplot(fig)

        # --- Classification avec Naive Bayes ---
        st.write("### Classification avec Naive Bayes")
        nb = GaussianNB()
        nb.fit(features, y)
        pred_nb = nb.predict(features)
        st.write("Prédictions :", pred_nb)
        st.write("Précision :", accuracy_score(y, pred_nb))
        st.write("La précision indique le pourcentage d'images correctement classifiées parmi toutes les images traitées.")
        st.write("Score F1 :", f1_score(y, pred_nb))
        st.write("Le score F1 est une moyenne harmonique entre la précision et le rappel, utile pour évaluer les performances du modèle.")

        # --- Classification avec Réseau de Neurones ---
        st.write("### Classification avec Réseau de Neurones")
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
        mlp.fit(features, y)
        pred_mlp = mlp.predict(features)
        st.write("Prédictions :", pred_mlp)
        st.write("Précision :", accuracy_score(y, pred_mlp))
        st.write("La précision indique le pourcentage d'images correctement classifiées par le réseau de neurones.")
        st.write("Score F1 :", f1_score(y, pred_mlp))
        st.write("Le score F1 fournit une mesure équilibrée des performances du réseau de neurones, en tenant compte à la fois de la précision et du rappel.")
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
st.write("La centralité mesure l'importance d'un nœud dans le graphe, en fonction du nombre de connexions qu'il a avec d'autres nœuds.")
st.write("Clustering :", clustering)
st.write("Le coefficient de clustering mesure la probabilité que les voisins d'un nœud soient également connectés entre eux, indiquant des structures en triangle dans le graphe.")

st.write("Merci d'avoir utilisé cette application !")
