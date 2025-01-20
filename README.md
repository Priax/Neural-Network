# Documentation MLP

## Introduction

Ce projet vise à concevoir un perceptron multicouche (MLP) pour modéliser les états de jeu d'échecs, avec comme objectif d'effectuer des prédictions précises sur des positions spécifiques du plateau. En utilisant des techniques comme la propagation avant, la rétropropagation et l'initialisation He, ce modèle apprend à partir des données pour optimiser ses prédictions. Ce document décrit en détail les composants, les algorithmes et les justifications des choix effectués dans l'implémentation.

## 1. Fonctions d'activation

```relu(x)``` : Renvoie le maximum entre 0 et x (fonction ReLU).

```relu_derivative(x)``` : Renvoie la dérivée de la fonction ReLU.

```sigmoid(x)``` : Applique la fonction sigmoïde sur x.

```sigmoid_derivative(x)``` : Renvoie la dérivée de la fonction sigmoïde.

```tanh(x)``` : Applique la fonction tangente hyperbolique.

```tanh_derivative(x)``` : Renvoie la dérivée de la fonction tangente hyperbolique.

```softmax(x)``` : Applique la fonction softmax pour convertir les logits en probabilités.

## 2. Fonctions de perte

```categorical_cross_entropy(y_true, y_pred)``` : Calcule l'entropie croisée pour la classification multi-classe.

```categorical_cross_entropy_derivative(y_true, y_pred)``` : Calcule la dérivée de l'entropie croisée.

## 3. Classe GeneralizedMLP

Classe implémentant un Multi Layered Perceptron (MLP) personnalisable.

### Méthodes principales

```__init__(layer_sizes, activations, loss)``` :

Initialise le MLP avec les tailles des couches, les fonctions d'activation, et la fonction de perte.

Vérifie la correspondance entre le nombre de couches et les fonctions d'activation.

Initialise les poids et les biais.

```_initialize_weights()``` :

Initialise les poids avec l'initialisation He.

Initialise les biais à 0.

```_get_activation(name)``` :

Retourne la fonction d'activation et sa dérivée correspondant au nom fourni.

```forward(X)``` :

Effectue une propagation avant sur les données d'entrée X, en calculant les sorties de chaque couche du réseau à partir des poids, des biais et des fonctions d'activation spécifiés. Cela permet de prédire les résultats en fonction des données d'entrée actuelles.

```backward(X, y, learning_rate)``` :

Calcule les gradients des poids et biais via rétropropagation.

Met à jour les poids et biais avec le taux d'apprentissage.

```train(X, y, epochs, learning_rate, batch_size)``` :

Entraîne le modèle sur les données d'entrée et de sortie X et y.

Affiche la perte et la précision tous les 10 epochs.

```save(filename)``` :

Sauvegarde les poids, les biais, et la configuration dans un fichier pickle.

```load(filename)``` :

Charge les poids, les biais, et la configuration à partir d'un fichier pickle.

## 1. Gestion des états du jeu d'échecs

```game_state_labels``` : Dictionnaire associant les états du jeu d'échecs (Check Black, Stalemate, Nothing...) à des indices numériques.

```parse_custom_fen(fen)``` :

Valide et extrait les informations d'un FEN (Forsyth-Edwards Notation).

Retourne un dictionnaire contenant les placements des pièces, les droits de roque, le nombre de coups, etc.

```parse_fens_from_directory(directory)``` :

Parcourt un répertoire et extrait tous les FEN valides à partir des fichiers texte.

```encode_chessboard(fen)``` :

Encode le plateau d'échecs représenté par un FEN en une matrice 1D utilisable pour le modèle.

Chaque case est représentée par un vecteur de 12 dimensions correspondant à toutes les pièces possibles.

## 1. Exemple d'utilisation

Taille d'entrée : 64 * 12 (plateau d'échecs encodé).

Couches cachées : [256, 128, 64]. <br>
Ces tailles ont été choisies pour équilibrer la capacité d'apprentissage et les besoins en calcul. Une couche initiale de 256 neurones permet d'extraire des motifs complexes des données brutes encodées. Les couches successives de 128 et 64 neurones réduisent progressivement la dimensionnalité, favorisant une meilleure généralisation et limitant le surapprentissage (overfitting). Cette architecture a montré une bonne performance pour capturer les relations non linéaires dans les états de jeu d'échecs.

Taille de sortie : 6 (correspondant à game_state_labels).

### Entraînement :

```py
mlp = GeneralizedMLP(
    layer_sizes=[64 * 12, 256, 128, 64, 6],
    activations=['relu', 'relu', 'relu', 'softmax']
)
mlp.train(X_train, y_train, epochs=100, learning_rate=0.01, batch_size=32)
```

## 6. Notes

Robustesse : Les fonctions de validation des FEN garantissent des entrées valides.

Modularité : Le modèle est hautement configurable (activation, taille des couches, perte).

## Améliorations possibles :

- Ajout de mécanismes de régularisation (dropout, L2).

- Support d'autres fonctions de perte.

## Note

Du Multithreading n'a pas été rajouté car il a été testé que le modèle fonctionnait mieux en mélangeant plusieurs données différentes (pour que le modèle ne s'adapte pas trop à un seul état).

Le modèle fonctionne donc mieux en récupérant plusieurs données différentes, qui seront ensuites shuffle entre elles, puis le modèle s'entraînera.
