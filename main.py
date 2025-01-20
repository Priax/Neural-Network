import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import pickle
import os

# Activation functions
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))

def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1 - np.tanh(x)**2

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss functions
def categorical_cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

def categorical_cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true

class GeneralizedMLP:
    def __init__(self, layer_sizes=None, activations=None, loss='categorical_cross_entropy'):
        """
        Initializes a customizable MLP.
        :param layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        :param activations: List of activation functions for each layer (excluding input layer)
        :param loss: Loss function to use (default: categorical_cross_entropy)
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_function = categorical_cross_entropy
        self.loss_derivative = categorical_cross_entropy_derivative
        self.weights = []
        self.biases = []

        if layer_sizes and activations:
            assert len(layer_sizes) - 1 == len(activations), \
                "Number of activations must match the number of layers minus input layer"
            self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights and biases for each layer."""
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(
                np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(1.0 / self.layer_sizes[i])
            )  # He initialization
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))

    def _get_activation(self, name):
        """Retrieves the activation function and its derivative by name."""
        activations = {'relu': (relu, relu_derivative),
                       'sigmoid': (sigmoid, sigmoid_derivative),
                       'tanh': (tanh, tanh_derivative),
                       'softmax': (softmax, None)}
        return activations[name]

    def forward(self, X):
        """Performs forward propagation."""
        self.Z = []  # Linear activations
        self.A = [X]  # Non-linear activations (starting with input)
        for i in range(len(self.weights)):
            z = np.dot(self.A[-1], self.weights[i]) + self.biases[i]
            self.Z.append(z)
            activation_func = self._get_activation(self.activations[i])[0]
            self.A.append(activation_func(z) if activation_func else z)
        return self.A[-1]

    def backward(self, X, y, learning_rate):
        """Performs backward propagation and updates weights and biases."""
        m = X.shape[0]
        dA = self.loss_derivative(y, self.A[-1])
        for i in reversed(range(len(self.weights))):
            activation_derivative = self._get_activation(self.activations[i])[1]
            dZ = dA * (activation_derivative(self.Z[i]) if activation_derivative else 1)
            dW = np.dot(self.A[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, self.weights[i].T)

            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

    def train(self, X, y, epochs, learning_rate, batch_size):
        ## Trains the MLP on the given data.

        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)

            if epoch % 10 == 0:
                loss = self.loss_function(y, self.forward(X))

                predictions = np.argmax(self.forward(X), axis=1)  # Classe prédite
                labels = np.argmax(y, axis=1)                    # Classe réelle
                accuracy = np.mean(predictions == labels)        # Moyenne des prédictions correctes

                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.16f}, Accuracy: {accuracy:.2%}")
    """
    def train(self, X, y, epochs, learning_rate, batch_size):
        loss_history = []
        accuracy_history = []

        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)

            sample_indices = np.random.choice(X.shape[0], size=100, replace=False)
            X_sample, y_sample = X[sample_indices], y[sample_indices]
            output = self.forward(X_sample)
            loss = self.loss_function(y_sample, output)
            predictions = np.argmax(output, axis=1)
            labels = np.argmax(y_sample, axis=1)
            accuracy = np.mean(predictions == labels)

            loss_history.append(loss)
            accuracy_history.append(accuracy)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}, Accuracy: {accuracy:.2%}")

        self.plot_training_history(loss_history, accuracy_history)
    """

    def plot_training_history(self, loss_history, accuracy_history):
        """Trace l'évolution de la perte et de l'accuracy pendant l'entraînement."""
        epochs = range(1, len(loss_history) + 1)

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(epochs, loss_history, color='tab:red', label='Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.plot(epochs, accuracy_history, color='tab:blue', label='Accuracy')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        fig.tight_layout()
        plt.title("Training Loss and Accuracy")
        plt.show()

    def save(self, filename):
        """Saves the model's weights and biases to a file."""
        with open(filename, 'wb') as f:
            pickle.dump({'weights': self.weights, 'biases': self.biases, 'activations': self.activations, 'layer_sizes': self.layer_sizes}, f)

    def load(self, filename):
        """Loads the model's weights and biases from a file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.weights = data['weights']
        self.biases = data['biases']
        self.activations = data['activations']
        self.layer_sizes = data['layer_sizes']


game_state_labels = {
    'Check White': 0,    # White check
    'Check Black': 1,    # Black check
    'Checkmate White': 2, # White checkmate
    'Checkmate Black': 3, # Black checkmate
    'Stalemate': 4,       # The game is over, and a stalemate has occurred
    'Nothing': 5,
}

"""
def parse_custom_fen(fen: str):
    parts = fen.split()

    if len(parts) < 6 or len(parts) > 8:
        raise ValueError("Invalid FEN: Must have 6, 7, or 8 fields.")

    piece_placement = parts[0]
    active_color = parts[1]
    castling_availability = parts[2]
    en_passant_target = parts[3]
    halfmove_clock = parts[4]
    fullmove_number = parts[5]

    game_status = " ".join(parts[6:]) if len(parts) > 6 else None

    ranks = piece_placement.split("/")
    if len(ranks) != 8:
        raise ValueError("Invalid FEN: Piece placement must have 8 ranks.")
    for rank in ranks:
        if not rank or not all(c.isdigit() or c in "PNBRQKpnbrqk" for c in rank):
            raise ValueError(f"Invalid rank in piece placement: {rank}")

        if sum(int(c) if c.isdigit() else 1 for c in rank) != 8:
            raise ValueError(f"Invalid rank length in piece placement: {rank}")

    if active_color not in ("w", "b"):
        raise ValueError("Invalid FEN: Active color must be 'w' or 'b'.")

    if castling_availability != "-" and not all(c in "KQkq" for c in castling_availability):
        raise ValueError("Invalid FEN: Castling availability must be '-', or a combination of 'KQkq'.")

    if en_passant_target != "-" and not (len(en_passant_target) == 2 and en_passant_target[0] in "abcdefgh" and en_passant_target[1] in "36"):
        raise ValueError("Invalid FEN: En passant target square is not valid.")

    if not halfmove_clock.isdigit() or int(halfmove_clock) < 0:
        raise ValueError("Invalid FEN: Halfmove clock must be a non-negative integer.")

    if not fullmove_number.isdigit() or int(fullmove_number) <= 0:
        raise ValueError("Invalid FEN: Fullmove number must be a positive integer.")

    return {
        "piece_placement": piece_placement,
        "active_color": active_color,
        "castling_availability": castling_availability,
        "en_passant_target": en_passant_target,
        "halfmove_clock": int(halfmove_clock),
        "fullmove_number": int(fullmove_number),
        "game_status": game_status,
    }

def parse_fens_from_directory(directory: str):
    parsed_fens = []

    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)

            if filepath.endswith('.txt'):
                with open(filepath, 'r') as file:
                    for line in file:
                        fen = line.strip()
                        if fen:
                            try:
                                parsed_fens.append(parse_custom_fen(fen))
                            except ValueError as e:
                                print(f"Error parsing FEN in {filepath}: {e}")

    return parsed_fens
"""
def parse_custom_fen(fen: str):
    parts = fen.split()

    if len(parts) < 6 or len(parts) > 8:
        raise ValueError("Invalid FEN: Must have 6, 7, or 8 fields.")

    piece_placement = parts[0]
    active_color = parts[1]
    castling_availability = parts[2]
    en_passant_target = parts[3]
    halfmove_clock = parts[4]
    fullmove_number = parts[5]

    game_status = " ".join(parts[6:]) if len(parts) > 6 else None

    ranks = piece_placement.split("/")
    if len(ranks) != 8:
        raise ValueError("Invalid FEN: Piece placement must have 8 ranks.")
    for rank in ranks:
        if not rank or not all(c.isdigit() or c in "PNBRQKpnbrqk" for c in rank):
            raise ValueError(f"Invalid rank in piece placement: {rank}")

        if sum(int(c) if c.isdigit() else 1 for c in rank) != 8:
            raise ValueError(f"Invalid rank length in piece placement: {rank}")

    if active_color not in ("w", "b"):
        raise ValueError("Invalid FEN: Active color must be 'w' or 'b'.")

    if castling_availability != "-" and not all(c in "KQkq" for c in castling_availability):
        raise ValueError("Invalid FEN: Castling availability must be '-', or a combination of 'KQkq'.")

    if en_passant_target != "-" and not (len(en_passant_target) == 2 and en_passant_target[0] in "abcdefgh" and en_passant_target[1] in "36"):
        raise ValueError("Invalid FEN: En passant target square is not valid.")

    if not halfmove_clock.isdigit() or int(halfmove_clock) < 0:
        raise ValueError("Invalid FEN: Halfmove clock must be a non-negative integer.")

    if not fullmove_number.isdigit() or int(fullmove_number) <= 0:
        raise ValueError("Invalid FEN: Fullmove number must be a positive integer.")

    return {
        "piece_placement": piece_placement,
        "active_color": active_color,
        "castling_availability": castling_availability,
        "en_passant_target": en_passant_target,
        "halfmove_clock": int(halfmove_clock),
        "fullmove_number": int(fullmove_number),
        "game_status": game_status,
    }

def parse_fens_from_file(filepath):
    parsed_fens = []
    with open(filepath, 'r') as file:
        for line in file:
            fen = line.strip()
            if fen:
                try:
                    parsed_fens.append(parse_custom_fen(fen))
                except ValueError as e:
                    print(f"Error parsing FEN in {filepath}: {e}")
    return parsed_fens

def parse_fens_from_directory(directory: str):
    txt_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.txt'):
                txt_files.append(os.path.join(root, filename))

    with Pool(cpu_count()) as pool:
        parsed_fens_list = pool.map(parse_fens_from_file, txt_files)

    parsed_fens = [fen for sublist in parsed_fens_list for fen in sublist]
    return parsed_fens

def encode_chessboard(fen):
    """
    Encodes the chessboard from a FEN string into a numerical format for model input.
    Each square is represented by a 12-dimensional vector, corresponding to 6 pieces (pawn, knight, bishop, rook, queen, king)
    for both white and black.
    """
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11   # Black pieces
    }

    board = np.zeros(64 * 12)
    row = 0
    col = 0
    for char in fen:
        if char.isdigit():
            col += int(char)
        elif char == '/':
            row += 1
            col = 0
        else:
            piece_index = piece_map.get(char, -1)
            if piece_index != -1:
                # Set the piece for the corresponding square
                board[row * 8 + col + piece_index * 64] = 1
            col += 1
    return board


if __name__ == '__main__':
    input_size = 64 * 12
    hidden_layers = [256, 128, 64]
    output_size = 6

    mlp = GeneralizedMLP(
        layer_sizes=[input_size] + hidden_layers + [output_size],
        activations=['relu', 'relu', 'relu', 'softmax']
    )

    # Load and encode FEN data
    fen_data = parse_fens_from_directory("./datasets")

    X_train = []
    y_train = []
    for fen in fen_data:
        board_encoded = encode_chessboard(fen['piece_placement'])
        X_train.append(board_encoded)
        if fen.get("game_status") in game_state_labels:
            label = np.zeros(output_size)
            label[game_state_labels[fen["game_status"]]] = 1
            y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print("Beginning training")
    mlp.train(X_train, y_train, epochs=100, learning_rate=0.01, batch_size=32)
    mlp.save("test.pkl")

    mlp.load("test.pkl")

    fen_test = "rnb2b1k/1p3Pp1/p1n1p2Q/7p/1P6/P2B4/1BP2PPP/R3K2R b KQ - 0 20" ## 7R/5kp1/4n2p/1r4P1/1P3K2/8/8/8 w - - 1 56 Check Black
    encoded_board = encode_chessboard(fen_test.split()[0])
    encoded_board = encoded_board.reshape(1, -1)

    prediction = mlp.forward(encoded_board)

    game_states = list(game_state_labels.keys())
    predicted_class = np.argmax(prediction)
    predicted_state = game_states[predicted_class]

    print(f"Predicted game state: {predicted_state} with probability {prediction[0][predicted_class]:.4f}")
