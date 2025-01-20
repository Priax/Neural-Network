#!/usr/bin/env python3
##
## EPITECH PROJECT, 2024
## B-CNA-500-MAR-5-1-neuralnetwork-vincent.montero-fontaine
## File description:
## neural_network_analyzer
##

import argparse
import numpy as np
import json
from main import GeneralizedMLP
import os

def load_config(config_file):
    """Charge les paramètres du modèle à partir d'un fichier JSON."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


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

    board = np.zeros(64 * 12)  # 64 squares, 12 possible piece types
    row = 0
    col = 0
    for char in fen:
        if char.isdigit():  # Empty squares are represented by digits
            col += int(char)
        elif char == '/':  # Separator between ranks
            row += 1
            col = 0
        else:
            piece_index = piece_map.get(char, -1)
            if piece_index != -1:
                board[row * 8 + col + piece_index * 64] = 1
            col += 1
    return board


game_state_labels = {
    'Check White': 0,    # White is in check
    'Check Black': 1,    # Black is in check
    'Checkmate White': 2, # White is checkmated
    'Checkmate Black': 3, # Black is checkmated
    'Stalemate': 4,       # The game is over, and a stalemate has occurred
    'Nothing': 5,
}


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


def parse_fen_file(filepath):
    """Parse un fichier contenant des FEN et des labels éventuels."""
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

def predict(model, file):
    """Effectue des prédictions sur les FEN fournies dans le fichier."""
    data = parse_fen_file(file)
    game_states = list(game_state_labels.keys())
    results = []
    for parsed_fen in data:
        input_vector = encode_chessboard(parsed_fen['piece_placement']).reshape(1, -1)
        output = model.forward(input_vector)
        predicted_class = np.argmax(output)
        predicted_state = game_states[predicted_class]
        results.append(predicted_state)
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyse des échiquiers avec un réseau de neurones.")
    parser.add_argument("--predict", action="store_true", help="Lancer en mode prédiction.")
    parser.add_argument("--train", action="store_true", help="Lancer en mode entraînement.")
    parser.add_argument("--save", help="Fichier pour sauvegarder le réseau après entraînement.")
    parser.add_argument("configfile", nargs='?', help="Fichier de configuration pour le réseau de neurones.")
    parser.add_argument("loadfile", help="Fichier contenant le réseau de neurones.")
    parser.add_argument("file", help="Fichier contenant les échiquiers en FEN.")
    args = parser.parse_args()

    input_size = 64 * 12
    output_size = 6
    if not args.predict:
        config = load_config(args.configfile)
        model = GeneralizedMLP(
            layer_sizes=[input_size] + config['layers'] + [output_size],
            activations=config['activations']
        )

    if args.predict:
        if not args.loadfile or not args.file:
            print("Erreur : Vous devez spécifier le fichier du réseau (--loadfile) et le fichier FEN (--file) pour la prédiction.")
            return
        model = GeneralizedMLP()
        model.load(args.loadfile)
        predictions = predict(model, args.file)
        for prediction in predictions:
            print(prediction)

    if args.train:
        if not args.save:
            print("Erreur : Vous devez spécifier un fichier de sauvegarde pour l'entraînement.")
            return
        fen_data = parse_fen_file(args.file)

        model = model.load(args.save)
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

        model.train(X_train, y_train, epochs=config['epochs'], learning_rate=config['learning_rate'], batch_size=config['batch_size'])
        model.save(args.save)

if __name__ == "__main__":
    main()
