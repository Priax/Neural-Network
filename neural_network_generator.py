#!/usr/bin/env python3
##
## EPITECH PROJECT, 2024
## B-CNA-500-MAR-5-1-neuralnetwork-vincent.montero-fontaine
## File description:
## neural_network_generator
##

import argparse
import os
import json
import pickle
import random
from typing import List, Dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate neural networks based on configuration files.")
    parser.add_argument('configs', nargs='+', help='Configuration file and number of networks to generate, in pairs.')
    return parser.parse_args()


def validate_config(config: Dict):
    required_keys = ["layers", "activations", "learning_rate", "loss_function"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in configuration: {key}")


def generate_neural_networks(config_file: str, nb: int):
    """Génère 'nb' réseaux neuronaux à partir du fichier de configuration 'config_file'."""
    with open(config_file, 'r') as f:
        config = json.load(f)

    validate_config(config)

    networks = []
    for i in range(nb):
        input_size = config['layers'][0] * 12
        hidden_layers = [256, 128, 64]
        output_size = 6

        layer_sizes = [input_size] + hidden_layers + [output_size]

        modified_config = config.copy()
        modified_config['layer_sizes'] = layer_sizes
        modified_config['seed'] = random.randint(0, 100000)

        networks.append(modified_config)

    return networks



def save_networks_as_pkl(networks: List[Dict], output_dir: str, config_name: str):
    """Saves generated networks as Pickle (.pkl) files."""
    os.makedirs(output_dir, exist_ok=True)
    for i, network in enumerate(networks):
        filename = os.path.join(output_dir, f"{config_name}_network_{i+1}.pkl")
        with open(filename, 'wb') as f:
            pickle.dump(network, f)
    print(f"Saved {len(networks)} networks as Pickle files in '{output_dir}'.")


def main():
    args = parse_arguments()

    if len(args.configs) % 2 != 0:
        raise ValueError("Arguments must be in pairs: config_file_1 nb_1 [config_file_2 nb_2 ...]")

    config_files = args.configs[::2]
    network_counts = args.configs[1::2]

    for config_file, count in zip(config_files, network_counts):
        if not os.path.isfile(config_file):
            print(f"Error: Config file {config_file} does not exist.")
            continue

        try:
            count = int(count)
        except ValueError:
            print(f"Error: {count} is not a valid number of networks.")
            continue

        output_dir = "generated_networks"
        print(f"Generating {count} networks based on {config_file}...")
        networks = generate_neural_networks(config_file, count)
        print(networks)
        save_networks_as_pkl(networks, output_dir, os.path.basename(config_file).split('.')[0])

        print(f"Saved {count} networks to {output_dir}/")


if __name__ == '__main__':
    main()
