"""
Basic analysis script for static.csv data.
Generates charts showing hardware and hyperparameter distributions.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
try:
    data = pd.read_csv("../data-collection/data/static.csv")
except FileNotFoundError:
    print("Error: static.csv not found in data-collection/")
    print("Run training_script.py first to generate data.")
    exit()

print(f"Loaded {len(data)} training runs\n")

# Operating System distribution
if 'os' in data.columns:
    print("Operating Systems:")
    os_counts = data['os'].value_counts()
    print(os_counts)

    plt.figure(figsize=(8, 5))
    os_counts.plot(kind='bar', color='blue')
    plt.title('Operating System Distribution')
    plt.xlabel('OS')
    plt.ylabel('Number of Runs')
    plt.tight_layout()
    plt.savefig('os_distribution.png')
    plt.close()

# GPU distribution
if 'gpu' in data.columns:
    print("\nGPU Types:")
    gpu_counts = data['gpu'].value_counts()
    print(gpu_counts)

    plt.figure(figsize=(10, 5))
    gpu_counts.plot(kind='bar', color='red')
    plt.title('GPU Distribution')
    plt.xlabel('GPU')
    plt.ylabel('Number of Runs')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('gpu_distribution.png')
    plt.close()

# Game distribution
if 'game_name' in data.columns:
    print("\nGames:")
    game_counts = data['game_name'].value_counts()
    print(game_counts)

    plt.figure(figsize=(10, 5))
    game_counts.plot(kind='bar', color='green')
    plt.title('Game Distribution')
    plt.xlabel('Game')
    plt.ylabel('Number of Runs')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('game_distribution.png')
    plt.close()

# Training type
if 'training_type' in data.columns:
    print("\nTraining Types:")
    training_counts = data['training_type'].value_counts()
    print(training_counts)

    plt.figure(figsize=(8, 8))
    training_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Training Type Distribution')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('training_type_distribution.png')
    plt.close()

# Batch size
if 'batch_size' in data.columns:
    print("\nBatch Size:")
    print(f"  Min: {data['batch_size'].min()}")
    print(f"  Max: {data['batch_size'].max()}")
    print(f"  Average: {data['batch_size'].mean():.2f}")

    batch_counts = data['batch_size'].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    batch_counts.plot(kind='bar', color='gray')
    plt.title('Batch Size Distribution')
    plt.xlabel('Batch Size')
    plt.ylabel('Number of Runs')
    plt.tight_layout()
    plt.savefig('batch_size_distribution.png')
    plt.close()

# Learning rate
if 'learning_rate' in data.columns:
    print("\nLearning Rate:")
    print(f"  Min: {data['learning_rate'].min()}")
    print(f"  Max: {data['learning_rate'].max()}")
    print(f"  Average: {data['learning_rate'].mean():.6f}")

    plt.figure(figsize=(10, 5))
    data['learning_rate'].hist(bins=20, color='blue', edgecolor='black')
    plt.title('Learning Rate Distribution')
    plt.xlabel('Learning Rate')
    plt.ylabel('Number of Runs')
    plt.tight_layout()
    plt.savefig('learning_rate_distribution.png')
    plt.close()

# Save summary
with open('analysis_summary.txt', 'w') as f:
    f.write("STATIC DATA ANALYSIS SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write(f"Total training runs: {len(data)}\n\n")

    if 'os' in data.columns:
        f.write("Operating Systems:\n")
        for os_name, count in data['os'].value_counts().items():
            f.write(f"  {os_name}: {count}\n")
        f.write("\n")

    if 'gpu' in data.columns:
        f.write("GPU Types:\n")
        for gpu_name, count in data['gpu'].value_counts().items():
            f.write(f"  {gpu_name}: {count}\n")
        f.write("\n")

    if 'game_name' in data.columns:
        f.write("Games:\n")
        for game_name, count in data['game_name'].value_counts().items():
            f.write(f"  {game_name}: {count}\n")
        f.write("\n")

    if 'batch_size' in data.columns:
        f.write(f"Batch Size Range: {data['batch_size'].min()} - {data['batch_size'].max()}\n")
        f.write(f"Batch Size Average: {data['batch_size'].mean():.2f}\n\n")

    if 'learning_rate' in data.columns:
        f.write(f"Learning Rate Range: {data['learning_rate'].min()} - {data['learning_rate'].max()}\n")
        f.write(f"Learning Rate Average: {data['learning_rate'].mean():.6f}\n")

print("\nGenerated 6 charts and sumary file.")
