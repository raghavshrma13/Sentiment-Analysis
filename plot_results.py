import matplotlib.pyplot as plt
import numpy as np

# Data points (dataset sizes)
sizes = [100, 1000, 10000]

# Sequential times (in milliseconds)
sequential_preprocessing = [48, 417, 3678]  # from your sequential output
sequential_embedding = [16, 138, 1276]

# Parallel times (in milliseconds)
parallel_preprocessing = [16, 111, 1012]  # from your parallel output
parallel_embedding = [2, 20, 241]

def create_comparison_plot():
    plt.figure(figsize=(12, 6))
    
    # Add inscription text
    inscription = "Parallel vs Sequential Sentiment Analysis\nRaghav Sharma (2023BCS50) & Gaurav Jhalani (2023BCS32)"
    plt.figtext(0.5, 1.05, inscription, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Preprocessing comparison
    plt.subplot(1, 2, 1)
    plt.plot(sizes, sequential_preprocessing, 'ro-', label='Sequential')
    plt.plot(sizes, parallel_preprocessing, 'bo-', label='Parallel')
    plt.title('Preprocessing Time Comparison')
    plt.xlabel('Dataset Size')
    plt.ylabel('Time (ms)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # Embedding comparison
    plt.subplot(1, 2, 2)
    plt.plot(sizes, sequential_embedding, 'ro-', label='Sequential')
    plt.plot(sizes, parallel_embedding, 'bo-', label='Parallel')
    plt.title('Embedding Time Comparison')
    plt.xlabel('Dataset Size')
    plt.ylabel('Time (ms)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.suptitle('Sequential vs Parallel Processing Times\nRaghav Sharma 2023BCS0050', y=1.05)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('performance_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_speedup_plot():
    # Calculate speedup
    preprocessing_speedup = [s/p for s, p in zip(sequential_preprocessing, parallel_preprocessing)]
    embedding_speedup = [s/p for s, p in zip(sequential_embedding, parallel_embedding)]
    
    plt.figure(figsize=(10, 6))
    
    # Add inscription text
    inscription = "Speedup Analysis\nRaghav Sharma (2023BCS50) & Gaurav Jhalani (2023BCS32)"
    plt.figtext(0.5, 1.05, inscription, ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.plot(sizes, preprocessing_speedup, 'ro-', label='Preprocessing')
    plt.plot(sizes, embedding_speedup, 'bo-', label='Embedding')
    plt.title('Speedup Analysis\nRaghav Sharma 2023BCS0050')
    plt.xlabel('Dataset Size')
    plt.ylabel('Speedup (Sequential/Parallel)')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust top margin for inscription
    
    # Save plots with higher resolution
    plt.savefig('speedup_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    create_comparison_plot()
    create_speedup_plot()
    print("Plots have been saved as 'performance_comparison.png' and 'speedup_analysis.png'")