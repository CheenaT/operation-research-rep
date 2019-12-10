"""
Code to draw plot for the documentation.

This plots a divergent fictitious play example.

The code should match the reference code in the documentation.
"""
import matplotlib.pyplot as plt
import nashpy as nash
import numpy as np

A = np.array([[4,0,6,2,2,1],[3,8,4,10,4,4],[1,2,6,5,0,0],[6,6,4,4,10,3],[10,4,6,4,0,9],[10,7,0,7,9,8]])
B = np.array([[-4,0,-6,-2,-2,-1],[-3,-8,-4,-10,-4,-4],[-1,-2,-6,-5,0,0],[-6,-6,-4,-4,-10,-3],[-10,-4,-6,-4,0,-9],[-10,-7,0,-7,-9,-8]])

game = nash.Game(A, B)
iterations = 10000
np.random.seed(0)
play_counts = tuple(game.fictitious_play(iterations=iterations))


plt.figure()
probabilities = [
    row_play_counts / np.sum(row_play_counts)
    for row_play_counts, col_play_counts in play_counts
]
for number, strategy in enumerate(zip(*probabilities)):
    plt.plot(strategy, label=f"$s_{number}$")

plt.xlabel("Iteration")
plt.ylabel("Probability")
plt.legend()
plt.title("Actions taken by row player")
plt.savefig("main.svg", transparent=True)
