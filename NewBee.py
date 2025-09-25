import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# Load a sample image (replace with your own)
image = cv2.imread('coin.jpg', cv2.IMREAD_GRAYSCALE)


# Fitness function to evaluate the quality of the edges detected
def fitness_function(edges):
    # The fitness function can be defined as the number of non-zero pixels (edges)
    # or some other metric like SSIM (Structural Similarity Index).
    return np.sum(edges)  # Simple fitness based on the number of edge pixels


# Apply Canny edge detection
def apply_edge_detection(threshold1, threshold2):
    return cv2.Canny(image, threshold1, threshold2)


# Bee Colony Optimization (BCO) algorithm for optimizing Canny thresholds
def bee_colony_optimization(image, population_size=50, iterations=100, explore_factor=0.3):
    # Initialize the bee population with random parameters (threshold1, threshold2)
    bees = []
    for _ in range(population_size):
        threshold1 = random.randint(50, 150)
        threshold2 = random.randint(150, 300)
        bees.append({'threshold1': threshold1, 'threshold2': threshold2, 'fitness': 0})

    best_solution = None
    best_fitness = -np.inf

    # Perform iterations
    for iteration in range(iterations):
        for bee in bees:
            # Apply edge detection with the current bee's thresholds
            edges = apply_edge_detection(bee['threshold1'], bee['threshold2'])
            # Evaluate fitness
            bee['fitness'] = fitness_function(edges)

            # Update the best solution found
            if bee['fitness'] > best_fitness:
                best_fitness = bee['fitness']
                best_solution = bee

        # Exploitation step: bees around the best solution try to improve
        for bee in bees:
            if bee != best_solution:
                # Slightly adjust the threshold values around the best solution
                bee['threshold1'] = int(best_solution['threshold1'] + random.randint(-5, 5))
                bee['threshold2'] = int(best_solution['threshold2'] + random.randint(-5, 5))

        # Exploration step: some bees explore new areas
        for bee in bees:
            if random.random() < explore_factor:
                bee['threshold1'] = random.randint(50, 150)
                bee['threshold2'] = random.randint(150, 300)

        print(f"Iteration {iteration + 1}/{iterations}: Best Fitness = {best_fitness}")

    return best_solution


# Running BCO to optimize edge detection
best_solution = bee_colony_optimization(image)

# Apply edge detection with the optimized thresholds
final_edges = apply_edge_detection(best_solution['threshold1'], best_solution['threshold2'])

# Show the result
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Optimized Edge Detection")
plt.imshow(final_edges, cmap='gray')

plt.show()

# Print the best thresholds found
print(f"Best Thresholds: threshold1 = {best_solution['threshold1']}, threshold2 = {best_solution['threshold2']}")
