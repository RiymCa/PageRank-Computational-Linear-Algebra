# PageRank Algorithm and Power Method

This repository contains the implementation and mathematical analysis of the PageRank algorithm, developed as a project for the Computational Linear Algebra for Large Scale Problems course.

The project focuses on computing the principal eigenvector of column-stochastic adjacency matrices using the Power Method. The methodology and theoretical foundation reference the provided paper, "The $25,000,000,000 Eigenvector: The Linear Algebra Behind Google" by Kurt Bryant and Tanya Leise. The implementation addresses common issues in real-world web graphs, such as disconnected subwebs and dangling nodes, by introducing a teleportation matrix and a damping factor. To handle large-scale networks efficiently, the algorithm transitions from a dense matrix representation to a Compressed Sparse Row (CSR) format.

### Repository Structure

* **THE $25,000,000,000 EIGENVECTOR THE LINEAR ALGEBRA BEHIND GOOGLE.pdf**: The foundational reference paper provided for the course.
* **Algorithm.pdf**: The main technical report detailing the mathematical foundations, algorithm modifications, and performance results.
* **Algorithm/**: Contains the Python implementation.
  * `main.py`: The primary script to execute the PageRank algorithm on paper webs and the `hollins.dat` dataset.
  * `utils.py` & `analysis.py`: Helper modules for CSR matrix multiplication, timing, and result visualization.
  * `hollins.dat`: The dataset used to test the algorithm on a real-world network.
* **Exercises/**: Contains theoretical derivations and computational exercises.
  * `Exercises.pdf` & `Exercises.ipynb`: Mathematical proofs, eigenvalue analysis, and Python implementations of the exercises.
  * `convergence.pdf`: Plots demonstrating the convergence rate of the Power Method.
### Authors
* Nima Shadab
* Riym Chaoui Aziz
