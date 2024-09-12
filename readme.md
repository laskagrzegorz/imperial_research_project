# Imperial Research Project

This repository contains the code and resources for an ongoing research project conducted at Imperial College London. The project focuses on detecting edges in graph structure of stationary Mutually Exciting Hawkes Processes.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Getting Started](#getting-started)
4. [Contributing](#contributing)
5. [License](#license)

## Project Overview

### Non-parametric Modelling of Graph Structure for Stationary Mutually Exciting Hawkes Processes

This project explores non-parametric approaches to model the structure of undirected graphs using stationary time series data. The focus is on applying algorithms traditionally used for Gaussian processes to Mutually Exciting Hawkes Processes (MEHP). These processes are widely used in fields such as epidemiology, finance, and social media analysis, where events are self-exciting and mutually exciting, meaning that the occurrence of one event increases the likelihood of subsequent events.
## Key Features

Key Features:
1. **Graphical Models for Time Series**: The project constructs graphical models to represent relationships between multiple time series. Each time series is represented as a vertex, and edges between vertices represent connections based on partial correlation.
2. **Spectral Domain Analysis**: Algorithms are applied in the frequency domain to test for the presence of edges in an undirected graph. This approach provides a more generalized model structure that does not assume Gaussianity, allowing the inclusion of non-Gaussian processes like the Mutually Exciting Hawkes Process.
3. **Performance Evaluation**: The project compares the performance of classical Gaussian-based methods (like Matsuda and Medkour algorithms) and extends them to handle MEHPs through simulations.
4. **Applications**: The methods explored in this project have potential applications in finance (for modeling financial contagion), epidemiology (for modeling disease spread), and social media (for modeling information diffusion).

This repository contains the code, simulation data, and algorithms used for non-parametric graph structure modeling in both Gaussian and Hawkes processes.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository:
git clone https://github.com/laskagrzegorz/imperial_research_project.git


2. Navigate to the project directory:
cd imperial_research_project


3. Set up the virtual environment:
python -m venv env source env/bin/activate # On Windows, use env\Scripts\activate


4. Install required dependencies:
pip install -r requirements.txt


5. Run the main script:
python main.py


## Contributing

We welcome contributions to this project. Here's how you can get involved:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes and commit them
4. Open a pull request

Please ensure that your contribution adheres to our coding standards and follows the project's guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
