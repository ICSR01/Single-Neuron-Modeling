# Single Neuron Modeling

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![Neuroscience](https://img.shields.io/badge/domain-Computational%20Neuroscience-success.svg)

A modular, Python-based evaluation pipeline for simulating single-neuron spiking behavior and validating mathematical models against biological patch-clamp data.

---

## Table of Contents
* [Part 1: The Fundamentals](#part-1-the-fundamentals)
    * [Description of Technology](#description-of-technology)
    * [Description of the Process](#description-of-the-process)
* [Part 2: Context & Depth](#part-2-context--depth)
    * [How the Project Came About](#how-the-project-came-about)
    * [The Motivation](#the-motivation)
    * [What Problem it Hopes to Solve](#what-problem-it-hopes-to-solve)
    * [Intended Use](#intended-use)
    * [Challenges](#challenges)
    * [Limitations](#limitations)
    * [Credits & Author](#credits--author)
* [Usage & Installation](#usage--installation)

---

## Part 1: The Fundamentals

### Description of Technology
This project is built purely in **Python**.
* **NumPy:** Used for vectorized numerical operations and efficient Euler integration of differential equations over time series.
* **Matplotlib:** Used for generating high-quality comparative visualizations (membrane potential traces and spike raster plots).

**Why these tools?** Python, supported by NumPy and Matplotlib, is the lingua franca of computational neuroscience and AI in healthcare. By avoiding heavy, specialized simulation frameworks, this raw implementation demonstrates a fundamental mathematical understanding of the underlying physiological mechanics.

### Description of the Process
The development workflow was split into two distinct architectural domains:
1. **The Modeling Module (`neuron_models.py`):** Encapsulates the pure mathematical logic, progressing from a basic Integrate-and-Fire (I&F) concept to a sophisticated Adaptive Threshold Leaky Integrate-and-Fire (LIF) model.
2. **The Testing Pipeline (`test_models.py`):** Handles data ingestion, simulation execution, and quantitative evaluation using the Van Rossum distance metric.

**Why this approach?** Separating the core logic from the execution scripts mirrors professional software engineering practices. It ensures the models are reusable, easier to debug, and decoupled from the specific experimental data being tested.

---

## Part 2: Context & Depth

### How the Project Came About
This repository was born out of the practical hands-on sessions at the **ACNEI Computational Neuroscience Introductory School**. It represents an independent contribution developed as part of the overarching project work for **Team Uropi Pod**. 

### The Motivation
As an early career computational neuroscientist focused on the intersection of medicine and technology, my primary drive is bridging the gap between clinical neurophysiology and artificial intelligence. Simulating biological neurons is the foundational step toward understanding complex neural networks, computational psychiatry, and advanced neuroimaging algorithms.

### What Problem it Hopes to Solve
Theoretical models of neurons often fail to capture biological complexity. This project addresses that gap by actively measuring how well mathematical abstractions (like leak conductances and dynamic thresholds) actually fit real-world *in vitro* data.

### Intended Use
* **Academic Portfolio:** To demonstrate competency in scientific computing and computational modeling to future research supervisors and collaborators.
* **Educational Reference:** To serve as a clear, step-by-step mathematical reference for other students moving from biological sciences into computational fields.

### Challenges
* **Parameter Tuning:** Matching the theoretical parameters (like the membrane time constant, $\tau_m$, and threshold decay) to accurately reflect the spiking frequency of the biological patch-clamp data.
* **Metric Implementation:** Translating the continuous evaluation of spike trains via the Van Rossum distance algorithm into an efficient, vectorized Python function.

### Limitations
* **Spatial Morphology:** These point-neuron models treat the neuron as a single uniform compartment, ignoring the complex spatial integration that occurs in actual dendritic trees.
* **Network Dynamics:** This repository focuses strictly on isolated single-cell dynamics and does not currently simulate synaptic communication or emergent network behaviors.

### Credits & Author
* **Independent Developer / Author:** Ilo Celestine Somadina — *Early career computational neuroscientist | Team Uropi Pod*
* **Theoretical Instruction:** Theoretical concepts were explained by AbdelQader AlKilany at the ACNEI Computational Neuroscience Introductory School.
* **Datasets:** Biological patch-clamp recordings provided via the TReND-CaMinA curriculum.

---

## Usage & Installation

### Requirements
Ensure your biological datasets are located in the `Data/` directory:
* `Data/train_current.csv`
* `Data/train_traces.csv`
* `Data/train_spikes.csv`

### Execution
Run the evaluation pipeline from your terminal to simulate the models and calculate the Van Rossum distance:

```bash
pip install numpy matplotlib
python test_models.py
