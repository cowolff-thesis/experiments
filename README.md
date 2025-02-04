# Master Thesis Experiments - Cornelius Wolff

This repository contains the experiments, plotting tools, and testing framework developed and used for the master thesis of Cornelius Wolff.

---

## Repository Structure

- **`Experiments/`**  
  Contains the core experimental scripts and configurations used for this thesis.

- **`Plotting/`**  
  Includes scripts and tools for visualizing the results of experiments, as well as the actual results. This folder provides utilities for generating charts, graphs, and other visual outputs used in the thesis.

- **`Testing/`**  
  Houses testing scripts to ensure the robustness and reliability of the codebase.

- **`.gitignore`**  
  Specifies intentionally untracked files to be ignored by `git`.

- **`README.md`**  
  Provides an overview of the repository, including its structure, purpose, and usage instructions.

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/cowolff-thesis/experiments
   cd your-repo-name
   ```

2. **Install dependencies**:  
   Ensure all required libraries and tools are installed. Use the provided `requirements.txt` or relevant package manager.

3. **Run an experiment**:  
   Navigate to the `Experiments/` folder and execute the scripts:
   ```bash
   python experiment_name.py
   ```

4. **Plot results**:  
   Utilize the plotting tools in the `Plotting/` folder to generate visualizations:
   ```bash
   python plot_results.py
   ```
   Often times, these scripts are also provided as jupyter notebooks.

5. **Run tests**:  
   Execute the test scripts to verify functionality:
   ```bash
   python -m unittest discover Testing/
   ```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgements

This repository was developed as part of Cornelius Wolff's master thesis. Special thanks to the entire lab of Prof. Elia Bruni at Osnabrück University for their support and guidance throughout the project.