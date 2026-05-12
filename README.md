# Deep Learning Projects

This repository contains a structured collection of deep learning projects ranging from foundational implementations to intermediate-level applications. Each project is developed using Python and documented through Jupyter Notebooks, making it straightforward to follow the workflow from data preparation through model evaluation.

The goal of this repository is to consolidate hands-on work done while learning and applying core deep learning concepts, and to serve as a reference for implementations that go beyond theoretical study.

---

## Repository Structure

```
Deep-Learning-Projects/
├── Project 1/
├── Project 2/
├── Project 3/
├── Project 4/
└── Project 5/
    └── Main Model Train/
```

Each project folder contains its own Jupyter Notebook(s) along with any associated data files or resources relevant to that specific implementation.

---

## Projects Overview

**Project 1**
Covers fundamental deep learning concepts including neural network architecture, forward propagation, and basic optimization. Serves as the foundation for subsequent projects in this repository.

**Project 2**
Builds on the basics by introducing more structured model training pipelines. Focuses on data preprocessing, model compilation, and evaluating performance metrics across training and validation sets.

**Project 3**
Implements a domain-specific deep learning model addressing a classification or regression problem. Explores hyperparameter tuning and discusses how architectural decisions affect model performance.

**Project 4**
Advances into intermediate territory with a more complex dataset or task. Covers topics such as regularization techniques, batch normalization, dropout layers, and strategies to mitigate overfitting.

**Project 5**
The most involved project in this collection. The `Main Model Train` subfolder contains the primary training pipeline, including data loading, model definition, training loops, and result analysis. This project demonstrates an end-to-end deep learning workflow.

---

## Technologies and Libraries Used

- **Language:** Python 3
- **Notebook Environment:** Jupyter Notebook
- **Core Libraries:**
  - TensorFlow / Keras or PyTorch (depending on the project)
  - NumPy
  - Pandas
  - Matplotlib / Seaborn
  - Scikit-learn

---

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/Md-Shaon-Khan/Deep-Learning-Projects.git
cd Deep-Learning-Projects
```

**2. Set up a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install the required dependencies**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

Or if using PyTorch:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision jupyter
```

**4. Launch Jupyter Notebook**

```bash
jupyter notebook
```

Navigate to the project folder of your choice and open the corresponding `.ipynb` file.

---

## Prerequisites

A basic understanding of the following is recommended before going through these projects:

- Python programming fundamentals
- Linear algebra and calculus basics
- Introductory machine learning concepts (what a model is, training vs. testing, loss functions)

No prior deep learning experience is required. The projects are structured to build understanding progressively.

---

## Notes

- All notebooks are written with readability in mind. Markdown cells explain the reasoning behind each step.
- Where applicable, comments within the code clarify non-obvious implementation choices.
- Projects are self-contained. You do not need to complete them in order, although earlier projects introduce concepts referenced in later ones.

---

## About

This repository reflects work done during my learning journey in deep learning. The projects here are not academic submissions or production systems. They are practical exercises built to develop intuition for how deep neural networks are designed, trained, and evaluated.

Feedback, suggestions, and constructive criticism are welcome.

---

## Author

**Md. Shaon Khan**
GitHub: [Md-Shaon-Khan](https://github.com/Md-Shaon-Khan)

---

## License

This repository is open for educational and personal use. If you reference or build upon any of this work, a mention or credit is appreciated.
