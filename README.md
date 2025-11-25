# 🧬 Linear Regression Analysis - Diabetes Dataset

**A Narrative-Driven Data Science Application**

Welcome to the **Diabetes Data Story**, an interactive Streamlit application that transforms standard data analysis into an engaging narrative. This project explores the famous Diabetes dataset to identify disease progression factors using machine learning techniques, featuring advanced 3D visualizations and a modern "Midnight Pro" UI.

## 🌟 Key Features

*   **📖 Story Mode**: Navigate through the analysis in chapters ("The Patient Data", "The Quest for Predictors", etc.) rather than a flat dashboard.
*   **🎨 Midnight Pro Theme**: A custom-designed dark UI with high-contrast typography and professional aesthetics for maximum readability.
*   **🧊 3D Visualizations**: Interactive 3D scatter plots and regression planes powered by Plotly to explore multi-dimensional relationships.
*   **🏎️ Algorithm Race**: A side-by-side comparison of **Linear Regression** vs. **XGBoost**, highlighting performance differences (MSE).
*   **🛤️ LARS Path**: Visualizing the **Least Angle Regression** algorithm to show how features enter the model, based on the seminal paper by Efron et al.

## 🛠️ Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/RafiulPaceProjects/Linear-Regression-Analysis-DD.git
    cd Linear-Regression-Analysis-DD
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    streamlit run streamlit_app.py
    ```

4.  **Open in Browser**:
    The app will typically launch at `http://localhost:8501`.

## 📂 Project Structure

*   `streamlit_app.py`: The main application file containing the UI and narrative logic.
*   `project_utils.py`: Helper functions for data loading, model training, and plotting.
*   `requirements.txt`: List of Python dependencies.
*   `Project_details/`: Directory containing the `diabetes.data.txt` dataset.

## 👨‍💻 Credits

**Created by:**
*   **Rafiul Haider**
*   **UID**: U02002983

## 📚 References

This project implements concepts from the following research:

> **"Least Angle Regression"**
> *   **Authors**: Bradley Efron, Trevor Hastie, Iain Johnstone, and Robert Tibshirani.
> *   **Publication**: Annals of Statistics.
> *   **Volume/Issue/Pages**: Vol. 32, No. 2, 407–499.
> *   **Year**: 2004.
> *   [Read the Paper (Project Euclid)](https://projecteuclid.org/euclid.aos/1083178935)
