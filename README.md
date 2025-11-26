# 🌸 Iris Flower Classification with SVM

A machine learning web application that uses Support Vector Machine (SVM) to classify Iris flowers based on their physical characteristics.

## 📋 Project Overview

This project demonstrates the implementation of a Support Vector Machine classifier for the classic Iris dataset, complete with an interactive web interface built using Streamlit.

### Features

- ✨ Interactive prediction interface
- 📊 Dataset exploration and visualization
- 📈 Multiple SVM kernel comparison (Linear, RBF, Polynomial)
- 🎯 Real-time predictions with confidence scores
- 📉 PCA visualization
- 🎨 Modern, user-friendly UI

## 🗂️ Dataset

**Iris Dataset** - A classic dataset in machine learning
- **Samples**: 150
- **Features**: 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
- **Classes**: 3 (Setosa, Versicolor, Virginica)

## 🛠️ Technologies Used

- **Python 3.8+**
- **scikit-learn** - Machine learning library
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations
- **Matplotlib & Seaborn** - Static visualizations

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/svm-iris-classifier.git
   cd svm-iris-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`

## 🚀 Usage

### Training the Model

Run the training script to train SVM models with different kernels:

```bash
python train_model.py
```

This will:
- Load the Iris dataset
- Train SVM models with Linear, RBF, and Polynomial kernels
- Compare model performances
- Save the best model and necessary files

### Running the Web App

Launch the Streamlit application:

```bash
streamlit run app.py
```

The app provides four main sections:

1. **🔮 Prediction** - Make predictions on new flower measurements
2. **📊 Dataset Explorer** - Explore the Iris dataset statistics
3. **📈 Visualization** - Visualize feature relationships and PCA
4. **ℹ️ About SVM** - Learn about Support Vector Machines

## 📁 Project Structure

```
svm-iris-classifier/
│
├── train_model.py          # Model training script
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
│
├── svm_model.pkl          # Trained SVM model (generated)
├── scaler.pkl             # Feature scaler (generated)
├── feature_names.pkl      # Feature names (generated)
├── target_names.pkl       # Target class names (generated)
├── iris_dataset.csv       # Iris dataset (generated)
└── model_comparison.png   # Model comparison plot (generated)
```

## 🧠 How SVM Works

**Support Vector Machine (SVM)** is a supervised learning algorithm that:

1. **Finds a hyperplane** that best separates different classes
2. **Maximizes the margin** between classes
3. **Uses support vectors** (closest points) to define the decision boundary
4. **Applies kernel tricks** to handle non-linear data

### Kernels Used

- **Linear Kernel**: For linearly separable data
- **RBF (Radial Basis Function)**: Handles non-linear patterns
- **Polynomial Kernel**: Captures polynomial relationships

## 📊 Model Performance

The model achieves high accuracy on the Iris dataset:
- Training with multiple kernels
- Cross-validation for robust evaluation
- Standardized features for better performance

## 🎯 Key Learnings

This project demonstrates:

1. **Classification** with Support Vector Machines
2. **Feature scaling** and preprocessing
3. **Model comparison** across different kernels
4. **Web deployment** with Streamlit
5. **Interactive visualizations** with Plotly
6. **Best practices** in ML project structure

## 🔮 Future Enhancements

- [ ] Add more classification datasets
- [ ] Implement cross-validation visualization
- [ ] Add model hyperparameter tuning interface
- [ ] Deploy to cloud platform (Streamlit Cloud/Heroku)
- [ ] Add batch prediction from CSV upload
- [ ] Implement confusion matrix visualization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created for **Lab 08: Support Vector Machine**

## 📚 References

- [Scikit-learn SVM Documentation](https://scikit-learn.org/stable/modules/svm.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)

---

**Built with ❤️ using Python and Streamlit**
