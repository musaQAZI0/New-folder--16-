"""
Streamlit Frontend for SVM Iris Classifier
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Plotly imports with error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    px = None  # type: ignore
    go = None  # type: ignore
    PLOTLY_AVAILABLE = False
    st.warning("Plotly is not installed. Some visualizations may not be available. Install with: pip install plotly")

from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(
    page_title="SVM Iris Classifier",
    page_icon="🌸",
    layout="wide"
)

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        with open('svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('target_names.pkl', 'rb') as f:
            target_names = pickle.load(f)
        return model, scaler, feature_names, target_names
    except FileNotFoundError:
        st.error("Model files not found. Please run train_model.py first!")
        return None, None, None, None

# Load model
model, scaler, feature_names, target_names = load_model()

# Title and description
st.title("🌸 Iris Flower Classification with SVM")
st.markdown("""
This application uses a **Support Vector Machine (SVM)** classifier to predict the species of Iris flowers
based on their physical characteristics.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
**Lab 08: Support Vector Machine**

This project demonstrates:
- SVM classification
- Multiple kernel comparison
- Interactive predictions
- Data visualization
""")

st.sidebar.header("Dataset Info")
st.sidebar.write("""
**Iris Dataset**
- 150 samples
- 3 species (Setosa, Versicolor, Virginica)
- 4 features (Sepal & Petal measurements)
""")

if model is not None and scaler is not None and feature_names is not None and target_names is not None:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Dataset Explorer", "📈 Visualization", "ℹ️ About SVM"])

    # Tab 1: Prediction
    with tab1:
        st.header("Make a Prediction")
        st.write("Enter the flower measurements to predict its species:")

        col1, col2 = st.columns(2)

        with col1:
            sepal_length = st.slider(
                "Sepal Length (cm)",
                min_value=4.0,
                max_value=8.0,
                value=5.8,
                step=0.1
            )
            sepal_width = st.slider(
                "Sepal Width (cm)",
                min_value=2.0,
                max_value=4.5,
                value=3.0,
                step=0.1
            )

        with col2:
            petal_length = st.slider(
                "Petal Length (cm)",
                min_value=1.0,
                max_value=7.0,
                value=4.3,
                step=0.1
            )
            petal_width = st.slider(
                "Petal Width (cm)",
                min_value=0.1,
                max_value=2.5,
                value=1.3,
                step=0.1
            )

        # Create input array
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predict button
        if st.button("🔍 Predict Species", type="primary"):
            # Scale input
            input_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.decision_function(input_scaled)[0]

            # Display result
            species = target_names[prediction]

            st.success(f"### Predicted Species: **{species.upper()}**")

            # Display input summary
            st.write("#### Input Features:")
            input_df = pd.DataFrame(input_data, columns=feature_names)
            st.dataframe(input_df, use_container_width=True)

            # Display confidence scores
            st.write("#### Decision Function Scores:")
            scores_df = pd.DataFrame({
                'Species': target_names,
                'Score': prediction_proba
            })
            st.dataframe(scores_df, use_container_width=True)

            # Visualize prediction
            if PLOTLY_AVAILABLE and go is not None:
                fig = go.Figure(data=[
                    go.Bar(x=target_names, y=prediction_proba,
                          marker_color=['green' if i == prediction else 'lightblue'
                                       for i in range(len(target_names))])
                ])
                fig.update_layout(
                    title="Decision Function Scores",
                    xaxis_title="Species",
                    yaxis_title="Score",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Install Plotly to see interactive visualizations: `pip install plotly`")

    # Tab 2: Dataset Explorer
    with tab2:
        st.header("Explore the Iris Dataset")

        try:
            df = pd.read_csv('iris_dataset.csv')

            st.write("### Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Features", len(feature_names))
            with col3:
                st.metric("Classes", len(target_names))

            st.write("### Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            st.write("### Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)

            st.write("### Class Distribution")
            species_counts = df['species_name'].value_counts()
            if PLOTLY_AVAILABLE and px is not None:
                fig = px.pie(values=species_counts.values, names=species_counts.index,
                            title="Iris Species Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(species_counts)

        except FileNotFoundError:
            st.warning("Dataset file not found. Please run train_model.py first!")

    # Tab 3: Visualization
    with tab3:
        st.header("Data Visualization")

        try:
            df = pd.read_csv('iris_dataset.csv')

            st.write("### Feature Comparison")
            col1, col2 = st.columns(2)

            with col1:
                feature_x = st.selectbox("Select X-axis feature", list(feature_names), index=0)
            with col2:
                feature_y = st.selectbox("Select Y-axis feature", list(feature_names), index=1)

            if PLOTLY_AVAILABLE and px is not None:
                fig = px.scatter(df, x=feature_x, y=feature_y, color='species_name',
                               title=f"{feature_x} vs {feature_y}",
                               labels={'species_name': 'Species'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Install Plotly to see interactive scatter plot: `pip install plotly`")

            st.write("### PCA Visualization")
            X = df[list(feature_names)].values
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
            df_pca['species'] = df['species_name']

            if PLOTLY_AVAILABLE and px is not None:
                fig = px.scatter(df_pca, x='PC1', y='PC2', color='species',
                               title='PCA: First Two Principal Components',
                               labels={'species': 'Species'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Install Plotly to see interactive PCA plot: `pip install plotly`")

            st.write(f"**Explained Variance:** PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}")

        except FileNotFoundError:
            st.warning("Dataset file not found. Please run train_model.py first!")

    # Tab 4: About SVM
    with tab4:
        st.header("About Support Vector Machine")

        st.markdown("""
        ### What is SVM?

        **Support Vector Machine (SVM)** is a supervised machine learning algorithm used for classification and regression tasks.

        #### Key Concepts:

        1. **Hyperplane**: A decision boundary that separates different classes
        2. **Support Vectors**: Data points closest to the hyperplane that influence its position
        3. **Margin**: The distance between the hyperplane and the nearest data points
        4. **Kernel Trick**: Transforms data to higher dimensions for non-linear classification

        #### SVM Kernels:

        - **Linear**: For linearly separable data
        - **RBF (Radial Basis Function)**: Most popular, handles non-linear data well
        - **Polynomial**: For data with polynomial relationships

        #### Advantages:
        - Effective in high-dimensional spaces
        - Memory efficient (uses support vectors)
        - Versatile (different kernel functions)

        #### This Implementation:
        - Dataset: Iris (150 samples, 3 classes)
        - Features: Sepal length/width, Petal length/width
        - Preprocessing: StandardScaler for feature normalization
        - Model: SVC with optimized kernel
        """)

        try:
            st.image('model_comparison.png', caption='Model Performance Comparison')
        except:
            pass

else:
    st.error("⚠️ Please train the model first by running: `python train_model.py`")

# Footer
st.markdown("---")
st.markdown("**Lab 08: Support Vector Machine** | Built with Streamlit")
