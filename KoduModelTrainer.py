import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Set up cache folder
CACHE_FOLDER = 'cache'
if not os.path.exists(CACHE_FOLDER):
    os.makedirs(CACHE_FOLDER)

# Load from cache function
def load_from_cache(filename):
    cache_file = os.path.join(CACHE_FOLDER, filename)
    return pd.read_csv(cache_file)

# Perform EDA using ydata-profiling
def perform_eda_with_ydata(df):
    st.subheader("📊 Exploratory Data Analysis (EDA) with ydata-profiling")
    profile = ProfileReport(df, explorative=True)
    st.write("Generating report... This may take a few moments.")
    profile.to_file("ydata_report.html")
    with open("ydata_report.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=1000, scrolling=True)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .stHeader {
        color: #4f8bf9;
    }
    .stButton button {
        background-color: #4f8bf9;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #3a6bbf;
    }
    .stMarkdown h1 {
        color: #4f8bf9;
    }
    .stMarkdown h2 {
        color: #4f8bf9;
    }
    .stMarkdown h3 {
        color: #4f8bf9;
    }
    .stSidebar {
        background-color: #ffffff;
    }
    .stSidebar .sidebar-content {
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for module selection
st.sidebar.title("🚀 Kodu ML Modules")
st.sidebar.markdown("**Select a module to get started:**")
module = st.sidebar.radio(
    "",
    ("📂 Data Collection", "🔍 Data Understanding", "🛠️ Preprocessing - Cast Datatypes",
     "🛠️ Preprocessing - Rename Columns", "🛠️ Preprocessing - Drop Columns",
     "🛠️ Preprocessing - Data Cleanup", "📊 Exploratory Data Analysis (EDA)",
     "📈 Outlier Detection and Handling", "🤖 Machine Learning Algorithm Recommendation",
     "📉 Linear Regression Model", "📤 Export Updated File")
)

# Main title and description
st.title("🤖 Kodu ML Framework")
st.markdown("""
    Welcome to the **Kodu ML Framework**! This tool helps you perform end-to-end machine learning tasks, 
    from data collection to model training and evaluation. Use the sidebar to navigate through the modules.
""")

# Data Collection Module
if module == "📂 Data Collection":
    st.header("📂 Data Collection")
    uploaded_files = st.file_uploader(
        "Upload your file(s)",
        type=['csv', 'parquet', 'xlsx', 'xls', 'json'],
        accept_multiple_files=True
    )
    if uploaded_files:
        dataframes = load_data(uploaded_files)
        file_options = [uploaded_file.name for uploaded_file in uploaded_files]
        selected_file = st.selectbox("Select file for machine learning", file_options)
        if st.button("Load Selected File"):
            selected_df = dataframes[file_options.index(selected_file)]
            cache_file = save_to_cache(selected_df, selected_file)
            st.success(f"File saved to cache: {cache_file}")

# Data Understanding Module
elif module == "🔍 Data Understanding":
    st.header("🔍 Data Understanding")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            st.write("**Shape:**", df.shape)
            st.write("**Description:**", df.describe())
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            st.write("**Numerical Columns:**", numerical_cols)
            for col in categorical_cols:
                st.write(f"**Categorical Column:** {col}")
                st.selectbox(f"Unique values for {col}", df[col].unique())

# Preprocessing - Cast Datatypes Module
elif module == "🛠️ Preprocessing - Cast Datatypes":
    st.header("🛠️ Preprocessing - Cast Datatypes")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            column_types = {}
            for col in df.columns:
                col_type = df[col].dtype
                column_types[col] = st.selectbox(
                    f"Select new datatype for {col} ({col_type})",
                    ['', 'int', 'float', 'str']
                )
            if st.button("Convert"):
                for col, new_type in column_types.items():
                    if new_type == 'int':
                        df[col].fillna(0, inplace=True)
                        df[col] = df[col].astype(int)
                    elif new_type == 'float':
                        df[col].fillna(0.0, inplace=True)
                        df[col] = df[col].astype(float)
                    elif new_type == 'str':
                        df[col] = df[col].astype(str)
                cache_file = save_to_cache(df, selected_file)
                st.success(f"File saved to cache with new datatypes: {cache_file}")
            elif st.button("Discard"):
                st.warning("Changes discarded")

# Preprocessing - Rename Columns Module
elif module == "🛠️ Preprocessing - Rename Columns":
    st.header("🛠️ Preprocessing - Rename Columns")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            new_columns = {}
            for col in df.columns:
                new_columns[col] = st.text_input(f"Rename {col}", col)
            if st.button("Rename"):
                df.rename(columns=new_columns, inplace=True)
                cache_file = save_to_cache(df, selected_file)
                st.success(f"File saved to cache with renamed columns: {cache_file}")
            elif st.button("Discard"):
                st.warning("Changes discarded")

# Preprocessing - Drop Columns Module
elif module == "🛠️ Preprocessing - Drop Columns":
    st.header("🛠️ Preprocessing - Drop Columns")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            drop_columns = []
            for col in df.columns:
                if st.checkbox(f"Drop {col}"):
                    drop_columns.append(col)
            if st.button("Drop"):
                df.drop(columns=drop_columns, inplace=True)
                cache_file = save_to_cache(df, selected_file)
                st.success(f"File saved to cache with dropped columns: {cache_file}")
            elif st.button("Discard"):
                st.warning("Changes discarded")

# Preprocessing - Data Cleanup Module
elif module == "🛠️ Preprocessing - Data Cleanup":
    st.header("🛠️ Preprocessing - Data Cleanup")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            null_columns = df.columns[df.isnull().any()].tolist()
            fillna_options = ['', 'Mean', 'Median', 'Mode', 'Custom']
            fillna_values = {}
            custom_values = {}
            for col in null_columns:
                fillna_method = st.selectbox(
                    f"Choose method to fill NaN values for {col}",
                    fillna_options
                )
                fillna_values[col] = fillna_method
                if fillna_method == 'Custom':
                    if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                        custom_values[col] = st.number_input(
                            f"Custom value for {col}",
                            format="%.2f"
                        )
                    elif df[col].dtype == 'object':
                        custom_values[col] = st.text_input(f"Custom value for {col}")
            if st.button("FillNa"):
                for col, method in fillna_values.items():
                    if method == 'Mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == 'Median':
                        df[col].fillna(df[col].median(), inplace=True)
                    elif method == 'Mode':
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif method == 'Custom':
                        df[col].fillna(custom_values[col], inplace=True)
                cache_file = save_to_cache(df, selected_file)
                st.success(f"File saved to cache with filled NaN values: {cache_file}")
            elif st.button("Discard"):
                st.warning("Changes discarded")

# Exploratory Data Analysis (EDA) Module
elif module == "📊 Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis (EDA)")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            perform_eda_with_ydata(df)

# Outlier Detection and Handling Module
elif module == "📈 Outlier Detection and Handling":
    st.header("📈 Outlier Detection and Handling")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            numerical_cols = df.select_dtypes(include=['number']).columns
            if len(numerical_cols) == 0:
                st.error("No numerical columns available for outlier detection.")
            else:
                st.subheader("Outlier Detection")
                outlier_method = st.selectbox(
                    "Select method for detecting outliers",
                    ["IQR", "Z-Score"]
                )
                outlier_cols = st.multiselect(
                    "Select columns for outlier detection",
                    numerical_cols
                )
                outlier_threshold = st.number_input(
                    "Set threshold for outliers (default is 1.5 for IQR, 3 for Z-Score)",
                    value=1.5 if outlier_method == "IQR" else 3.0
                )
                if st.button("Detect Outliers"):
                    outlier_summary = {}
                    for col in outlier_cols:
                        if outlier_method == "IQR":
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = df[(df[col] < Q1 - outlier_threshold * IQR) | (df[col] > Q3 + outlier_threshold * IQR)]
                        elif outlier_method == "Z-Score":
                            mean = df[col].mean()
                            std = df[col].std()
                            outliers = df[(df[col] - mean).abs() > outlier_threshold * std]
                        outlier_summary[col] = len(outliers)
                    st.write("Outlier Summary (Number of Outliers):")
                    st.json(outlier_summary)

                st.subheader("Outlier Handling")
                handling_method = st.selectbox(
                    "Select method to handle outliers",
                    ["Remove", "Replace"]
                )
                if handling_method == "Replace":
                    replace_value = st.number_input(
                        "Enter value to replace outliers",
                        value=0.0
                    )
                if st.button("Handle Outliers"):
                    for col in outlier_cols:
                        if outlier_method == "IQR":
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = (df[col] < Q1 - outlier_threshold * IQR) | (df[col] > Q3 + outlier_threshold * IQR)
                        elif outlier_method == "Z-Score":
                            mean = df[col].mean()
                            std = df[col].std()
                            outliers = (df[col] - mean).abs() > outlier_threshold * std
                        if handling_method == "Remove":
                            df = df[~outliers]
                        elif handling_method == "Replace":
                            df.loc[outliers, col] = replace_value
                    cache_file = save_to_cache(df, selected_file)
                    st.success(f"Outliers handled successfully. Updated file saved to cache: {cache_file}")

# Machine Learning Algorithm Recommendation Module
elif module == "🤖 Machine Learning Algorithm Recommendation":
    st.header("🤖 Machine Learning Algorithm Recommendation")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            st.subheader("Step 1: Select Target Column")
            target_col = st.selectbox(
                "Select target variable (or None for clustering)",
                [None] + df.columns.tolist()
            )
            if target_col:
                target_type = df[target_col].dtype
                if pd.api.types.is_numeric_dtype(target_type):
                    task_type = "Regression"
                elif pd.api.types.is_categorical_dtype(target_type) or df[target_col].nunique() < 10:
                    task_type = "Classification"
                else:
                    task_type = "Classification (High Cardinality)"
            else:
                task_type = "Clustering"
            st.write(f"Detected task type: **{task_type}**")

            st.subheader("Step 2: Dataset Characteristics")
            num_features = df.select_dtypes(include=['number']).shape[1]
            cat_features = df.select_dtypes(include=['object']).shape[1]
            num_rows = df.shape[0]
            st.write(f"**Number of Rows:** {num_rows}")
            st.write(f"**Number of Numerical Features:** {num_features}")
            st.write(f"**Number of Categorical Features:** {cat_features}")

            st.subheader("Recommended Algorithms")
            if task_type == "Regression":
                st.write("- **Linear Regression**: If data is linear and small.")
                st.write("- **Random Forest Regressor**: Handles non-linearity and works well with medium-sized data.")
                st.write("- **Gradient Boosting Regressor**: For non-linear relationships and large datasets.")
            elif task_type == "Classification":
                st.write("- **Logistic Regression**: If linear and small.")
                st.write("- **Random Forest Classifier**: Handles mixed data types well.")
                st.write("- **Gradient Boosting Classifier**: For tabular data.")
            elif task_type == "Clustering":
                st.write("- **K-Means Clustering**: Simple and scalable for numerical data.")
                st.write("- **DBSCAN**: Detects clusters of varying shapes and sizes.")

# Linear Regression Model Module
elif module == "📉 Linear Regression Model":
    st.header("📉 Linear Regression Model")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            st.subheader("Step 1: Select Features and Target")
            features = st.multiselect(
                "Select Features (Independent Variables)",
                df.columns.tolist()
            )
            target = st.selectbox(
                "Select Target (Dependent Variable)",
                [col for col in df.columns if col not in features]
            )
            if features and target:
                X = df[features]
                y = df[target]
                test_size = st.slider("Test Set Percentage", 0.1, 0.5, 0.2, step=0.05)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"**R² Score:** {r2:.2f}")
                st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
                st.write("**Model Coefficients:**")
                coefficients = pd.DataFrame({
                    "Feature": features,
                    "Coefficient": model.coef_
                })
                st.write(coefficients)

# Export Updated File Module
elif module == "📤 Export Updated File":
    st.header("📤 Export Updated File")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            st.write(df.head())
            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                file_name=selected_file,
                mime="text/csv"
            )

