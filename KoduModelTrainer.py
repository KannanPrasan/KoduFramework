import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Set up cache folder
CACHE_FOLDER = 'cache'
if not os.path.exists(CACHE_FOLDER):
    os.makedirs(CACHE_FOLDER)

def load_data(uploaded_files):
    dataframes = []
    encodings = ['utf-8', 'latin1', 'iso-8859-1']
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split('.')[-1].lower()
        for encoding in encodings:
            try:
                if ext == 'csv':
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                elif ext == 'parquet':
                    df = pd.read_parquet(uploaded_file)
                elif ext == 'xlsx' or ext == 'xls':
                    df = pd.read_excel(uploaded_file)
                elif ext == 'json':
                    df = pd.read_json(uploaded_file)
                else:
                    st.error(f"Unsupported file format: {ext}")
                    return None
                dataframes.append(df)
                break
            except UnicodeDecodeError:
                continue
        else:
            st.error(f"Failed to read file {uploaded_file.name} with supported encodings.")
            return None
    return dataframes

def save_to_cache(df, filename):
    cache_file = os.path.join(CACHE_FOLDER, filename)
    df.to_csv(cache_file, index=False)
    return cache_file

def load_from_cache(filename):
    cache_file = os.path.join(CACHE_FOLDER, filename)
    return pd.read_csv(cache_file)

def perform_eda(df):
    st.subheader("Univariate Analysis")

    # Flexible Visualization Options
    plot_types = st.multiselect("Select visualization types", ["Histogram", "Boxplot", "Violinplot"])
    numerical_cols = df.select_dtypes(include=['number']).columns

    for col in numerical_cols:
        st.write(f"Visualizations for {col}")
        for plot_type in plot_types:
            fig, ax = plt.subplots()
            if plot_type == "Histogram":
                sns.histplot(df[col], kde=True, ax=ax)
            elif plot_type == "Boxplot":
                sns.boxplot(x=df[col], ax=ax)
            elif plot_type == "Violinplot":
                sns.violinplot(x=df[col], ax=ax)
            st.pyplot(fig)

    # Customizable Pairplots
    st.subheader("Customizable Pairplots")
    pairplot_cols = st.multiselect("Select columns for pairplot", numerical_cols, default=numerical_cols)
    if len(pairplot_cols) > 1:
        diag_kind = st.selectbox("Select diagonal type for pairplot", ["auto", "hist", "kde"])
        hue = st.selectbox("Select hue for pairplot (categorical column)", [None] + df.select_dtypes(include=['object']).columns.tolist())
        st.write(f"Pairplot for selected columns: {pairplot_cols}")
        fig = sns.pairplot(df[pairplot_cols], diag_kind=diag_kind, hue=hue)
        st.pyplot(fig)

    # Data Distribution Insights
    st.subheader("Data Distribution Insights")
    st.write("Skewness and Kurtosis for numerical columns:")
    dist_insights = pd.DataFrame({
        "Column": numerical_cols,
        "Skewness": [df[col].skew() for col in numerical_cols],
        "Kurtosis": [df[col].kurtosis() for col in numerical_cols]
    })
    st.dataframe(dist_insights)

def view_and_export():
    st.subheader("View & Export Cached Files")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file to view/export", cached_files)
        df = load_from_cache(selected_file)
        st.write(df.head())
        st.download_button("Download CSV", df.to_csv(index=False), file_name=selected_file, mime="text/csv")




# Sidebar for module selection
st.sidebar.title("Kodu ML Modules")
#st.set_page_config(layout="wide")
st.title("Kodu ML Framework")
module = st.sidebar.radio("Select a Module",
                          ("Data Collection", "Data Understanding",
                           "Preprocessing - Cast Datatypes", "Preprocessing - Rename Columns",
                           "Preprocessing - Drop Columns", "Preprocessing - Data Cleanup",
                           "Exploratory Data Analysis (EDA)", "Outlier Detection and Handling",
                           "Machine Learning Algorithm Recommendation", "Linear Regression Model",
                           "Export Updated File"))

# Data Collection Module
if module == "Data Collection":
    st.title("Data Collection")
    uploaded_files = st.file_uploader("Upload your file(s)", type=['csv', 'parquet', 'xlsx', 'xls', 'json'],
                                      accept_multiple_files=True)
    if uploaded_files:
        dataframes = load_data(uploaded_files)
        file_options = [uploaded_file.name for uploaded_file in uploaded_files]
        selected_file = st.selectbox("Select file for machine learning", file_options)
        if st.button("Load Selected File"):
            selected_df = dataframes[file_options.index(selected_file)]
            cache_file = save_to_cache(selected_df, selected_file)
            st.success(f"File saved to cache: {cache_file}")

# Data Understanding Module
elif module == "Data Understanding":
    st.title("Data Understanding")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            st.write("Shape:", df.shape)
            st.write("Description:", df.describe())
            numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            st.write("Numerical Columns:", numerical_cols)
            for col in categorical_cols:
                st.write(f"Categorical Column: {col}")
                st.selectbox(f"Unique values for {col}", df[col].unique())

# Preprocessing - Cast Datatypes Module
elif module == "Preprocessing - Cast Datatypes":
    st.title("Preprocessing - Cast Datatypes")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            column_types = {}
            for col in df.columns:
                col_type = df[col].dtype
                column_types[col] = st.selectbox(f"Select new datatype for {col} ({col_type})", ['','int', 'float', 'str'])
            if st.button("Convert"):
                for col, new_type in column_types.items():
                    if new_type == 'int':
                        df[col].fillna(0, inplace=True)  # Fill NaNs with 0 before converting to int
                        df[col] = df[col].astype(int)
                    elif new_type == 'float':
                        df[col].fillna(0.0, inplace=True)  # Fill NaNs with 0.0 before converting to float
                        df[col] = df[col].astype(float)
                    elif new_type == 'str':
                        df[col] = df[col].astype(str)
                    else:
                        df[col]=df[col]
                cache_file = save_to_cache(df, selected_file)
                st.success(f"File saved to cache with new datatypes: {cache_file}")
            elif st.button("Discard"):
                st.warning("Changes discarded")

# Preprocessing - Rename Columns Module
elif module == "Preprocessing - Rename Columns":
    st.title("Preprocessing - Rename Columns")
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
elif module == "Preprocessing - Drop Columns":
    st.title("Preprocessing - Drop Columns")
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
elif module == "Preprocessing - Data Cleanup":
    st.title("Preprocessing - Data Cleanup")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            null_columns = df.columns[df.isnull().any()].tolist()
            fillna_options = [''] + ['Mean', 'Median', 'Mode', 'Custom']  # Default value is empty
            fillna_values = {}
            custom_values = {}
            for col in null_columns:
                fillna_method = st.selectbox(f"Choose method to fill NaN values for {col}", fillna_options)
                fillna_values[col] = fillna_method
                if fillna_method == 'Custom':
                    if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                        custom_values[col] = st.number_input(f"Custom value for {col}", format="%.2f")
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

elif module == "Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)
            perform_eda(df)

# Outlier Detection and Handling Module
elif module == "Outlier Detection and Handling":
    st.title("Outlier Detection and Handling")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)

            numerical_cols = df.select_dtypes(include=['number']).columns
            if len(numerical_cols) == 0:
                st.error("No numerical columns available for outlier detection.")
            else:
                # Outlier Detection
                st.subheader("Outlier Detection")
                outlier_method = st.selectbox("Select method for detecting outliers", ["IQR", "Z-Score"])
                outlier_cols = st.multiselect("Select columns for outlier detection", numerical_cols)
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

                # Outlier Handling
                st.subheader("Outlier Handling")
                handling_method = st.selectbox("Select method to handle outliers", ["Remove", "Replace"])
                if handling_method == "Replace":
                    replace_value = st.number_input("Enter value to replace outliers", value=0.0)

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

# Machine Learning Recommendation Module
elif module == "Machine Learning Algorithm Recommendation":
    st.title("Machine Learning Algorithm Recommendation")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)

            st.subheader("Step 1: Select Target Column")
            target_col = st.selectbox("Select target variable (or None for clustering)", [None] + df.columns.tolist())

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

            # Recommendations based on task type
            st.subheader("Recommended Algorithms")
            if task_type == "Regression":
                if num_features > 20:
                    st.write("- **Lasso Regression**: For feature selection in high-dimensional data.")
                    st.write("- **Ridge Regression**: Handles multicollinearity well.")
                st.write("- **Linear Regression**: If data is linear and small.")
                st.write("- **Random Forest Regressor**: Handles non-linearity and works well with medium-sized data.")
                st.write(
                    "- **Gradient Boosting (e.g., XGBoost, LightGBM)**: For non-linear relationships and large datasets.")
            elif task_type == "Classification":
                if cat_features > 0:
                    st.write("- **Logistic Regression**: If linear and small.")
                    st.write("- **Random Forest Classifier**: Handles mixed data types well.")
                    st.write("- **Gradient Boosting (e.g., XGBoost, LightGBM)**: For tabular data.")
                    st.write("- **Naive Bayes**: Works well with categorical data.")
                else:
                    st.write("- **SVM (Support Vector Machine)**: Good for small datasets.")
                    st.write("- **K-Nearest Neighbors (KNN)**: Simple and intuitive.")
            elif task_type == "Clustering":
                st.write("- **K-Means Clustering**: Simple and scalable for numerical data.")
                st.write("- **DBSCAN**: Detects clusters of varying shapes and sizes.")
                st.write("- **Hierarchical Clustering**: Useful for small datasets.")

            st.subheader("Additional Notes")
            if num_rows < 1000:
                st.write(
                    "- **Small Dataset**: Consider simpler models like Logistic Regression, Linear Regression, or KNN.")
            elif num_rows > 10000:
                st.write(
                    "- **Large Dataset**: Consider scalable models like Random Forest, Gradient Boosting, or Neural Networks.")

# Linear Regression Machine Learning Module
elif module == "Linear Regression Model":
    st.title("Linear Regression Machine Learning")
    cached_files = os.listdir(CACHE_FOLDER)
    if cached_files:
        selected_file = st.selectbox("Select file from cache", cached_files)
        if selected_file:
            df = load_from_cache(selected_file)

            st.subheader("Step 1: Select Features and Target")
            features = st.multiselect("Select Features (Independent Variables)", df.columns.tolist())
            target = st.selectbox("Select Target (Dependent Variable)",
                                  [col for col in df.columns if col not in features])

            if features and target:

                # Step 2: Train-Test Split
                X = df[features]
                y = df[target]
                test_size = st.slider("Test Set Percentage", 0.1, 0.5, 0.2, step=0.05)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                # Step 3: Train the Model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Step 4: Evaluate the Model
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"**R² Score:** {r2:.2f}")
                #st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")

                # Display Coefficients
                st.write("Model Coefficients:")
                coefficients = pd.DataFrame({
                    "Feature": features,
                    "Coefficient": model.coef_
                })
                st.write(coefficients)

                # Step 5: Predict New Values
                st.subheader("Make Predictions")
                new_values = {}
                for feature in features:
                    feature_dtype = df[feature].dtype
                    if pd.api.types.is_numeric_dtype(feature_dtype):
                        new_values[feature] = st.number_input(f"Enter value for {feature}", format="%.4f")
                    else:
                        new_values[feature] = st.text_input(f"Enter value for {feature}")

                if st.button("Predict"):
                    input_data = pd.DataFrame([new_values])
                    prediction = model.predict(input_data)[0]
                    st.success(f"Predicted Value: {prediction:.4f}")

elif module == "Export Updated File":
    view_and_export()