import getpass

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif  # Keep f_classif
import plotly.express as px
import io
import base64
import os  # For environment variables
from dotenv import load_dotenv  # For .env file
from groq import Groq
from typing import Optional, Dict, Any, Tuple

load_dotenv()  # Load environment variables from .env file

# --- Groq Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or getpass.getpass("Groq API Key (recommend to set in .env): ")
LLM_MODEL = "llama-3.3-70b-versatile" #  Correct Groq model name (Llama 3 70B) -  VERY IMPORTANT
# Or, for Llama 2 70B:
# LLM_MODEL = "llama2-70b-4096"


@st.cache_resource
def get_groq_client():
    """Initializes and caches the Groq client."""
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set.  Set it as an environment variable or in a .env file.")
        return None
    return Groq(api_key=GROQ_API_KEY)


def get_llm_guidance(prompt: str, data_description: Optional[str] = None) -> str:
    """Gets guidance from Groq's LLM."""
    client = get_groq_client()
    if client is None:
        return "Groq client initialization failed. Check your API key."

    if data_description:
        full_prompt = f"Dataset Description:\n{data_description}\n\nUser Prompt:\n{prompt}"
    else:
        full_prompt = prompt

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            model=LLM_MODEL,
            temperature=0.5,  # Optional: Adjust for creativity
            max_tokens=1024,   # Optional: Limit response length
            top_p=1,           # Optional
            stop=None,          # Optional
            stream=False,       # Optional
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error during Groq API call: {e}")
        return "Groq API call failed. Check the error message."


# --- Data Preprocessing Functions (No Changes Here) ---

def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """Handles missing values."""
    if strategy == 'drop':
        return df.dropna()

    imputer = SimpleImputer(strategy=strategy)
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    categorical_cols = df.select_dtypes(exclude=np.number).columns
    for col in categorical_cols:
        most_frequent = df[col].mode()[0]
        df[col] = df[col].fillna(most_frequent)
    return df

def detect_and_handle_outliers(df: pd.DataFrame, method: str = 'IQR', threshold: float = 1.5) -> pd.DataFrame:
    """Detects and handles outliers (IQR method only)."""
    if method != 'IQR':
        st.error("Only IQR method is currently supported.")
        return df  # Return original DataFrame if method is not supported

    df_cleaned = df.copy()
    for col in df_cleaned.select_dtypes(include=np.number).columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Performs basic data cleaning."""
    df = df.drop_duplicates()
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].str.strip()
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Performs feature engineering."""
    if 'feature_1' in df.columns and 'feature_2' in df.columns:
        if pd.api.types.is_numeric_dtype(df['feature_1']) and pd.api.types.is_numeric_dtype(df['feature_2']):
            df['total_score'] = df['feature_1'] + df['feature_2']

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

def feature_selection(df: pd.DataFrame, target_column: str, k: int = 5) -> Optional[pd.DataFrame]:
    """Selects top k features using SelectKBest."""
    if target_column not in df.columns:
        st.error("Target column not found.")
        return None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y)

    numeric_X = X.select_dtypes(include=np.number)
    if numeric_X.shape[1] == 0:
        st.error("No numeric features for selection.")
        return None

    k = min(k, numeric_X.shape[1])  # Ensure k is within bounds
    selector = SelectKBest(score_func=f_classif, k=k)  # Or chi2 for classification with non-negative features
    try:
        selector.fit(numeric_X, y)
    except ValueError as e:
        st.error(f"Feature selection error: {e}.  This often happens if features have NaN values after other processing steps.  Review your data cleaning and imputation.")
        return None

    selected_features = numeric_X.columns[selector.get_support()].tolist()

    st.write("Feature Selection Scores (ANOVA F-statistic):")
    scores = pd.DataFrame({'Feature': numeric_X.columns, 'Score': selector.scores_}).sort_values(by='Score', ascending=False)
    st.dataframe(scores)
    return df[selected_features + [target_column]]

def train_model(df: pd.DataFrame, target_column: str, model_type: str = 'linear_regression', test_size: float = 0.2, random_state: int = 42) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Trains a model."""
    if target_column not in df.columns:
        st.error("Target column not found.")
        return None, None

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y)

    X = X.select_dtypes(include=np.number)
    if X.shape[1] == 0:
        st.error("No numeric features for training.")
        return None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'logistic_regression':
        model = LogisticRegression(solver='liblinear', random_state=random_state)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state, n_estimators=100)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=random_state, n_estimators=100)
    else:
        st.error("Invalid model type.")
        return None, None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {}
    if model_type == 'linear_regression':
        metrics['MSE'] = mean_squared_error(y_test, y_pred)
        metrics['R2'] = r2_score(y_test, y_pred)
    else:
        metrics['Accuracy'] = accuracy_score(y_test, y_pred)
        metrics['Classification Report'] = classification_report(y_test, y_pred)
        metrics['Confusion Matrix'] = confusion_matrix(y_test, y_pred)

    return model, metrics

# --- Data Visualization Functions ---

def plot_data(df: pd.DataFrame, plot_type: str, x_col: Optional[str] = None, y_col: Optional[str] = None, color_col: Optional[str] = None):
    """Generates plots using Plotly."""
    if plot_type == 'histogram':
        for col in df.select_dtypes(include=np.number).columns:
            fig = px.histogram(df, x=col, title=f'Histogram of {col}')
            st.plotly_chart(fig)

    elif plot_type == 'scatter':
        if x_col is None or y_col is None:
            st.error("x_col and y_col are required for scatter plots.")
            return
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f'Scatter Plot of {x_col} vs {y_col}')
        st.plotly_chart(fig)

    elif plot_type == 'box':
        if x_col is None:
            st.error("x_col is required for box plots.")
            return
        if y_col and df[y_col].dtype == 'object':
            fig = px.box(df, x=x_col, y=y_col, title=f'Box Plot of {x_col} by {y_col}')
        else:
            fig = px.box(df, x=x_col, title=f'Box Plot of {x_col}')
        st.plotly_chart(fig)

    elif plot_type == 'heatmap':
        corr_matrix = df.select_dtypes(include=np.number).corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title='Correlation Heatmap')
        st.plotly_chart(fig)

    elif plot_type == 'line':
        if x_col is None or y_col is None:
            st.error("x_col and y_col are required for line plots.")
            return
        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"Line plot of {y_col} over {x_col}")
        st.plotly_chart(fig)

    else:
        st.error("Invalid plot type selected.")

# --- Report Generation ---

def generate_report(df: pd.DataFrame, model=None, metrics: Optional[Dict[str, Any]] = None, data_description: Optional[str] = None) -> str:
    """Generates a comprehensive report."""
    report = "# Data Analysis Report\n\n"

    if data_description:
        report += f"**Dataset Description:** {data_description}\n\n"

    report += "## Data Overview\n\n"
    report += f"Number of rows: {df.shape[0]}\n\n"
    report += f"Number of columns: {df.shape[1]}\n\n"
    report += "Column Names:\n\n" + ', '.join(df.columns) + "\n\n"
    report += "Data Types:\n\n" + str(df.dtypes) + "\n\n"
    report += "Descriptive Statistics:\n\n"
    report += df.describe().to_markdown() + "\n\n"

    report += "## Missing Values\n\n"
    report += df.isnull().sum().to_markdown() + "\n\n"

    if model:
        report += "## Model Training\n\n"
        report += f"Model Type: {type(model).__name__}\n\n"
        report += "Evaluation Metrics:\n\n"
        if metrics:
            if 'Classification Report' in metrics:
                report += "Classification Report:\n\n"
                report += str(metrics['Classification Report']) + "\n\n"  # Already a string
            if 'Confusion Matrix' in metrics:
                report += "Confusion Matrix:\n\n"
                report += str(metrics['Confusion Matrix']) + "\n\n"
            for key, value in metrics.items():
                if key not in ['Classification Report', 'Confusion Matrix']:
                    report += f"- {key}: {value}\n\n"
    return report
# --- Download Function ---
def get_table_download_link(df: pd.DataFrame, filename: str = "data.csv", link_text: str = "Download data as CSV") -> str:
    """Generates a link to download a Pandas DataFrame as a CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'

def get_report_download_link(report: str, filename: str = "report.md", link_text: str = "Download report as Markdown") -> str:
    """Generates a link to download a string as a Markdown file."""
    b64 = base64.b64encode(report.encode()).decode()
    return f'<a href="data:file/markdown;base64,{b64}" download="{filename}">{link_text}</a>'


# --- Streamlit App ---

def main():
    """Main function: Streamlit app UI and logic."""
    st.title("Automated Data Analysis with LLM Guidance")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Original Data Preview:")
            st.dataframe(df.head())

            data_description = st.text_area("Provide a brief description of your dataset (optional):", height=100)

            # --- Sidebar for User Input ---
            st.sidebar.header("Data Preprocessing Options")
            missing_value_strategy = st.sidebar.selectbox("Handle Missing Values:", ["mean", "median", "most_frequent", "drop"])
            outlier_handling = st.sidebar.selectbox("Handle Outliers:", ["None", "IQR"])
            do_feature_engineering = st.sidebar.checkbox("Perform Feature Engineering", value=True)

            st.sidebar.header("Model Training Options")
            target_column = st.sidebar.selectbox("Select Target Column:", df.columns)
            model_type = st.sidebar.selectbox("Select Model Type:",
                                              ["linear_regression", "logistic_regression", "random_forest",
                                               "gradient_boosting"])
            do_feature_selection = st.sidebar.checkbox("Perform Feature Selection", value=True)
            if do_feature_selection:
                num_features = st.sidebar.slider("Number of Features to Select (k):", min_value=1,
                                                 max_value=len(df.columns) - 1, value=5, step=1)
            else:
                num_features = None

            st.sidebar.header("Visualization Options")
            plot_types = st.sidebar.multiselect("Select Plots to Generate:",
                                                ["histogram", "scatter", "box", "heatmap", "line"],
                                                default=["histogram", "heatmap"])
            if 'scatter' in plot_types or 'line' in plot_types or 'box' in plot_types:
                x_col = st.sidebar.selectbox("Select X-axis Column:", df.columns)
                y_col = st.sidebar.selectbox("Select Y-axis Column:", df.columns)
            else:
                x_col = None
                y_col = None
            if 'scatter' in plot_types or 'line' in plot_types:
                color_col = st.sidebar.selectbox("Select Color Column (optional):", [None] + list(df.columns))
            else:
                color_col = None

            # --- LLM Guidance Section ---
            st.sidebar.header("LLM Guidance")
            llm_prompt = st.sidebar.text_input("Ask the LLM for guidance:", "How should I handle missing values?")
            if st.sidebar.button("Get Guidance"):
                with st.spinner("LLM is thinking..."):
                    llm_response = get_llm_guidance(llm_prompt, data_description)  # No pipe needed
                st.sidebar.write(llm_response)

            # --- Data Processing and Analysis ---
            if st.button("Process Data and Analyze"):
                with st.spinner("Processing data..."):
                    df_processed = df.copy()
                    df_processed = data_cleaning(df_processed)
                    df_processed = handle_missing_values(df_processed, strategy=missing_value_strategy)
                    if outlier_handling == "IQR":
                        df_processed = detect_and_handle_outliers(df_processed)
                    if do_feature_engineering:
                        df_processed = feature_engineering(df_processed)
                    if do_feature_selection and target_column is not None and num_features is not None:
                        df_processed = feature_selection(df_processed, target_column, k=num_features)
                        if df_processed is None:
                            return  # Stop if feature selection fails.

                    st.write("### Processed Data Preview:")
                    st.dataframe(df_processed.head())
                    st.markdown(get_table_download_link(df_processed, filename="processed_data.csv"),
                                unsafe_allow_html=True)

                    # --- Model Training (Conditional) ---
                    if target_column:
                        model, metrics = train_model(df_processed, target_column, model_type=model_type)
                        if model:
                            st.write("### Model Training Results:")
                            if metrics:
                                for key, value in metrics.items():
                                    if key not in ['Classification Report', 'Confusion Matrix']:
                                        st.write(f"- {key}: {value}")
                                if 'Classification Report' in metrics:
                                    st.text(metrics['Classification Report'])
                                if 'Confusion Matrix' in metrics:
                                    st.write("Confusion Matrix:")
                                    st.write(metrics['Confusion Matrix'])

                        else:
                            st.error("Model training failed. See error messages above.")
                    else:
                        st.warning("Please select a target column to train a model.")

                # --- Data Visualization ---
                st.write("### Data Visualization")
                for plot_type in plot_types:
                    plot_data(df_processed, plot_type, x_col, y_col, color_col)

                # --- Report Generation ---
                with st.spinner("Generating report..."):
                    report = generate_report(df_processed, model, metrics, data_description)
                st.write("### Data Analysis Report")
                st.markdown(report, unsafe_allow_html=False)
                st.markdown(get_report_download_link(report), unsafe_allow_html=True)


        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()