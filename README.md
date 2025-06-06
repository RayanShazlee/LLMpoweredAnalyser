# Automated Data Analysis and Reporting Tool with LLM Guidance

## Features

*   **Data Upload:** Upload CSV files for analysis.
*   **Automated Data Preprocessing:**
    *   Handles missing values (mean, median, most frequent, or drop).
    *   Detects and removes outliers using the Interquartile Range (IQR) method.
    *   Cleans data (removes duplicates, standardizes column names).
*   **Feature Engineering:**
    *   Creates new features from existing ones (e.g., summing numerical columns).
    *   One-hot encodes categorical features.
*   **Feature Selection:** Selects the most relevant features using `SelectKBest` (ANOVA F-statistic or chi-squared).
*   **Model Training:**
    *   Trains machine learning models: Linear Regression, Logistic Regression, Random Forest, and Gradient Boosting.
    *   Performs train/test splitting and feature scaling.
    *   Displays evaluation metrics (MSE, R-squared, accuracy, classification report, confusion matrix).
*   **Interactive Data Visualization:**
    *   Generates interactive plots using Plotly:
        *   Histograms
        *   Scatter plots
        *   Box plots
        *   Heatmaps
        *   Line Plots
*   **LLM-Powered Guidance:**
    *   Integrates with the Groq API to provide intelligent guidance.
    *   Users can ask questions about their data and analysis process.
    *   The LLM offers context-aware suggestions and explanations.  _(Requires a Groq API key.)_
*   **Report Generation:**
    *   Automatically generates a comprehensive data analysis report in Markdown format.
    *   Includes data overview, preprocessing steps, model training results, and key findings.
*   **Downloadable Results:**
    *   Download processed data as a CSV file.
    *   Download the analysis report as a Markdown file.

## Technologies Used

*   **Frontend:** Streamlit
*   **Backend:** Python
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn
*   **Visualization:** Plotly
*   **LLM Integration:** Groq API (using the `groq` Python client)
*   **Environment Management:** Uses `.env` files for secure API key storage.

## Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/RayanShazlee/LLMpoweredAnalyser.git
    cd LLMpoweredAnalyser
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Groq API Key:**
    *   Obtain an API key from [Groq](https://console.groq.com/).
    *   Create a `.env` file in the project's root directory:
        ```
        GROQ_API_KEY="your_groq_api_key"
        ```
        Replace `"your_groq_api_key"` with your actual API key.  **Do not commit the `.env` file to version control!**  Add `.env` to your `.gitignore` file.

5.  **Run the application:**

    ```bash
    streamlit run app.py
    ```

    This will open the application in your web browser.

## Usage

1.  **Upload Data:**  Use the "Upload a CSV file" button to upload your data.
2.  **Data Description (Optional):**  Provide a brief description of your dataset in the text area. This helps the LLM provide better guidance.
3.  **Configure Options:** Use the sidebar to:
    *   Choose how to handle missing values and outliers.
    *   Enable/disable feature engineering and selection.
    *   Select a target column and model type for training.
    *   Choose which plots to generate.
    *   Ask the LLM for guidance.
4.  **Process Data and Analyze:** Click the "Process Data and Analyze" button to run the analysis pipeline.
5.  **Review Results:**
    *   View the processed data preview.
    *   Examine the model training results (if applicable).
    *   Explore the interactive visualizations.
    *   Read the generated data analysis report.
6.  **Download Results:** Download the processed data and the report using the provided links.

## Contributing

Contributions are welcome!  Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Make your changes and commit them (`git commit -m "Add your feature"`).
4.  Push to your branch (`git push origin feature/your-feature`).
5.  Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  (You'll need to create a `LICENSE` file and put the MIT License text in it.)

## Troubleshooting
*   **`RuntimeError: no running event loop` or `RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!`:** These errors are often related to PyTorch installation or conflicts. Try the solutions outlined in the previous responses (reinstalling PyTorch, creating a new virtual environment, etc.).

*  **Groq API Errors:** Make sure your `GROQ_API_KEY` is correctly set and that you have sufficient credits/access on your Groq account.  Check the Groq API documentation for specific error messages.

* **Model Not Found:** Ensure you're using a valid Groq model name (e.g., `llama2-70b-4096` or `llama3-70b-8192`). Consult the Groq documentation for the most up-to-date list of available models.
#   L L M p o w e r e d A n a l y s e r 
 
 
