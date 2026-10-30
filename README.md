# College Student SGPA Prediction

## About the Project

This is a machine learning mini-project that predicts the estimated SGPA of a college student based on academic performance and study habits.

The application takes details such as attendance, study hours, internal marks, assignment completion, previous SGPA, current backlogs, lab performance, and class participation. Based on these inputs, it predicts an estimated SGPA out of 10.

The project is built using Python, Scikit-learn, Pandas, and Streamlit.

## Features

- Predicts estimated semester SGPA.
- Takes academic details through a simple web interface.
- Uses Linear Regression and Random Forest Regression.
- Shows model evaluation results.
- Allows users to explore the sample dataset.
- Generates sample data for project demonstration.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

## Requirements

Make sure Python is installed on your computer.

You can check the installed version by running:

```bash
python --version
```

## Installation and Setup

### Step 1: Open the Project Folder

Open the project folder in VS Code.

Open the terminal using:

**Terminal → New Terminal**

### Step 2: Create a Virtual Environment

Run the following command:

```bash
python -m venv .venv
```

### Step 3: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install the Required Libraries

Run:

```bash
python -m pip install --upgrade pip
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

Start the Streamlit application with:

```bash
python -m streamlit run app.py
```

After running the command, open the local address displayed in the terminal. Usually, it is:

`http://localhost:8501`

## How to Use the Application

1. Open the application in your browser.
2. Go to the SGPA prediction section.
3. Enter the student's academic details.
4. Select the required machine learning model, if the option is available.
5. Click the prediction button.
6. View the estimated SGPA.
7. Explore the dataset and model evaluation results in the corresponding section.

## Machine Learning Models

Two regression algorithms are used in this project:

**1. Linear Regression**

This model estimates SGPA by learning the relationship between the input features and the target value.

**2. Random Forest Regressor**

This model uses multiple decision trees to generate a prediction. It can learn more complex relationships between academic features.

The models are evaluated using a train-test split and metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.

## Dataset

The project uses a sample dataset generated for demonstration and model training. It contains academic features and a target column representing the current semester SGPA.

The dataset is not collected from actual college students. Therefore, the predictions should be treated as estimates for educational demonstration rather than actual academic results.

## Project Structure

```text
College_Student_SGPA_Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT_OUTLINE.md
│
└── data/
    └── college_student_sample.csv
```

The sample CSV file may be generated automatically when the application runs for the first time.

## Limitations

- The dataset contains generated sample data rather than real student records.
- Prediction accuracy depends on the quality and patterns of the training data.
- The predicted SGPA is only an estimate and may differ from the actual result.
- The project is intended for learning and demonstration purposes.

## Conclusion

This project demonstrates how machine learning can be used to estimate a college student's SGPA from academic performance indicators. It also provides practical experience with data preprocessing, regression models, model evaluation, and building a simple web application using Streamlit.