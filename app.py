from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_PATH = DATA_DIR / "college_student_sample.csv"

st.set_page_config(page_title="College Student SGPA Prediction", layout="wide")

NUMERIC_FEATURES = [
    "attendance_pct",
    "study_hours_per_day",
    "internal_marks_pct",
    "assignment_completion_pct",
    "previous_sgpa",
    "backlogs",
    "lab_performance_pct",
]
CATEGORICAL_FEATURES = ["class_participation"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "current_semester_sgpa"


def create_demo_dataset(path: Path, rows: int = 600) -> pd.DataFrame:
    """Create clearly labelled synthetic demo data; not real college records."""
    import numpy as np

    rng = np.random.default_rng(42)
    attendance = rng.uniform(45, 100, rows)
    study_hours = rng.uniform(0.5, 8, rows)
    internal = rng.uniform(25, 100, rows)
    assignments = rng.uniform(20, 100, rows)
    previous_sgpa = rng.uniform(4.0, 10.0, rows)
    backlogs = rng.choice([0, 1, 2, 3, 4], size=rows, p=[0.60, 0.18, 0.11, 0.07, 0.04])
    lab = rng.uniform(25, 100, rows)
    participation = rng.choice(["Low", "Average", "Good"], size=rows, p=[0.2, 0.5, 0.3])
    participation_score = pd.Series(participation).map({"Low": -0.35, "Average": 0.0, "Good": 0.35}).to_numpy()

    # Synthetic relationship used only to make a working demonstration.
    score = (
        0.9
        + 0.018 * attendance
        + 0.16 * study_hours
        + 0.018 * internal
        + 0.009 * assignments
        + 0.27 * previous_sgpa
        - 0.38 * backlogs
        + 0.012 * lab
        + participation_score
        + rng.normal(0, 0.65, rows)
    )
    sgpa = np.clip(score, 0.0, 10.0).round(2)

    df = pd.DataFrame({
        "attendance_pct": attendance.round(1),
        "study_hours_per_day": study_hours.round(1),
        "internal_marks_pct": internal.round(1),
        "assignment_completion_pct": assignments.round(1),
        "previous_sgpa": previous_sgpa.round(2),
        "backlogs": backlogs,
        "lab_performance_pct": lab.round(1),
        "class_participation": participation,
        "current_semester_sgpa": sgpa,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


@st.cache_data
def load_data(path_string: str) -> pd.DataFrame:
    path = Path(path_string)
    if not path.exists():
        return create_demo_dataset(path)
    return pd.read_csv(path)


@st.cache_resource
def train_models(data: pd.DataFrame):
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200, random_state=42, min_samples_leaf=2
            )),
        ]),
    }
    fitted = {}
    metrics = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        fitted[name] = model
        metrics.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": mean_squared_error(y_test, preds) ** 0.5,
            "R²": r2_score(y_test, preds),
        })
    return fitted, pd.DataFrame(metrics), X_test, y_test


st.title(" College Student SGPA Prediction")
st.write(
    "A Machine Learning Mini Project that estimates a college student's "
    "current-semester SGPA (0–10) from academic and study-related inputs."
)


try:
    data = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Could not load or create the sample dataset: {exc}")
    st.stop()

models, metrics, X_test, y_test = train_models(data)

tab_predict, tab_eval, tab_about = st.tabs(
    ["🔮 Predict SGPA", "📊 Dataset & Evaluation", "ℹ️ About the Project"]
)

with tab_predict:
    st.subheader("Enter college student information")
    st.caption(
        "Use academic information that would be available before the semester result is known. "
    )
    with st.form("student_form"):
        col1, col2 = st.columns(2)
        with col1:
            attendance = st.slider("Attendance (%)", 0, 100, 85)
            study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
            internal = st.slider("Internal assessment marks (%)", 0, 100, 75)
            assignments = st.slider("Assignment completion (%)", 0, 100, 90)
        with col2:
            previous_sgpa = st.slider("Previous semester SGPA", 0.0, 10.0, 7.5, 0.1)
            backlogs = st.selectbox("Current backlogs", [0, 1, 2, 3, 4, 5, 6])
            lab = st.slider("Practical / lab performance (%)", 0, 100, 80)
            participation = st.selectbox("Class participation", ["Low", "Average", "Good"], index=1)
        model_choice = st.selectbox("Choose prediction model", list(models.keys()), index=1)
        submitted = st.form_submit_button("Predict expected SGPA", type="primary")

    if submitted:
        student = pd.DataFrame([{
            "attendance_pct": attendance,
            "study_hours_per_day": study_hours,
            "internal_marks_pct": internal,
            "assignment_completion_pct": assignments,
            "previous_sgpa": previous_sgpa,
            "backlogs": backlogs,
            "lab_performance_pct": lab,
            "class_participation": participation,
        }])
        prediction = float(models[model_choice].predict(student[FEATURES])[0])
        prediction = max(0.0, min(10.0, prediction))
        st.success(f"Estimated current-semester SGPA: **{prediction:.2f} / 10**")
        st.caption(
            "This is an estimate produced by a model trained on synthetic demo data, not an "
            "official result or a reliable assessment of an individual student."
        )

with tab_eval:
    st.subheader("Sample dataset overview")
    st.write(f"Rows: **{len(data)}** · Input features: **{len(FEATURES)}** · Target: **{TARGET}**")
    st.dataframe(data.head(10), use_container_width=True)
    st.caption("The CSV file is saved at `data/college_student_sample.csv`.")
    st.subheader("Model evaluation on a held-out test split")
    st.dataframe(metrics.round(3), use_container_width=True)
    st.caption(
        "MAE and RMSE are SGPA-scale error measures (lower is better). R² measures how well "
        "predictions explain variation in this test split. Because the data is synthetic, "
        "these metrics are for learning the workflow only—not evidence of real-college performance."
    )
    fig, ax = plt.subplots()
    ax.bar(metrics["Model"], metrics["MAE"])
    ax.set_ylabel("MAE (SGPA points; lower is better)")
    ax.set_title("Model comparison on synthetic sample data")
    st.pyplot(fig)
    plt.close(fig)

with tab_about:
    st.subheader("How the app works")
    st.markdown("""
    1. **Sample data:** A reproducible synthetic dataset is created locally the first time the app runs.
    2. **Preprocessing:** Numeric missing values are imputed and scaled; the participation category is one-hot encoded.
    3. **Training:** Linear Regression and Random Forest Regressor are trained on an 80% training split.
    4. **Evaluation:** Both models are tested on a separate 20% split using MAE, RMSE, and R².
    5. **Prediction:** The chosen trained model estimates SGPA from the values entered in the form.

    **Limitations:** The generated data uses an illustrative formula, not actual student outcomes. For a valid college project, obtain permission to use anonymised records, define the target SGPA consistently with your college rules, and evaluate on real held-out records.
    """)
    st.subheader("Input fields")
    st.write(", ".join(FEATURES))
    st.write("Prediction target: `current_semester_sgpa` (0–10).")
