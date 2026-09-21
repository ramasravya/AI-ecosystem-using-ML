import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from agent_workflow import run_agent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plant Intelligence",
    page_icon="🌱",
    layout="wide",
)


# ============================================================
# PROJECT PATH
# ============================================================

# Project root:
# AI ecosystem using ML/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLANT_DATASET_PATH = PROJECT_ROOT / "plant intelligence"


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🌱 Plant Intelligence")

st.subheader(
    "AI-based analysis of vegetation, forest cover and plant ecosystem indicators."
)

st.divider()


# ============================================================
# CHECK DATASET FOLDER
# ============================================================

if not PLANT_DATASET_PATH.exists():
    st.error("Plant Intelligence dataset folder was not found.")
    st.info(f"Expected folder:\n{PLANT_DATASET_PATH}")
    st.stop()


# ============================================================
# FIND CSV FILES
# ============================================================

csv_files = sorted(PLANT_DATASET_PATH.rglob("*.csv"))

if len(csv_files) == 0:
    st.warning(
        "No CSV files were found inside the Plant Intelligence dataset folder."
    )
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

datasets = {}

for file in csv_files:
    try:
        df = pd.read_csv(file)

        # Remove completely empty rows/columns where possible.
        df = df.dropna(axis=0, how="all")
        df = df.dropna(axis=1, how="all")

        datasets[file.name] = {
            "data": df,
            "path": file,
        }

    except Exception as e:
        st.warning(f"Could not load {file.name}: {e}")


if len(datasets) == 0:
    st.error("None of the CSV files could be loaded.")
    st.stop()


# ============================================================
# SUCCESS MESSAGE
# ============================================================

st.success(
    f"🌿 Plant ecosystem datasets loaded successfully — "
    f"{len(datasets)} CSV files found."
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.header("🌿 Vegetation Dataset Overview")

total_records = sum(
    len(item["data"])
    for item in datasets.values()
)

total_features = sum(
    len(item["data"].columns)
    for item in datasets.values()
)

total_missing = sum(
    int(item["data"].isnull().sum().sum())
    for item in datasets.values()
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Datasets", len(datasets))

with col2:
    st.metric("Total Records", total_records)

with col3:
    st.metric("Total Features", total_features)

with col4:
    st.metric("Missing Values", total_missing)


# ============================================================
# DATASET TABLE
# ============================================================

st.subheader("📊 Available Vegetation Datasets")

dataset_summary = []

for filename, item in datasets.items():
    df = item["data"]
    path = item["path"]

    dataset_summary.append(
        {
            "Dataset": filename,
            "Ecosystem": path.parent.name,
            "Records": len(df),
            "Features": len(df.columns),
            "Missing Values": int(df.isnull().sum().sum()),
        }
    )

summary_df = pd.DataFrame(dataset_summary)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# ECOSYSTEM COMPARISON
# ============================================================

st.header("🌳 Ecosystem Comparison")

ecosystem_counts = {}

for item in datasets.values():
    ecosystem = item["path"].parent.name

    if ecosystem not in ecosystem_counts:
        ecosystem_counts[ecosystem] = 0

    ecosystem_counts[ecosystem] += len(item["data"])

ecosystem_df = pd.DataFrame(
    list(ecosystem_counts.items()),
    columns=["Ecosystem", "Records"],
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌿 Records by Ecosystem")

    if not ecosystem_df.empty:
        st.bar_chart(
            ecosystem_df.set_index("Ecosystem")
        )

with col2:
    st.subheader("📈 Ecosystem Statistics")

    st.dataframe(
        ecosystem_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DATASET EXPLORER
# ============================================================

st.header("🔎 Dataset Explorer")

selected_dataset = st.selectbox(
    "Select a vegetation dataset",
    list(datasets.keys()),
    key="dataset_explorer",
)

selected_df = datasets[selected_dataset]["data"]

st.write(f"### {selected_dataset}")

st.write(
    f"**Rows:** {selected_df.shape[0]}  |  "
    f"**Columns:** {selected_df.shape[1]}"
)

st.dataframe(
    selected_df.head(20),
    use_container_width=True,
)


# ============================================================
# NUMERICAL ANALYSIS
# ============================================================

st.header("📊 Vegetation Data Analysis")

numeric_columns = selected_df.select_dtypes(
    include="number"
).columns.tolist()

if len(numeric_columns) > 0:

    selected_feature = st.selectbox(
        "Select a numerical feature",
        numeric_columns,
        key="numerical_feature",
    )

    feature_data = pd.to_numeric(
        selected_df[selected_feature],
        errors="coerce",
    ).dropna()

    if len(feature_data) > 0:

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Mean",
                round(feature_data.mean(), 3),
            )

        with col2:
            st.metric(
                "Minimum",
                round(feature_data.min(), 3),
            )

        with col3:
            st.metric(
                "Maximum",
                round(feature_data.max(), 3),
            )

        with col4:
            st.metric(
                "Standard Deviation",
                round(feature_data.std(), 3),
            )

        st.subheader(
            f"Distribution of {selected_feature}"
        )

        st.bar_chart(
            feature_data.value_counts().sort_index()
        )

    else:
        st.info("No valid numerical values are available.")

else:
    st.info(
        "This dataset does not contain numerical features."
    )


# ============================================================
# CATEGORICAL ANALYSIS
# ============================================================

st.header("🌿 Vegetation Classification Insights")

categorical_columns = selected_df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

if len(categorical_columns) > 0:

    selected_category = st.selectbox(
        "Select a vegetation category",
        categorical_columns,
        key="categorical_feature",
    )

    category_counts = (
        selected_df[selected_category]
        .astype(str)
        .replace("nan", np.nan)
        .dropna()
        .value_counts()
        .head(15)
    )

    st.subheader(
        f"Distribution of {selected_category}"
    )

    st.bar_chart(category_counts)

    st.dataframe(
        category_counts.rename("Count").reset_index(),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info(
        "No categorical features were detected in this dataset."
    )


# ============================================================
# PLANT INTELLIGENCE STATUS
# ============================================================

st.header("🧠 Plant Intelligence Assessment")

coverage_score = 100

if total_missing > 0 and total_records > 0:

    missing_ratio = total_missing / max(
        total_records * max(total_features, 1),
        1,
    )

    coverage_score = max(
        0,
        min(100, round((1 - missing_ratio) * 100)),
    )


if coverage_score >= 90:

    status = "Healthy Data Coverage"

    status_message = (
        "The vegetation datasets contain relatively complete observations."
    )

elif coverage_score >= 70:

    status = "Moderate Data Coverage"

    status_message = (
        "Some missing information is present and should be considered during analysis."
    )

else:

    status = "Limited Data Coverage"

    status_message = (
        "The available data contains substantial missing information."
    )


st.info(
    f"🌱 Plant Intelligence Status: **{status}**"
)

st.write(status_message)


# ============================================================
# AI APPLICATION AREAS
# ============================================================

st.header("🤖 Potential AI Applications")

applications = [
    "🌿 Vegetation health classification",
    "🌳 Forest-cover analysis",
    "🌱 Plant stress detection",
    "🔥 Deforestation monitoring",
    "♻️ Restoration priority identification",
    "🦋 Biodiversity habitat support",
    "📊 Vegetation pattern detection",
    "🌍 Ecosystem change monitoring",
]

for application in applications:
    st.write(f"• {application}")


# ============================================================
# SUSTAINABILITY INSIGHTS
# ============================================================

st.header("🌍 Sustainability Insights")

st.success(
    """
The Plant Intelligence module can support ecosystem monitoring by
analysing vegetation patterns, ecosystem categories and environmental
indicators.

These insights can support conservation planning, biodiversity
protection, forest monitoring and ecological restoration.
"""
)


# ============================================================
# RESPONSIBLE AI
# ============================================================

st.header("⚖️ Responsible AI Considerations")

st.write(
    """
Plant intelligence should be treated as decision-support information.
AI-based observations should be validated using reliable ecological
data and expert knowledge before environmental decisions are made.
"""
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### ⚖️ Fairness")
    st.write(
        "Avoid conclusions based on incomplete or unrepresentative data."
    )

with col2:
    st.markdown("### 🔍 Transparency")
    st.write(
        "Clearly explain which datasets and indicators support an insight."
    )

with col3:
    st.markdown("### 🌱 Ethics")
    st.write(
        "Use AI to support conservation rather than causing ecological harm."
    )

with col4:
    st.markdown("### 🔐 Privacy")
    st.write(
        "Avoid collecting unnecessary personal or sensitive information."
    )


# ============================================================
# ML-BASED VEGETATION CLASSIFICATION
# ============================================================

st.divider()

st.header("🤖 ML-Based Vegetation Classification")

st.write(
    "A Random Forest machine learning model is used to "
    "classify vegetation categories from ecological data."
)


# ============================================================
# FIND DATASETS CONTAINING CLASS
# ============================================================

ml_datasets = []

for filename, item in datasets.items():

    data = item["data"].copy()

    # Make column matching robust to accidental spaces.
    data.columns = data.columns.astype(str).str.strip()

    if "CLASS" in data.columns:

        ml_datasets.append(
            {
                "filename": filename,
                "data": data,
            }
        )


if len(ml_datasets) == 0:

    st.warning(
        "No vegetation dataset containing a CLASS column was found. "
        "The ML classification section cannot run."
    )

else:

    # ========================================================
    # SELECT ML DATASET
    # ========================================================

    ml_dataset_name = st.selectbox(
        "Select dataset for ML prediction",
        [item["filename"] for item in ml_datasets],
        key="ml_dataset_selector",
    )

    selected_ml_data = next(
        item["data"]
        for item in ml_datasets
        if item["filename"] == ml_dataset_name
    ).copy()

    st.subheader("🌱 ML Dataset")

    st.write(
        f"Training data contains **{len(selected_ml_data)} records**."
    )

    # ========================================================
    # TARGET CLEANING
    # ========================================================

    target = "CLASS"

    # IMPORTANT:
    # Convert the target to one consistent type.
    # This fixes:
    # TypeError: '<' not supported between instances of 'int' and 'str'
    selected_ml_data[target] = (
        selected_ml_data[target]
        .astype("string")
        .str.strip()
    )

    # Remove missing/empty target values.
    selected_ml_data = selected_ml_data[
        selected_ml_data[target].notna()
        & (selected_ml_data[target] != "")
    ].copy()

    # ========================================================
    # SEPARATE FEATURES AND TARGET
    # ========================================================

    X = selected_ml_data.drop(
        columns=[target]
    ).copy()

    y = selected_ml_data[target].astype(str)

    # Keep numerical features only.
    X = X.select_dtypes(
        include=np.number
    ).copy()

    # Prevent duplicate feature names from causing pandas/ML shape problems.
    X = X.loc[:, ~X.columns.duplicated()].copy()

    # Replace infinite values.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove completely empty feature columns.
    X = X.dropna(
        axis=1,
        how="all",
    )

    if X.shape[1] == 0:

        st.error(
            "❌ No numerical features were found in the selected dataset."
        )

    else:

        # Fill missing numerical values with the column median.
        for column in X.columns:

            median_value = X[column].median()

            if pd.isna(median_value):
                median_value = 0

            X[column] = X[column].fillna(
                median_value
            )

        # ====================================================
        # CLASS DISTRIBUTION
        # ====================================================

        st.subheader("🌿 Vegetation Class Distribution")

        class_counts = y.value_counts()

        class_distribution_df = (
            class_counts
            .rename("Number of Records")
            .reset_index()
        )

        class_distribution_df.columns = [
            "Vegetation Class",
            "Number of Records",
        ]

        st.dataframe(
            class_distribution_df,
            use_container_width=True,
            hide_index=True,
        )

        # ====================================================
        # REMOVE CLASSES WITH LESS THAN 2 RECORDS
        # ====================================================

        # Stratified train/test splitting requires at least
        # two records per class.
        valid_classes = class_counts[
            class_counts >= 2
        ].index

        valid_mask = y.isin(valid_classes)

        X = X.loc[valid_mask].reset_index(drop=True)
        y = y.loc[valid_mask].reset_index(drop=True)

        if y.nunique() < 2:

            st.error(
                "❌ At least two vegetation classes with two or more "
                "records each are required for classification."
            )

        else:

            st.write(
                f"📊 **Features used for training:** {X.shape[1]}"
            )

            st.write(
                f"🎯 **Vegetation classes used:** {y.nunique()}"
            )

            # =================================================
            # TRAIN / TEST SPLIT
            # =================================================

            # Ensure the stratification target is a clean 1-D Series.
            y = pd.Series(y).astype(str).reset_index(drop=True)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )

            # =================================================
            # RANDOM FOREST MODEL
            # =================================================

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )

            # =================================================
            # TRAIN MODEL
            # =================================================

            with st.spinner(
                "🤖 Training Random Forest model..."
            ):

                model.fit(
                    X_train,
                    y_train,
                )

            # =================================================
            # PREDICTION
            # =================================================

            y_pred = model.predict(
                X_test
            )

            # =================================================
            # MODEL ACCURACY
            # =================================================

            accuracy = accuracy_score(
                y_test,
                y_pred,
            )

            st.subheader("📊 Model Performance")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.2f}%",
                )

            with col2:
                st.metric(
                    "Training Records",
                    len(X_train),
                )

            with col3:
                st.metric(
                    "Testing Records",
                    len(X_test),
                )

            # =================================================
            # ACTUAL VS PREDICTED
            # =================================================

            st.subheader("🔍 Actual vs Predicted Classes")

            result_df = pd.DataFrame(
                {
                    "Actual Class": y_test.values,
                    "Predicted Class": y_pred,
                }
            )

            st.dataframe(
                result_df.head(20),
                use_container_width=True,
                hide_index=True,
            )

            # =================================================
            # CLASSIFICATION REPORT
            # =================================================

            st.subheader("📋 Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0,
            )

            report_df = (
                pd.DataFrame(report)
                .transpose()
                .round(3)
            )

            st.dataframe(
                report_df,
                use_container_width=True,
            )

            # =================================================
            # CONFUSION MATRIX
            # =================================================

            st.subheader("🔎 Confusion Matrix")

            class_labels = model.classes_

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=class_labels,
            )

            cm_df = pd.DataFrame(
                cm,
                index=class_labels,
                columns=class_labels,
            )

            cm_df.index.name = "Actual"
            cm_df.columns.name = "Predicted"

            st.dataframe(
                cm_df,
                use_container_width=True,
            )

            # =================================================
            # FEATURE IMPORTANCE
            # =================================================

            st.subheader(
                "🌿 Important Vegetation Features"
            )

            importance_df = pd.DataFrame(
                {
                    "Feature": X.columns,
                    "Importance": model.feature_importances_,
                }
            )

            importance_df = (
                importance_df
                .sort_values(
                    by="Importance",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            st.dataframe(
                importance_df.head(15),
                use_container_width=True,
                hide_index=True,
            )

            st.bar_chart(
                importance_df
                .head(15)
                .set_index("Feature")
            )

            # =================================================
            # INTERACTIVE PREDICTION
            # =================================================

            st.divider()

            st.subheader(
                "🔮 Predict Vegetation Class"
            )

            st.write(
                "Select an ecological record and let the trained "
                "Random Forest model predict its vegetation class."
            )

            selected_record_number = st.number_input(
                "Select record number",
                min_value=0,
                max_value=len(X) - 1,
                value=0,
                step=1,
                key="prediction_record",
            )

            # Select exactly one row as a 2-D DataFrame.
            # This avoids pandas "wrong number of dimensions" errors.
            selected_record_number = int(selected_record_number)

            selected_record = X.iloc[[selected_record_number]].copy()

            # Keep exactly the same feature order used during training.
            selected_record = selected_record.reindex(
                columns=X_train.columns
            )

            st.write("Selected ecological record:")

            st.dataframe(
                selected_record,
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                "🌱 Predict Vegetation Class",
                type="primary",
            ):

                prediction = model.predict(
                    selected_record
                )[0]

                st.success(
                    f"🌿 Predicted Vegetation Class: **{prediction}**"
                )

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        selected_record
                    )[0]

                    probability_df = pd.DataFrame(
                        {
                            "Vegetation Class": model.classes_.astype(str),
                            "Probability": probabilities * 100,
                        }
                    )

                    probability_df["Probability"] = (
                        probability_df["Probability"].round(2)
                    )

                    probability_df = (
                        probability_df
                        .sort_values(
                            by="Probability",
                            ascending=False,
                        )
                        .reset_index(drop=True)
                    )

                    st.subheader(
                        "🎯 Prediction Confidence"
                    )

                    st.dataframe(
                        probability_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader("📊 Confidence Chart")

                    st.bar_chart(
                        probability_df.set_index(
                            "Vegetation Class"
                        )["Probability"]
                    )

                    top_class = str(
                        probability_df.iloc[0]["Vegetation Class"]
                    )
                    top_probability = float(
                        probability_df.iloc[0]["Probability"]
                    )

                    st.info(
                        f"🎯 Highest model confidence: "
                        f"**{top_class} ({top_probability:.2f}%)**"
                    )


 # =================================================
            # AI ECOSYSTEM ASSISTANT
            # =================================================

            st.divider()

            st.subheader("🤖 AI Ecosystem Assistant")

            st.write(
                "Ask a question about the selected ecological record, "
                "vegetation prediction, or ecosystem sustainability."
            )

            agent_question = st.text_area(
                "Ask the AI:",
                placeholder=(
                    "Example: What is the predicted vegetation class "
                    "and why is vegetation conservation important?"
                ),
                key="agent_question"
            )

            if st.button(
                "🔍 Ask AI Ecosystem Assistant",
                type="primary",
                key="ask_agent"
            ):

                if agent_question.strip():

                    with st.spinner(
                        "🤖 AI Ecosystem Agent is analyzing..."
                    ):

                        agent_result = run_agent(
                            agent_question,
                            model,
                            selected_record
                        )

                    st.write("### 🧠 Agent Workflow")

                    tools_used = agent_result["tools_used"]

                    st.info(
                        "Tools used: "
                        + ", ".join(tools_used)
                    )

                    if agent_result["prediction"] is not None:

                        prediction_result = (
                            agent_result["prediction"]
                        )

                        st.write(
                            "### 🌱 Random Forest Prediction"
                        )

                        st.write(
                            f"**Predicted Vegetation Class:** "
                            f"{prediction_result['predicted_class']}"
                        )

                        if prediction_result["confidence"] is not None:

                            st.write(
                                f"**Model Confidence:** "
                                f"{prediction_result['confidence']:.2f}%"
                            )

                    if agent_result["knowledge"]:

                        st.write(
                            "### 📚 Retrieved Ecosystem Knowledge"
                        )

                        for number, context in enumerate(
                            agent_result["knowledge"],
                            start=1
                        ):

                            with st.expander(
                                f"Knowledge Result {number}"
                            ):

                                st.write(context)

                else:

                    st.warning(
                        "Please enter a question before asking the AI."
                    )


# ============================================================
# FINAL PROJECT CONNECTION
# ============================================================

st.divider()

st.caption(
    "AI Ecosystem Intelligence System • "
    "Plant Intelligence Module • AI for Sustainability"
)
