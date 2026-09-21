import pandas as pd

from rag_engine import retrieve_context


# ============================================================
# TOOL 1 — RANDOM FOREST PREDICTION
# ============================================================

def prediction_tool(model, selected_record):

    prediction = model.predict(selected_record)[0]

    result = {
        "predicted_class": str(prediction),
        "confidence": None,
        "probabilities": None
    }

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            selected_record
        )[0]

        probability_df = pd.DataFrame(
            {
                "Vegetation Class": model.classes_.astype(str),
                "Probability": probabilities * 100
            }
        )

        probability_df["Probability"] = (
            probability_df["Probability"].round(2)
        )

        probability_df = (
            probability_df
            .sort_values(
                by="Probability",
                ascending=False
            )
            .reset_index(drop=True)
        )

        result["confidence"] = float(
            probability_df.iloc[0]["Probability"]
        )

        result["probabilities"] = probability_df

    return result


# ============================================================
# TOOL 2 — RAG KNOWLEDGE RETRIEVAL
# ============================================================

def knowledge_tool(question):

    context = retrieve_context(
        question,
        k=3
    )

    return context


# ============================================================
# AGENT ROUTER
# ============================================================

def decide_tools(user_query):

    query = user_query.lower()

    prediction_words = [
        "predict",
        "prediction",
        "class",
        "vegetation class",
        "record",
        "confidence",
        "model"
    ]

    knowledge_words = [
        "why",
        "explain",
        "ecosystem",
        "forest",
        "biodiversity",
        "conservation",
        "climate",
        "sustainability",
        "sdg",
        "vegetation",
        "environment"
    ]

    use_prediction = any(
        word in query
        for word in prediction_words
    )

    use_knowledge = any(
        word in query
        for word in knowledge_words
    )

    if use_knowledge and not use_prediction:
        return ["knowledge"]

    if use_prediction and not use_knowledge:
        return ["prediction"]

    if use_prediction and use_knowledge:
        return ["prediction", "knowledge"]

    return ["knowledge"]


# ============================================================
# AGENTIC WORKFLOW
# ============================================================

def run_agent(
    user_query,
    model,
    selected_record
):

    tools_to_use = decide_tools(user_query)

    result = {
        "tools_used": tools_to_use,
        "prediction": None,
        "knowledge": []
    }

    if "prediction" in tools_to_use:

        result["prediction"] = prediction_tool(
            model,
            selected_record
        )

    if "knowledge" in tools_to_use:

        result["knowledge"] = knowledge_tool(
            user_query
        )

    return result


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("🌱 AI ECOSYSTEM AGENT WORKFLOW")
    print("=" * 60)

    print("\nAvailable tools:")
    print("1. Random Forest Prediction")
    print("2. ChromaDB Knowledge Retrieval")

    print("\nAgent workflow ready.")