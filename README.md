# AI Ecosystem Using ML

The project demonstrates how AI can support environmental data analysis and ecosystem monitoring. It can help users quickly explore vegetation patterns, obtain machine-learning-based vegetation predictions, and retrieve relevant knowledge about conservation and sustainability. By combining predictive analytics with a curated ecological knowledge base, the system can support environmental education, research, and preliminary decision-making. 

The project also demonstrates a practical application of AI for sustainability and can be extended in the future with additional environmental datasets, real-time data sources, and more advanced AI agents.

---

## Key Features
* **Vegetation & Ecosystem Analytics:** Exploratory analysis of ecological datasets (vegetation mapping, air quality, forest fires, biodiversity).
* **Predictive ML Modeling:** Machine learning pipeline for environmental pattern classification and prediction.
* **Knowledge Retrieval & Agent Workflows:** Integrates an ecological knowledge base (RAG engine via ChromaDB) and agentic workflows to retrieve context-aware conservation insights.
* **Interactive Exploration:** Streamlit dashboard interface for inspecting data, correlation heatmaps, and running real-time predictions.

---

## Project Structure
```text
├── src/                      # Source modules (APIs, RAG engine, workflows)
├── vegetation mapping dataset/ # Processed ecological data
├── dashboard.py              # User interface / dashboard
├── train_ecosystem_ai.py     # Model training pipeline
├── prediction.py             # Model inference logic
├── ecosystem_knowledge.txt   # Curated conservation knowledge base
└── .gitignore                # Ignored virtual environments, archives & binaries
