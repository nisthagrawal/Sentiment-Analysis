Warning: fs was declared with const; use let for reassignable variables.
# AI Review Intelligence

An early sentiment analysis project built around the IMDB movie review dataset. It includes a TensorFlow/Keras binary classifier, a FastAPI prediction endpoint, and a Streamlit dashboard.

## Current scope

- Predicts **Positive** or **Negative** sentiment from review text.
- Uses the trained model and vocabulary in `code/model_files/`.
- Provides an API at `/predict` and interactive API documentation at `/docs`.
- Displays a local session's predictions and simple emotion keyword matches in the dashboard.

The emotion display is a small keyword heuristic, not an emotion model. The summary is a short extract of the input, not generated text. The dashboard currently expects the API at `http://127.0.0.1:8000`.

## Run locally

Use Python 3.10 and create a virtual environment from the project root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the API from the `code` directory:

```bash
cd code
uvicorn main:app --reload
```

In a second terminal, activate the same environment and start the dashboard:

```bash
cd code
streamlit run app.py
```

The API docs are at <http://127.0.0.1:8000/docs>; the dashboard opens in the browser. Keep the model files and `word_index.json` in `code/model_files/`.

## Project layout

```text
code/                 API, Streamlit app, prediction code, and model assets
outputs_images/       UI captures and system diagrams
AI_Review_Intelligence_System_Report.pdf
```

`code/Train.py` currently loads a saved model and vocabulary and predicts text; it does not train a model. The original IMDB training dataset is not included in this repository.

## Next research steps

Treat this as a study project and build up its evidence and evaluation before making claims about real-world mood detection. A useful path is to add documented datasets with licensing and source information, compare multi-class emotion or sentiment labels, report per-class metrics and data limitations, and then build a dashboard around reproducible dataset and model evaluation. Any later system that adapts responses to a person's inferred mood should make the inference uncertain and user-controllable.
