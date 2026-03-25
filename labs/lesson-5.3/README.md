# Lab 5.3: API, Serving & Putting It All Together

## Objective

Wrap your trained model in a FastAPI application, add input validation, write tests, and
document the complete capstone project.

## Prerequisites

- Completed Lab 5.1 with saved preprocessing objects in `data/processed/`
- Completed Lab 5.2 with a saved model checkpoint in `models/final_model.pt`

## Getting Started

1. Install dependencies: `pip install fastapi uvicorn pydantic requests`
2. Open `starter.py` which contains the FastAPI app scaffolding.
3. Customize the `PredictionRequest` schema to match your dataset's features.
4. Run the server: `uvicorn starter:app --host 0.0.0.0 --port 8000 --reload`
5. Run the test script: `python test_api.py`

## Files in This Lab

| File | Purpose |
|------|---------|
| `starter.py` | FastAPI application with model loading and prediction endpoint |
| `test_api.py` | Test script that validates the API responses |
| `README.md` | This file |

## What to Deliver

By the end of this lab you should have:

- A working FastAPI app that loads the model at startup
- `GET /health` endpoint returning status
- `POST /predict` endpoint accepting JSON and returning predictions
- Input validation that rejects missing or out-of-range fields (422 response)
- Deterministic predictions (same input always gives same output)
- All tests in `test_api.py` passing
- A project README.md with setup instructions, results table, and project structure
- A `requirements.txt` file
- A completed self-assessment rubric

## Self-Assessment Checklist

Score yourself honestly on each item (Yes / Partial / No):

### Data Pipeline (Phase 1)
- [ ] Data loaded and inspection questions answered
- [ ] At least 5 EDA plots saved
- [ ] Missing values handled with reasoning
- [ ] At least 2 engineered features
- [ ] Train/val/test split saved with preprocessors

### Modeling (Phase 2)
- [ ] Baseline models trained and evaluated
- [ ] Neural network trained with early stopping
- [ ] Learning curves plotted and interpreted
- [ ] At least 3 experiments with logged results
- [ ] Test set evaluated exactly once

### API & Deployment (Phase 3)
- [ ] POST /predict returns correct predictions
- [ ] Input validation rejects bad inputs
- [ ] Test script passes all cases
- [ ] Predictions are deterministic

### Documentation
- [ ] README with setup, results, and structure
- [ ] requirements.txt present
- [ ] Code is modular (model.py, app.py separate)
- [ ] Key decisions documented

### Scoring Guide
- 90-100: Excellent — portfolio ready
- 75-89: Good — minor polish needed
- 60-74: Adequate — several areas need improvement
- Below 60: Incomplete — revisit the phases with gaps
