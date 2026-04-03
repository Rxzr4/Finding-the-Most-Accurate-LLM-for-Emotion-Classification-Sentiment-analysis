# Finding-the-Most-Accurate-LLM-for-Emotion-Classification-Sentiment-analysis
**Author:** Umar Raza — University of Hertfordshire, BSc Computer Science

## Overview
This project compares three LLMs (LLaMA 3.1 8B, GLM-4.5 Air, Trinity Large) 
on 28-class emotion detection using the GoEmotions dataset, then launches a 
chatbot with the most accurate model.

## Results
| Model | Accuracy |
|---|---|
| Trinity Large | 34.34% |
| LLaMA 3.1 8B | 29.17% |
| GLM-4.5 Air | 0% (API errors) |

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API keys to `fyp_sentiment_analysis.py`
4. Place GoEmotions TSV files in a `data/` folder
5. Run: `python fyp_sentiment_analysis.py`

## Dataset
GoEmotions by Google Research (Demszky et al., 2020) — 58,000 Reddit comments 
annotated with 28 emotion categories.

## Project Structure
- `fyp_sentiment_analysis.py` — main script
- `comparison_results.json` — evaluation results
- `data/` — GoEmotions dataset files
- `emotions.txt` — list of 28 emotion labels
