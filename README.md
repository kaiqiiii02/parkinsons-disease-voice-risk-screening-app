# Parkinson's Disease Voice Risk Screening App

A web-based screening tool that analyzes voice recordings to detect patterns associated with Parkinson's disease. Built with Streamlit and powered by an XGBoost classifier trained on the Parkinson's voice dataset from Kaggle.

## Live Demo

[Try the app on Streamlit Community Cloud](#) <!-- Update this link after deployment -->

## Features

- Browser-based voice recording (3 samples of sustained "aaaaah")
- Real-time acoustic feature extraction using Praat (parselmouth)
- 22 MDVP features analyzed by XGBoost model
- Personalized layman explanations of voice patterns
- Risk gauge visualization and dashboard with screening history
- 7-page guided journey designed for non-technical users

## Tech Stack

- **Frontend & Backend**: Streamlit
- **Voice Analysis**: Praat (parselmouth), librosa
- **Machine Learning**: XGBoost, scikit-learn
- **Model Pipeline**: Winsorize → Log1p → StandardScaler → XGBoost

## Model Performance (on test set)

| Metric | Score |
|---|---|
| Accuracy | 0.744 |
| Precision | 0.750 |
| Recall | 0.968 |
| F1-score | 0.845 |

## Local Setup

```bash
git clone https://github.com/kaiqiiii02/parkinsons-disease-voice-risk-screening-app.git
cd parkinsons-disease-voice-risk-screening-app
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer

This app is an academic prototype developed for **WQD7001 Principles of Data Science** at Universiti Malaya. It is **not a medical diagnosis tool**. Results should be interpreted as a screening indicator only. Please consult a qualified healthcare professional for clinical assessment of any Parkinson's-related concerns.

## Authors

**Group 20** — WQD7001 Principles of Data Science, Universiti Malaya

## Dataset

Parkinson's Disease Classification Dataset, accessed via Kaggle.

## License

This project is released for academic and educational purposes.
