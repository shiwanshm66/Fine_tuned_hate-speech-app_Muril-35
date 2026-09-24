# 🛡️ Fine-Tuned MuRIL Hate Speech Detection

A multilingual hate speech detection system built by fine-tuning Google's MuRIL Transformer model. The model classifies text as Hate or Non-Hate across English, Hindi, and Hinglish, providing confidence scores and supporting real-time predictions through a Streamlit interface.

---

## 📌 Features

- Fine-tuned MuRIL Transformer
- Supports English, Hindi, and Hinglish
- Real-time hate speech prediction
- Confidence score visualization
- Adjustable prediction threshold
- Batch CSV prediction
- Interactive Streamlit Web Application
- High multilingual performance

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- MuRIL
- Streamlit
- Pandas
- NumPy
- Scikit-learn

---

## Dataset

The model was trained on a multilingual hate speech dataset containing Hindi, English, and Hinglish social media text.

---

## Model Architecture

```
Input Text
      │
      ▼
MuRIL Tokenizer
      │
      ▼
Fine-Tuned MuRIL Transformer
      │
      ▼
Classification Head
      │
      ▼
Hate / Non-Hate
```

---

## Project Structure

```
Fine_tuned_hate-speech-app_Muril/
│
├── data/
├── deployment/
├── preprocessing/
├── training/
├── evaluation/
├── inference/
├── features/
├── utils/
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ArpitSingh18-hub/Fine_tuned_hate-speech-app_Muril.git

cd Fine_tuned_hate-speech-app_Muril

pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

---

## Example

**Input**

```
यह एक उदाहरण टेक्स्ट है।
```

**Prediction**

```
Label: Hate Speech

Confidence: 96%
```

---

## Model Performance

| Metric | Score |
|---------|-------|
| Precision | 0.95 |
| Recall | 0.94 |
| F1-Score | 0.94 |

---

## Applications

- Social Media Moderation
- Online Community Management
- Content Filtering
- Cyberbullying Detection
- Public Safety Monitoring

---

## Future Improvements

- Support additional Indian languages
- Multi-label classification
- Explainable AI (XAI)
- Faster inference
- REST API deployment

---

## GitHub Repository
https://github.com/shiwanshm66/Fine_tuned_hate-speech-app_Muril-35


---

## Author

**Arpit Singh**

B.Tech CSE (AI & ML)
