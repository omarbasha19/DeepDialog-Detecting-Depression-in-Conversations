# DeepDialog: Detecting Depression in Conversations Using Deep Learning

A full pipeline to classify human conversational text into depressive and non-depressive classes using deep learning and natural language processing (NLP). This project explores the possibility of leveraging dialogue content to support mental health screening and early intervention through AI.

---

## 🧠 Project Overview

| Item                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
|  Objective              | Detect signs of depression from raw conversational text                     |
|  Input Format           | Text-based dialogues (single or multi-turn)                                 |
|  Task                  | Binary text classification: `Depression (1)` or `Non-Depression (0)`        |
|  Output                | Trained deep learning model (`.h5`), labeled dataset (`.csv`)               |
|  Tools Used            | Python, TensorFlow/Keras, Pandas, NLTK, Scikit-learn                        |
| 📁 Dataset               | Manually labeled text, structured by folder into two classes                |

---

## 📁 Dataset Description

The dataset was collected from local directories using the following structure:

```
📁 Dataset/
├── Depression/         # Labeled as 1
└── Non-Depression/     # Labeled as 0
```

### 📌 Dataset Details (after merging)

| Column               | Description                                                 |
|----------------------|-------------------------------------------------------------|
| `text`               | Original text dialogue                                      |
| `label`              | Binary class: 1 = Depression, 0 = Non-Depression            |
| `preprocessed_text`  | Cleaned, tokenized, and normalized version of the dialogue |

- Total Samples: ~1000+
- Class Distribution: Approx. 50/50
- Data Format: CSV (`augmented_dataset.csv`)
- Language: English
- Ethics: All texts are anonymized; no personal identifiers included

---

##  Preprocessing Pipeline

Text preprocessing was applied to all dialogue lines to reduce noise and improve model generalization.

| Step                  | Technique                                                    |
|-----------------------|--------------------------------------------------------------|
| Lowercasing           | Convert all text to lowercase                                |
| Punctuation Removal   | Remove special characters and non-textual symbols            |
| Stopword Removal      | Remove filler words using NLTK stopword list                 |
| Tokenization          | Split text into tokens (words)                               |
| Lemmatization         | Reduce words to root form (e.g., "running" → "run")          |

All cleaned texts are stored in the `preprocessed_text` column.

---

##  Model Architecture

Built using **TensorFlow / Keras**, the model architecture supports both CNN and LSTM variants.

### 📌 Sample CNN-Based Architecture

```
Input → Embedding → Conv1D → GlobalMaxPool → Dense → Dropout → Output
```

| Layer                  | Purpose                                              |
|------------------------|------------------------------------------------------|
| `Embedding`            | Converts tokens to word vectors                      |
| `Conv1D`               | Learns local patterns in the text                    |
| `GlobalMaxPooling1D`   | Reduces temporal output to flat feature map          |
| `Dense`                | Fully connected for feature transformation           |
| `Dropout`              | Prevents overfitting                                 |
| `Sigmoid` Output       | Produces binary classification score (0-1)           |

###  Compilation Details

| Parameter      | Value                     |
|----------------|---------------------------|
| Loss Function  | Binary Crossentropy       |
| Optimizer      | Adam                      |
| Metrics        | Accuracy, Precision, Recall, F1 Score |

---

## 📊 Model Evaluation Metrics

| Metric            | Value   | Description                                                  |
|-------------------|---------|--------------------------------------------------------------|
| **Accuracy**       | 0.91    | Overall proportion of correct predictions                   |
| **Precision**      | 0.89    | Ratio of true positives over all predicted positives        |
| **Recall**         | 0.92    | Ratio of true positives over all actual positives           |
| **F1 Score**       | 0.90    | Harmonic mean of precision and recall                       |
| **Confusion Matrix** | TP: 235<br>FP: 30<br>FN: 20<br>TN: 215 | Shows counts of true/false positives/negatives             |

Final model is saved as:

```
📁 best_model.h5
```

---

##  Inference Example

To use the trained model for prediction:

```python
from tensorflow.keras.models import load_model
model = load_model("best_model.h5")

# After applying the same preprocessing steps:
pred = model.predict(processed_input)
```

---

## 📂 Repository Structure

| File                     | Description                                              |
|--------------------------|----------------------------------------------------------|
| `Code.ipynb`             | Main notebook for loading, preprocessing, training       |
| `augmented_dataset.csv`  | Cleaned and labeled dialogue dataset                     |
| `best_model.h5`          | Saved Keras model after training                         |
| `README.md`              | Full project documentation (this file)                   |

---

## 🚀 Applications

| Domain                   | Use Case                                                 |
|---------------------------|----------------------------------------------------------|
| Mental Health             | Early detection and triage of depressive behavior         |
| AI Chatbots               | Enhance conversational agents with emotional awareness    |
| Educational Psychology    | NLP-based analysis of student or user emotional state     |
| Support Systems           | Automatic flagging of at-risk conversations               |

---

## ⚖️ Ethical Use & Disclaimer

-  **Privacy**: Dataset is anonymized and does not include personal information.  
-  **Not a diagnostic tool**: This model is intended for **educational and research use only**.  
-  **Responsibility**: Any deployment in sensitive applications must involve proper validation, oversight, and clinical backing.

---



##  Conclusion

This project demonstrates the potential of deep learning models in detecting depression from conversational text, highlighting the intersection between **natural language processing** and **mental health analysis**. By leveraging a clean, labeled dataset and a structured modeling pipeline, we achieved strong performance across key classification metrics such as accuracy, precision, and recall.

The model not only provides accurate predictions but also opens up opportunities for integration into real-time conversational platforms, chatbots, and mental health screening tools — provided that **ethical safeguards** and **clinical validation** are maintained.

This work represents a step toward building AI systems that are **human-centered**, **ethically aware**, and **socially beneficial** — especially in domains as sensitive and impactful as mental health.

> While AI cannot replace professional diagnosis, it can assist in guiding attention to where it’s needed most.

---

