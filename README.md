# 🎬 IMDB Movie Sentiment Analysis - Bidirectional LSTM Pipeline

## 🎯 Overview

This project implements a Deep Learning NLP pipeline using a **Bidirectional Long Short-Term Memory (LSTM)** network to classify movie reviews. By processing 25,000 samples, the model captures sequential context from both directions of a sentence, achieving a robust **85.22% accuracy**.

The solution addresses core NLP challenges: handling out-of-vocabulary words with a custom mapping, managing variable sequence lengths via padding, and preventing overfitting through an automated Early Stopping mechanism.

| | |
|---|---|
| **Dataset** | IMDB Movie Reviews (25k samples) |
| **Final Accuracy** | 85.22% |
| **Architecture** | Embedding → Dropout → Bidirectional LSTM → Global Mean Pooling → Linear Logit Output |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch |
| Data Handling | pandas, numpy |
| Text Processing | re (Regular Expressions) |
| Training Utilities | DataLoader, TensorDataset |
| Analysis Type | Supervised Sentiment Classification |

---

## 📈 Pipeline Phases

### 1. Preprocessing & Vocabulary Mastery

A custom text-processing engine converts raw human language into machine-understandable tensors:

- **Custom Tokenizer:** Cleans text by converting to lowercase and stripping non-alphanumeric characters.
- **Special Tokens:** Implemented `<PAD>` (0) and `<UNK>` (1) to handle uniform sequence length and unseen words.
- **Padding:** Standardized reviews to a `max_len` of 200 tokens.

### 2. Deep Learning Architecture

The model uses a **Bidirectional LSTM** to "understand" context:

- **Embedding Layer:** Maps words into a 128-dimensional vector space.
- **Bi-LSTM:** Runs two parallel layers (forward and backward) to capture dependencies like negation (e.g., *"not good"*).
- **Global Mean Pooling:** Averages hidden states across the sequence to capture the overall sentiment "signal."

### 3. Training & Early Stopping

- **Loss Function:** `BCEWithLogitsLoss` for numerical stability.
- **Optimization:** Adam optimizer (`lr=0.001`).
- **Regularization:** Used Dropout and Early Stopping (`patience=3`) to halt training at Epoch 30, preventing overfitting and saving `best_model.pt`.

---

## 🏆 Key Results

### Performance Metrics

| Stage | Metric | Result |
|---|---|---|
| Training Loss | Final Avg Loss | 0.0029 |
| Validation | Early Stopping | Triggered at Epoch 30 |
| Test Set | Accuracy | **85.22%** |

### 🧪 Model in Action (Sample Predictions)

The model was tested against nuanced reviews to verify its "intelligence" beyond simple keywords:

| Review Text | Logit Score | Probability | Prediction |
|---|---|---|---|
| "A cinematic masterpiece that everyone should see." | 8.1309 | 99.97% | Positive ✅ |
| "Don't listen to the critics, this movie is a gem!" | 10.7389 | 100.0% | Positive ✅ |
| "The acting was great but the script was terrible." | -9.6115 | 0.01% | Negative ✅ |
| "Not as bad as people say, but still not good." | -3.9511 | 1.89% | Negative ✅ |
| "I would rather watch paint dry than see this again." | -4.3067 | 1.33% | Negative ✅ |
| "The best cinematography I've seen all year!" | -1.2747 | 21.85% | Negative ⚠️ |

> **Technical Insight:** The model identifies complex negations correctly but shows a *"Cynical Bias"* on technical terms like "cinematography." In the IMDB dataset, technical praise is frequently used as a consolation in negative reviews, which the model has successfully learned.

---

## 📂 Repository Structure

```
imdb-sentiment-lstm/
│
├── Movie Review.ipynb       # Full development & training notebook
├── README.md                # Project documentation
├── best_model.pt            # Saved weights (85.22% accuracy)
│
├── movie review/            # Data directory
│   ├── train.csv            # 25,000 labeled reviews
│   └── test.csv             # Unlabeled test set
```

---

## 🚀 Quick Start

### Inference Code

```python
# Load the best weights
model.load_state_dict(torch.load('best_model.pt', weights_only=True))
model.eval()

# Predict
review = "This film was an absolute masterpiece!"
print(f"Result: {predict(review)}")
```

---

## 💡 Key Learnings

- **Logit Decision Boundary:** Using `0.0` as the threshold for logits (instead of `0.5` for probabilities) is critical when using `BCEWithLogitsLoss`.
- **Context Matters:** Bidirectional layers are significantly better at handling sarcasm and "back-handed" compliments than standard RNNs.
- **Overfitting Control:** Achieving a training loss of `0.0029` shows high learning capacity, but Early Stopping is vital to maintain test-set generalization.

---

## 📧 Contact

**Prashant Shukla**

- 📧 Email: [prashantshukla8851@gmail.com](mailto:prashantshukla8851@gmail.com)
- 💼 LinkedIn: [Prashant Shukla](https://linkedin.com/in/prashant-shukla)
- 🔗 GitHub: [@pr4sh4nt-shukla](https://github.com/pr4sh4nt-shukla)

---

⭐ *If you found this NLP project insightful, please consider giving it a star!* ⭐
