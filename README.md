# DeepAudit
# 🔐 DeepAudit: Smart Contract Vulnerability Detection using Deep Learning

A Flask-based web application that uses advanced deep learning algorithms (CNN, EfficientNet B2, and Xception) to detect vulnerabilities in Ethereum smart contracts by converting compiled bytecode into images and classifying them.

---

## 📄 Project Overview

This project explores the use of machine learning — specifically **CNN**, **EfficientNet B2**, and **Xception** architectures — for detecting vulnerabilities in smart contracts. It converts smart contract bytecode into RGB images, classifies them using deep learning, and presents results via a user-friendly Flask web interface.

---

## 📌 Key Features

- 🧠 **Deep Learning Models**: Trained CNN, Xception, and EfficientNet B2 on labeled contract vulnerability data.
- 🔍 **Vulnerability Detection**: Supports 7 common vulnerability types including:
  - Reentrancy
  - Timestamp Dependency
  - Integer Overflow
  - Dangerous Delegatecall
  - Unchecked External Call
  - Ether Strict Equality
  - Block Number Dependency
- 🌐 **Web Application**: Built using Flask to allow upload of Solidity `.sol` files for instant security assessment.
- 📊 **Evaluation Metrics**: Accuracy, Precision, Recall, and F1-Score reported for each algorithm.

---

## 🧪 Models Compared

| Model           | Accuracy |
|----------------|----------|
| CNN            | 67%      |
| Xception       | 76%      |
| **EfficientNet B2** | **79%** ✅  |

EfficientNet B2 demonstrated the best overall accuracy and is deployed in the web app.

## 📂 Dataset Download

You can download the full labeled dataset used for training and evaluation from the link below:

👉 [Click here to download the dataset from Google Drive]
([https://drive.google.com/file/d/1AbCDEfgHIjKLmnOPQ/view?usp=sharing](https://drive.google.com/drive/folders/1tHpT4y6gZeY8XKovyYDqbAbzr1jheGOR?usp=sharing))
---



