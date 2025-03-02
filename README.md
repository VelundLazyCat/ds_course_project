# **Customer Churn Prediction Project**

## 📌 **Project Overview**
This project aims to predict **customer churn** for a telecommunications company using **machine learning** and **neural network models**.  
The application provides a **web-based interface** using **Streamlit**, allowing users to upload customer data and receive churn predictions.

---

## 🚀 **Features**
- ✅ **Multiple classification models:**
  - **Random Forest Classifier**
  - **Logistic Regression Classifier**
  - **Stochastic Gradient Descent (SGD) Classifier**
  - **Ensemble ML Classifier (Stacking)**
  - **Neural Network Classifier**
- 🔄 **Data preprocessing and normalization**
- 📊 **Interactive web application for predictions**
- 🐳 **Dockerized environment for easy deployment**

---

## 📁 **Project Structure**
. ├── app.py # Main entry point (Streamlit application) ├── functions.py # Helper functions for data processing and predictions ├── models_ml.py # Machine learning model training and evaluation ├── internet_service_churn.csv # Dataset (input data) ├── accuracy_ml.pkl # Accuracy scores for ML models ├── accuracy_nn.pkl # Accuracy scores for neural network model ├── model_log.pkl # Logistic Regression model ├── model_NN1.keras # Trained neural network model ├── model_sgd.pkl # SGD Classifier model ├── model_tree.pkl # Random Forest model ├── stacking_model_ml.pkl # Ensemble Stacking model ├── values.pkl # Preprocessing constants for standardization ├── Dockerfile # Docker container setup ├── docker-compose.yml # Docker Compose configuration ├── README.md # Project documentation 
## ⚙️ **Installation and Usage**

### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/your-repo-url
cd customer-churn-prediction
### 2️⃣ **Running with Docker**
To build and run the application using Docker, execute the following command:
docker-compose up --build
This will start a Streamlit application available at:
🔗 http://localhost:8501 
### 3️⃣ **Running with Docker**
If you prefer to run the application without Docker:
pip install poetry
poetry install
poetry run streamlit run app.py
## 🎯 **Application Usage**
1. 📂 **Upload** a CSV file containing customer data.
2. 🏷️ **Choose a classification model** from the sidebar.
3. 🔎 **Click on "Predict"** to get the churn prediction results.
4. 📥 **Download the results** as a CSV file.

---

## 🧠 **Machine Learning Models**
This project includes various **machine learning models**, trained using **Scikit-learn** and **TensorFlow**:

- 📌 **Random Forest Classifier**: A robust ensemble model using decision trees.
- 📌 **Logistic Regression**: A simple yet effective linear model.
- 📌 **SGD Classifier**: A fast and efficient classifier for large-scale data.
- 📌 **Stacking Model**: Combines multiple models to improve performance.
- 📌 **Neural Network**: A deep learning approach using **TensorFlow**.

---

## 🛠️ **Technologies Used**
- 🐍 **Python 3.10**
- 🌍 **Streamlit**
- 🔬 **Scikit-learn**
- 🧠 **TensorFlow**
- 🗂 **Pandas**
- 🐳 **Docker**
- 📦 **Poetry (Dependency Management)**

---

## 📦 **Deployment**
This project is containerized using **Docker** and can be deployed easily using **Docker Compose**.  
The **`Dockerfile`** includes all necessary dependencies, and the **`docker-compose.yml`** file simplifies the deployment process.

---

## 👨‍💻 **Contributors**
- ✨ **Your Name**
- ✨ **Other Team Members**

---

## 📜 **License**
This project is licensed under the **MIT License**.

---

## ℹ️ **Additional Notes**
- ✅ **Ensure that all `.pkl` and `.keras` files are included** in the project directory before running the application.
- ✅ If running without Docker, make sure **all dependencies** are installed using **Poetry**.
- ✅ For **best performance**, the dataset should be **preprocessed** before making predictions.

