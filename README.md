# Smartphone Sentiment Analyzer with Chatbot

## 📌 Description
A Python-based project that scrapes smartphone reviews, performs sentiment analysis, and integrates a chatbot to help users explore phones based on features and review sentiment.

## 🛠️ Technologies Used
- Python (Jupyter Notebook, Scripts)
- Selenium (for web scraping)
- Pandas, NumPy (data processing)
- Keras / TensorFlow (chatbot model)
- Flask (web interface)
- CSV (for storing phone and review data)

## 🚀 Features
- Scrapes smartphone reviews using Selenium
- Stores and processes review data into structured CSVs
- Chatbot interface trained with a deep learning model
- Interactively guides users in choosing phones based on sentiment and specs

## 📁 Project Structure
```
22i0469_AIA_Project/
├── 22i0469_AIA_Project.ipynb        # Main notebook (scraping + preprocessing)
├── app.py                           # Flask web app for chatbot
├── chatbot_model.h5                 # Trained Keras chatbot model
├── reviews.csv                      # Raw scraped reviews
├── phone_info.csv                   # Phone specifications
├── new_reviews.csv                  # Cleaned/processed reviews
├── database.csv                     # Possibly merged or final dataset
└── ...
```

## 🔧 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Flask app:
   ```bash
   python app.py
   ```
3. Explore the Jupyter Notebook for scraping or dataset preparation:
   ```bash
   jupyter notebook 22i0469_AIA_Project.ipynb
   ```

## 👤 Author
**Ahmed Khan**  
ID: 22i0469
