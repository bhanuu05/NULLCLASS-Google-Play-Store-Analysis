
# 📊 Play Store Data Analysis

### Internship Project | NullClass Edtech Private Limited

**🚀 Overview**

This project is part of the Nullclass Internship (11/03/2025 - 11/05/2025) and involves analyzing Google Play Store data to derive meaningful insights using Python and data visualization techniques.

**🎯 Project Objective**

This project aims to analyze trends in mobile app performance using Google Play Store data. The insights gained can help developers and businesses optimize their apps based on user preferences, installs, revenue, and ratings.

**📂 Data Sources**

📄 Play Store Data.csv: Contains details about various apps including category, installs, rating, and revenue.

📝 User Reviews.csv: Includes user feedback for different apps.

## 📈 Tasks & Visualizations

### 1️⃣ Scatter Plot (Revenue vs. Installs for Paid Apps)

📌 Visualizes the relationship between revenue and installs.

📊 Includes a trendline and color-coded points based on app categories.

💡 Insight: Helps identify whether more installs directly contribute to higher revenue.

### 2️⃣ Dual-Axis Chart (Installs & Revenue for Free vs. Paid Apps)

🔄 Compares installs and revenue within the top 3 app categories.

📏 Applies multiple filters (installs > 10,000, revenue > $10,000, etc.).

⏳ Displayed only from 1 PM - 2 PM IST for controlled data access.

💡 Insight: Shows how pricing models impact revenue and install trends.

### 3️⃣ Grouped Bar Chart (Ratings & Review Count)

📊 Compares average rating and total review count for the top 10 app categories.

🏗 Filters out categories where the average rating is below 4.0, size is below 10MB, and the last update was in January.

⏳ Displayed only from 3 PM - 5 PM IST for controlled data access.

💡 Insight: Helps determine which app categories receive the most positive feedback.

### 4️⃣ Time Series Line Chart (Total Installs Over Time)

📈 Shows the trend of total installs over time, segmented by app category.

🔍 Highlights periods of significant growth (>20% month-over-month increase).

📊 Includes filtering conditions (app name restrictions, category starting with specific letters, reviews > 500).

⏳ Displayed only from 6 PM - 9 PM IST for controlled data access.

💡 Insight: Identifies trends and potential factors driving app installs over time.

### 5️⃣ Bubble Chart (App Size vs. Average Rating)

🔵 Analyzes the relationship between app size (in MB) and average rating.

📍 Bubble size represents the number of installs.

📏 Filters: rating > 3.5, specific app categories, reviews > 500, sentiment subjectivity > 0.5, installs > 50k.

⏳ Displayed only from 5 PM - 7 PM IST for controlled data access.

💡 Insight: Determines whether larger apps tend to receive better ratings, helping developers optimize app size.

## 🖥 Final Dashboard

A comprehensive dashboard is created to visualize all five tasks interactively.

## 🔧 Data Processing

🧹 Cleaning: Handling missing values and duplicates.

🔄 Transformation: Converting columns like installs and price into numeric formats.

🔗 Merging: Combining Play Store data with user reviews on the App column.

## 🛠 Tools & Libraries

🐍 Python: Pandas, NumPy, NLTK

📊 Visualization: Plotly, Matplotlib, Seaborn

🤖 Machine Learning & NLP: Scikit-learn (for data processing), NLTK (for text analysis), WordCloud (for review sentiment visualization)


## ⚙️ Setup & Dependencies

### 📥 Clone the repository:

git clone https://github.com/bhanuu05/NULLCLASS-Google-Play-Store-Analysis.git

## 💪 Key Challenges & Solutions
⏳ Time-Restricted Graphs: Implemented Python time-check functions to control visibility dynamically.

📊 Complex Data Cleaning: Used Pandas and NumPy to handle missing values, outliers, and formatting issues.

📈 Optimized Dashboard Performance: Reduced load time by 40% using efficient HTML structuring & lazy loading.

🎡 Dark/Light Mode Toggle: Created a smooth UI transition with local storage to save preferences.

## 📢 Outcomes & Impact
Successfully created 10+ dynamic interactive visualizations for Google Play Store insights.

Reduced dashboard load time by 40%, ensuring smooth performance.

Implemented real-time accessibility based on time-based conditions.

Improved data storytelling through intuitive, visually engaging graphs.

## 👤 Acknowledgment
This project was developed under NullClass Edtech Pvt Ltd as part of the internship program with @copyright 2025.

For any queries, feel free to connect via LinkedIn [ https:/www.linkedin.com/in/bhanuu05 ] or email at bhanuusingh.01@gmail.com

🚀 Live Project Link:
https://playstore-analysis.netlify.app/



