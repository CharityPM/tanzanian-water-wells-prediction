# 💧 Waterpoint Status Classification in Tanzania

This project focuses on predicting the functionality status of waterpoints in Tanzania using machine learning. Access to clean water is essential, and identifying non-functional waterpoints ensures timely maintenance and resource allocation.

## 🔍 Project Objectives

- **Classify** waterpoints into two categories:  
  - `functional`  
  - `needs_attention`
- **Compare** multiple classification models.
- **Select** the best performing model.
- **Evaluate** using relevant metrics like accuracy, precision, recall, and F1-score.
- **Visualize** results for clear interpretation.

---

## 📂 Dataset

The dataset consists of waterpoints across various regions in Tanzania, with features including:

- `amount_tsh` (total static head in meters)
- `gps_height`
- `population`
- `well_age`
- `installer`, `basin`, `region`, `scheme_management`, etc.

Target variable:
- `status_group`: Whether a waterpoint is `functional` or `needs_attention`.

---

## 🧪 Models Developed

- **Logistic Regression**:  
  Accuracy = 77.49%  
  - Good at identifying functional wells.
  - Missed several wells needing attention.

- **Decision Tree**:  
  Accuracy = 77.0%  
  - High recall for functional wells.
  - Moderate performance for identifying faulty wells.

- **Random Forest (Baseline)**:  
  Accuracy = 81.87%  
  - Balanced performance across both classes.
  - Best among base models.

- **Random Forest (Tuned)**:  
  Accuracy = 82%  
  - Improved recall and F1-score.
  - Robust and reliable classifier.

---

## 📊 Exploratory Data Analysis & Visualization

### Load the data

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('waterpoint_data.csv')
df.head()

Sample Visualizations
python
Copy
Edit
# Class distribution
sns.countplot(data=df, x='status_group')
plt.title('Waterpoint Status Distribution')
plt.xticks(rotation=15)
plt.show()
# Population vs Status
sns.boxplot(data=df, x='status_group', y='population')
plt.yscale('log')
plt.title('Population by Waterpoint Status')
plt.show()

|| Metric           | Functional | Needs Attention |
| ---------------- | ---------- | --------------- |
| Precision        | 0.82       | 0.82            |
| Recall           | 0.86       | 0.77            |
| F1-Score         | 0.84       | 0.79            |
| Overall Accuracy | 82%        | -               |
              |
 ROC Curve
The ROC AUC for the final model was 0.89, indicating a strong ability to distinguish between classes.

python
Copy
Edit
from sklearn.metrics import roc_curve, roc_auc_score
y_probs = rf_model.predict_proba(X_val_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_val, y_probs, pos_label='needs_attention')
roc_auc = roc_auc_score(y_val, y_probs)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.grid()
plt.show()
✅ Conclusion
Random Forest outperformed logistic regression and decision tree classifiers.

It balanced false positives and false negatives better.

High accuracy, precision, and recall make it suitable for real-world deployment.

🔁 Recommendations
Monitor false positives: Ensure predicted functional wells are actually working.

Field validation: Periodically validate predictions with on-ground checks.

Model retraining: Update model periodically with new data to maintain accuracy.

Use ensemble voting: Combine models in production for even better performance.

🧑‍💻 Tech Stack
Python 🐍

pandas, numpy

scikit-learn

matplotlib, seaborn

📁 Project Structure
kotlin
Copy
Edit
├── data/
│   └── waterpoint_data.csv
├── notebooks/
│   └── model_building.ipynb
├── README.md
└── requirements.txt
✨ Acknowledgements
This project is inspired by the real-world water infrastructure challenges in Tanzania. The dataset was derived from the Taarifa project and other public water point mapping initiatives.
