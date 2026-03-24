import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1 Caricamento dati
data = pd.read_csv("01_tickets_sintetici.csv", sep=",", encoding="cp1252")
# data["subject"] = data["subject"].fillna("")
# data["description"] = data["description"].fillna("")
# data["category"] = data["category"].fillna("")
# data["subject"] = data["subject"].str.lower()
# data["description"] = data["description"].str.lower()

# 2 Feature e target
X = data[["title", "body", "category"]]
y = data["priority"]

# 3 Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4 Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("title_tfidf", TfidfVectorizer(), "title"),
        ("body_tfidf", TfidfVectorizer(), "body"),
        ("category_ohe", OneHotEncoder(handle_unknown="ignore"), ["category"])
    ]
)

# 5 Pipeline completa
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 6 Addestramento
model.fit(X_train, y_train)

# 7 Predizione
y_pred = model.predict(X_test)

accuratezza = accuracy_score(y_test, y_pred)
matrice_confusione = confusion_matrix(y_test, y_pred)
classificazione = classification_report(y_test, y_pred)

# 8 Valutazione
print(f"Accuratezza: {accuratezza * 100:.2f}%" )
print("\nMatrice di confusione:")
print(matrice_confusione)
print("\nClassification Report:")
print(classificazione)
