import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('fake_job_postings.csv')
df = df.dropna(subset=['description', 'fraudulent'])

# Downsample real jobs
df_fake = df[df['fraudulent'] == 1]
df_real = df[df['fraudulent'] == 0].sample(n=len(df_fake), random_state=42)
df_balanced = pd.concat([df_fake, df_real])

# ----> HERE! <----
scam_jobs = pd.DataFrame({
    'title': [
        "Work from Home Opportunity",
        "Earn $5000 a week easily",
        "Instant Data Entry Money",
        "Congratulations! Send bank details"
    ],
    'description': [
        "Earn thousands from home, no experience required. Apply now!",
        "Work only 2 hours a day, no skills needed. Huge payouts!",
        "No interview, no background check. Instant payment on signup.",
        "You've been selected for a high paying job. Send your bank details to proceed."
    ],
    'requirements': ["", "", "", ""],
    'company_profile': ["", "", "", ""],
    'benefits': ["", "", "", ""],
    'fraudulent': [1, 1, 1, 1]
})
df_balanced = pd.concat([df_balanced, scam_jobs], ignore_index=True)

# Combine features
X = (
    df_balanced['title'].fillna('') + " " +
    df_balanced['description'].fillna('') + " " +
    df_balanced.get('requirements', '').fillna('') + " " +
    df_balanced.get('company_profile', '').fillna('') + " " +
    df_balanced.get('benefits', '').fillna('')
)
y = df_balanced['fraudulent']

vectorizer = TfidfVectorizer(stop_words='english', max_features=7000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

test_jobs = [
    "Earn $2000 per week from home! No experience required. All you need is a computer and internet connection. Apply now and start earning today!",
    "Exclusive offer! Data entry jobs available immediately. No interview, instant payment, limited positions. Send your details to get started!",
    "Work remotely and make $500 daily just by posting ads online. No skills or qualification needed. Hurry, spots filling fast!",
    "Immediate hiring for online survey jobs. Get paid for every survey you complete. No background check, start today!",
    "Congratulations! You've been selected for a high paying job. Just send your bank details for verification and get started instantly.",
    "Send money to get job"
]
test_vec = vectorizer.transform(test_jobs)
test_preds = model.predict(test_vec)
for job, pred in zip(test_jobs, test_preds):
    print(f"Job: {job}\nPrediction (0=Real, 1=Fake): {pred}\n")
    import pickle
with open("job_fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("job_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
    import pickle

# Load saved model and vectorizer
with open("job_fraud_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("job_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Example new jobs to test
test_jobs = [
    "Earn $2000 per week from home! No experience required. Apply now and start earning today!",
    "We are looking for a Software Engineer to join our team. Competitive salary and benefits.",
    "Congratulations! You've won a job offer. Send your bank details to proceed.",
    "Work remotely and make $500 daily just by posting ads online.",
    "Looking for a data analyst with experience in Python and SQL."
]

# Combine title and description if needed (here, just using job text)
test_vec = vectorizer.transform(test_jobs)
test_preds = model.predict(test_vec)

for job, pred in zip(test_jobs, test_preds):
    print(f"Job: {job}\nPrediction (0=Real, 1=Fake): {pred}\n")