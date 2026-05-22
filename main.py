import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


# ==============================================================
# 1. LOAD DATA
# ==============================================================

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'level'
]

print("Loading data...")
df_train = pd.read_csv(train_url, names=columns)
df_test = pd.read_csv(test_url, names=columns)

df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'], inplace=True)

print(f"Training set: {df_train.shape}")
print(f"Test set: {df_test.shape}")


# ==============================================================
# 2. ENCODE CATEGORICAL FEATURES
# ==============================================================

df_full = pd.concat([df_train, df_test], ignore_index=True)

cat_cols = ['protocol_type', 'service', 'flag']

for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])


# ==============================================================
# 3. MAP ATTACKS TO 5 CATEGORIES
# ==============================================================

category_map = {
    'normal': 'Normal',

    # DoS
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',

    # Probe
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',

    # R2L
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'ftp_write': 'R2L',
    'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L', 'warezmaster': 'R2L',
    'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',

    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')


# ==============================================================
# 4. PREPARE FEATURES AND LABELS
# ==============================================================

df_full.drop(columns=['num_outbound_cmds', 'class'], inplace=True)

train_len = len(df_train)

df_train_processed = df_full.iloc[:train_len].copy()
df_test_processed = df_full.iloc[train_len:].copy()

X_train = df_train_processed.drop(columns=['category'])
y_train = df_train_processed['category']

X_test = df_test_processed.drop(columns=['category'])
y_test = df_test_processed['category']

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())


# ==============================================================
# 5. TRAIN MODEL
# ==============================================================

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

print("\nRunning 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1
)

print(f"Cross-validation macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


print("\nTraining final model...")
model.fit(X_train, y_train)


# ==============================================================
# 6. TEST MODEL
# ==============================================================

print("\nEvaluating on test set...")
y_pred = model.predict(X_test)

test_macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"\nFinal test macro F1-score: {test_macro_f1:.4f}")

print("\nClassification report:")
report = classification_report(y_test, y_pred)
print(report)


# Save report text to file
with open("classification_report.txt", "w") as f:
    f.write(f"Cross-validation macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
    f.write(f"Final test macro F1-score: {test_macro_f1:.4f}\n\n")
    f.write(report)


# ==============================================================
# 7. CONFUSION MATRIX
# ==============================================================

labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]

cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

print("\nSaved confusion_matrix.png")
print("Saved classification_report.txt")