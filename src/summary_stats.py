import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

root = Path(__file__).resolve().parent.parent

matches = pd.read_csv(root / "data" / "processed" / "matches_clean.csv")

print("1) Result split (%)")
print((matches["Result"].value_counts(normalize=True) * 100).round(1))

print("\n2) Home win % by season")
print((matches.groupby("Season")["Result"].apply(lambda s: (s == "Home Win").mean() * 100)).round(1))

print("\n3) Average goals per match:", round(matches["TotalGoals"].mean(), 2))

home = matches.assign(Team=matches["HomeTeam"], Points=matches["Target"].map({0: 3, 1: 1, 2: 0}))
away = matches.assign(Team=matches["AwayTeam"], Points=matches["Target"].map({2: 3, 1: 1, 0: 0}))
games = pd.concat([home[["Team", "Points"]], away[["Team", "Points"]]])
table = games.groupby("Team").agg(Matches=("Points", "count"), Points=("Points", "sum"))
table["PointsPerMatch"] = (table["Points"] / table["Matches"]).round(2)
print("\n4) Top 5 teams by points per match (76+ matches)")
print(table[table["Matches"] >= 76].sort_values("PointsPerMatch", ascending=False).head(5))

df = pd.read_csv(root / "data" / "processed" / "model_features.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
feature_cols = [c for c in df.columns if c.startswith(("Home_", "Away_")) or c.endswith("_form_diff")]

train = df[df["Season"] != "2025-26"]
test = df[df["Season"] == "2025-26"]
X_train, y_train = train[feature_cols], train["Target"]
X_test, y_test = test[feature_cols], test["Target"]

rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)

print("\n5) Test matches:", len(test), "| Random Forest accuracy:", round(accuracy_score(y_test, pred), 3))
print("\n6) Random Forest classification report")
print(classification_report(y_test, pred, target_names=["Home Win", "Draw", "Away Win"], zero_division=0))

print("7) Top 5 features")
print(pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(5).round(3))