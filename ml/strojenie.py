"""Walidacja krzyżowa, strojenie i zamrożenie modelu (spotkanie 5).
Uruchom z katalogu głównego repozytorium:  python ml/strojenie.py

Wszystko liczone WYŁĄCZNIE na zbiorze treningowym z ml/podzial.json — test zostaje zamknięty do Z6.
1. GridSearchCV: max_depth × min_samples_leaf, StratifiedGroupKFold(5), scoring = f1_macro.
2. Wybór: najwyższe macro-F1 w CV; przy remisie prostszy model (mniejsza głębokość, potem większy liść).
3. Raport klas, macierz pomyłek i ROC (OvR) z predykcji out-of-fold wybranego ustawienia.
Wynik: ml/wyniki_cv.csv, ml/glebokosc.png, ml/roc.png, ml/raport_oof.json, ml/model.joblib, ml/model_info.json
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.metrics import (auc, balanced_accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

KATALOG = Path(__file__).parent
SEED = 42
CEL = "diagnoza"
SYMB = ["rozrusznik", "kontrolka_fi", "objaw"]
LICZB = ["napiecie_v"]
WEJSCIA = SYMB + LICZB
PRZECINEK = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ","))   # przecinek dziesiętny na osiach
KLASA_ROC = "zaplon"            # klasa dodatnia dla ROC: najsłabiej reprezentowana diagnoza
METRYKA = "f1_macro"            # ustalona przed strojeniem

# --- dane: tylko trening ---
dane = pd.read_csv(KATALOG / "dane.csv")
podzial = json.loads((KATALOG / "podzial.json").read_text(encoding="utf-8"))
trening = dane[dane["sample_id"].isin(podzial["trening_ids"])].reset_index(drop=True)
X, y = trening[WEJSCIA], trening[CEL]
grupy = X.astype(str).agg("|".join, axis=1)          # identyczne wejścia = jedna grupa

kod = ColumnTransformer([("sym", OneHotEncoder(handle_unknown="ignore"), SYMB)],
                        remainder="passthrough", verbose_feature_names_out=False)
model = Pipeline([("kod", kod), ("drzewo", DecisionTreeClassifier(criterion="entropy", random_state=SEED))])
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

GLEBOKOSCI = [2, 3, 4, 5, 6, 8, None]
LISCIE = [1, 5, 10]
siatka = {"drzewo__max_depth": GLEBOKOSCI, "drzewo__min_samples_leaf": LISCIE}


def wybierz_prostszy(wyniki):
    """Indeks ustawienia: max średnie macro-F1 w CV; remis → mniejsza głębokość, potem większy min_samples_leaf."""
    srednie = np.round(wyniki["mean_test_score"], 6)
    najlepsze = [i for i, s in enumerate(srednie) if s == srednie.max()]
    def klucz(i):
        p = wyniki["params"][i]
        g = p["drzewo__max_depth"] if p["drzewo__max_depth"] is not None else 10 ** 6
        return (g, -p["drzewo__min_samples_leaf"])
    return min(najlepsze, key=klucz)


szukaj = GridSearchCV(model, siatka, scoring=METRYKA, cv=cv, return_train_score=True, refit=wybierz_prostszy)
szukaj.fit(X, y, groups=grupy)
r = szukaj.cv_results_
nr = szukaj.best_index_
najlepszy = szukaj.best_estimator_
param = szukaj.best_params_

# --- 1. wyniki strojenia ---
tabela = pd.DataFrame({
    "max_depth": [p["drzewo__max_depth"] if p["drzewo__max_depth"] is not None else "None" for p in r["params"]],
    "min_samples_leaf": [p["drzewo__min_samples_leaf"] for p in r["params"]],
    "f1_macro_trening": np.round(r["mean_train_score"], 4),
    "f1_macro_cv_srednia": np.round(r["mean_test_score"], 4),
    "f1_macro_cv_odch": np.round(r["std_test_score"], 4),
    "ranga": r["rank_test_score"],
})
tabela.to_csv(KATALOG / "wyniki_cv.csv", index=False)
print(tabela.to_string(index=False))
print("\nWybrane ustawienie:", param, f"macro-F1 CV = {r['mean_test_score'][nr]:.4f} ± {r['std_test_score'][nr]:.4f}")

# --- 2. wykres głębokości (min_samples_leaf = 1) ---
etyk = [str(g) if g is not None else "bez limitu" for g in GLEBOKOSCI]
wiersze = [tabela[(tabela["min_samples_leaf"] == 1) & (tabela["max_depth"].astype(str) == str(g))].iloc[0]
           for g in GLEBOKOSCI]
tren = [w["f1_macro_trening"] for w in wiersze]
wal = [w["f1_macro_cv_srednia"] for w in wiersze]
odch = [w["f1_macro_cv_odch"] for w in wiersze]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(etyk, tren, marker="o", label="trening")
ax.errorbar(etyk, wal, yerr=odch, marker="s", capsize=4, label="walidacja (CV, średnia ± odch.)")
ax.set_xlabel("max_depth (min_samples_leaf = 1)")
ax.set_ylabel("macro-F1")
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(PRZECINEK)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.savefig(KATALOG / "glebokosc.png", dpi=200, bbox_inches="tight")

# --- 3. raport klas z predykcji out-of-fold (wybrane ustawienie, te same foldy) ---
y_oof = cross_val_predict(najlepszy, X, y, groups=grupy, cv=cv)
p_oof = cross_val_predict(najlepszy, X, y, groups=grupy, cv=cv, method="predict_proba")
klasy = list(najlepszy.classes_)
kolejnosc = list(y.value_counts().index)            # od najliczniejszej
raport = classification_report(y, y_oof, labels=kolejnosc, output_dict=True, zero_division=0)
macierz = confusion_matrix(y, y_oof, labels=kolejnosc)
print("\n" + classification_report(y, y_oof, labels=kolejnosc, digits=3, zero_division=0))
print("balanced accuracy:", round(balanced_accuracy_score(y, y_oof), 4))
print("Macierz pomyłek (wiersze: prawda, kolumny: predykcja):", kolejnosc)
print(macierz)

# --- 3a. dla porównania: płytsze drzewo (max_depth = 4) — też tylko out-of-fold na treningu ---
plytszy = clone(model).set_params(drzewo__max_depth=4, drzewo__min_samples_leaf=1)
y_oof4 = cross_val_predict(plytszy, X, y, groups=grupy, cv=cv)
p_oof4 = cross_val_predict(plytszy, X, y, groups=grupy, cv=cv, method="predict_proba")
raport4 = classification_report(y, y_oof4, labels=kolejnosc, output_dict=True, zero_division=0)
macierz4 = confusion_matrix(y, y_oof4, labels=kolejnosc)
print("\nPorównanie — max_depth = 4 (out-of-fold):")
print(classification_report(y, y_oof4, labels=kolejnosc, digits=3, zero_division=0))
print(macierz4)

# --- 4. ROC OvR dla jednej klasy ---
y_bin = (y == KLASA_ROC).astype(int)
wynik = p_oof[:, klasy.index(KLASA_ROC)]
fpr, tpr, _ = roc_curve(y_bin, wynik)
auc_klasy = auc(fpr, tpr)
fpr4, tpr4, _ = roc_curve(y_bin, p_oof4[:, klasy.index(KLASA_ROC)])
auc4 = auc(fpr4, tpr4)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, marker="o", label=f"wybrany model (max_depth = {param['drzewo__max_depth']}), AUC = {auc_klasy:.3f}".replace(".", ","))
ax.plot(fpr4, tpr4, marker="^", label=f"dla porównania: max_depth = 4, AUC = {auc4:.3f}".replace(".", ","))
ax.plot([0, 1], [0, 1], "--", color="grey", label="model losowy")
ax.set_title(f"ROC (OvR), klasa dodatnia: {KLASA_ROC}, support = {int(y_bin.sum())}")
ax.set_xlabel("odsetek fałszywych alarmów (FPR)")
ax.set_ylabel("czułość (TPR)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.xaxis.set_major_formatter(PRZECINEK); ax.yaxis.set_major_formatter(PRZECINEK)
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.savefig(KATALOG / "roc.png", dpi=200, bbox_inches="tight")
print(f"\nROC: klasa {KLASA_ROC}, OvR, support {int(y_bin.sum())}, CV out-of-fold, AUC = {auc_klasy:.4f} (max_depth=4: {auc4:.4f})")

(KATALOG / "raport_oof.json").write_text(json.dumps({
    "zbior": "trening, predykcje out-of-fold (StratifiedGroupKFold 5)",
    "klasy": kolejnosc,
    "raport": raport,
    "macro_f1": round(f1_score(y, y_oof, average="macro"), 4),
    "balanced_accuracy": round(balanced_accuracy_score(y, y_oof), 4),
    "macierz_pomylek": macierz.tolist(),
    "roc": {"klasa_dodatnia": KLASA_ROC, "strategia": "one-vs-rest", "support": int(y_bin.sum()),
            "zbior": "trening, CV out-of-fold", "auc": round(auc_klasy, 4), "auc_max_depth_4": round(auc4, 4)},
    "porownanie_max_depth_4": {"raport": raport4, "macro_f1": round(f1_score(y, y_oof4, average="macro"), 4),
                               "balanced_accuracy": round(balanced_accuracy_score(y, y_oof4), 4),
                               "macierz_pomylek": macierz4.tolist()},
}, ensure_ascii=False, indent=2), encoding="utf-8")

# --- 5. zamrożenie modelu (refit na całym treningu) ---
joblib.dump(najlepszy, KATALOG / "model.joblib")
drzewo = najlepszy.named_steps["drzewo"]
info = {
    "model": "Pipeline(OneHotEncoder + DecisionTreeClassifier(criterion='entropy'))",
    "parametry": {"max_depth": param["drzewo__max_depth"], "min_samples_leaf": param["drzewo__min_samples_leaf"],
                  "criterion": "entropy", "random_state": SEED},
    "metryka_wyboru": METRYKA,
    "wynik_cv": {"srednia": round(float(r["mean_test_score"][nr]), 4), "odch": round(float(r["std_test_score"][nr]), 4)},
    "walidacja": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), grupa = identyczne wejścia",
    "regula_wyboru": "najwyższe macro-F1 w CV; remis → mniejsza głębokość, potem większy min_samples_leaf",
    "klasy": klasy,
    "cechy": WEJSCIA,
    "glebokosc": int(drzewo.get_depth()),
    "liscie": int(drzewo.get_n_leaves()),
    "n_trening": int(len(trening)),
    "zbior_testowy": "nieużywany (Z6)",
}
(KATALOG / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nmodel_info.json:", json.dumps(info, ensure_ascii=False))
