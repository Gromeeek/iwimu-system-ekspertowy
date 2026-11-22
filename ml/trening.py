"""Podział danych i drzewo decyzyjne (spotkanie 4).
Uruchom z katalogu głównego repozytorium:  python ml/trening.py

1. Podział: StratifiedGroupKFold(5) — jeden fold jako test (warstwowo według diagnozy,
   grupa = identyczna kombinacja wejść, więc powtórzenia nie trafiają do obu części).
   Identyfikatory zapisane w ml/podzial.json; zbiór testowy pozostaje nieużywany do Z6.
2. Pipeline: OneHotEncoder dla cech kategorycznych + DecisionTreeClassifier; fit tylko na treningu.
3. Wynik: ml/drzewo.txt (export_text), ml/drzewo.png (plot_tree).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree  # noqa: E402

KATALOG = Path(__file__).parent
SEED = 42
CEL = "diagnoza"
SYMB = ["rozrusznik", "kontrolka_fi", "objaw"]     # cechy kategoryczne → one-hot
LICZB = ["napiecie_v"]                             # cecha liczbowa → bez zmian
WEJSCIA = SYMB + LICZB                             # X: tylko dane dostępne przy decyzji

dane = pd.read_csv(KATALOG / "dane.csv")
grupy = dane[WEJSCIA].astype(str).agg("|".join, axis=1)    # identyczne wejścia = jedna grupa

# --- 1. Podział (tworzony raz; potem zawsze wczytywany z pliku) ---
plik_podzialu = KATALOG / "podzial.json"
if not plik_podzialu.exists():
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    idx_tren, idx_test = next(cv.split(dane, dane[CEL], groups=grupy))
    podzial = {
        "metoda": "StratifiedGroupKFold(n_splits=5, shuffle=True), pierwszy fold jako test",
        "grupa": "identyczna kombinacja wejść " + ", ".join(WEJSCIA),
        "warstwowanie": CEL,
        "test_size": round(len(idx_test) / len(dane), 3),
        "random_state": SEED,
        "test_ids": sorted(int(i) for i in dane.loc[idx_test, "sample_id"]),
        "trening_ids": sorted(int(i) for i in dane.loc[idx_tren, "sample_id"]),
    }
    plik_podzialu.write_text(json.dumps(podzial, ensure_ascii=False, indent=1), encoding="utf-8")
podzial = json.loads(plik_podzialu.read_text(encoding="utf-8"))

trening = dane[dane["sample_id"].isin(podzial["trening_ids"])]
test = dane[dane["sample_id"].isin(podzial["test_ids"])]
assert not set(grupy[trening.index]) & set(grupy[test.index]), "te same wejścia w treningu i teście"

print("Liczebność klas (wszystkie / unikalne wejścia / trening / test):")
tabela = pd.DataFrame({
    "przypadki": dane[CEL].value_counts(),
    "unikalne": grupy.groupby(dane[CEL]).nunique(),
    "trening": trening[CEL].value_counts(),
    "test": test[CEL].value_counts(),
}).fillna(0).astype(int).sort_values("przypadki", ascending=False)
tabela.loc["Razem"] = [len(dane), grupy.nunique(), len(trening), len(test)]
print(tabela.to_string(), "\n")

# --- 2. Pipeline: kodowanie i drzewo w jednym obiekcie ---
kod = ColumnTransformer(
    [("sym", OneHotEncoder(handle_unknown="ignore"), SYMB)],
    remainder="passthrough",                        # napiecie_v bez zmian; drzewo samo szuka progu
    verbose_feature_names_out=False,
)
model = Pipeline([
    ("kod", kod),
    ("drzewo", DecisionTreeClassifier(criterion="entropy", random_state=SEED)),
])
X_tren, y_tren = trening[WEJSCIA], trening[CEL]
model.fit(X_tren, y_tren)                           # tylko trening! test zostaje zamknięty do Z6

drzewo = model.named_steps["drzewo"]
cechy = list(model.named_steps["kod"].get_feature_names_out())
print(f"Kryterium: {drzewo.criterion}, głębokość: {drzewo.get_depth()}, liście: {drzewo.get_n_leaves()}, "
      f"korzeń: {cechy[drzewo.tree_.feature[0]]} <= {drzewo.tree_.threshold[0]:.2f}")
print(f"Zgodność z etykietami silnika na treningu: {model.score(X_tren, y_tren):.3f}\n")

# --- 3. Reguły drzewa i rysunek ---
reguly = export_text(drzewo, feature_names=cechy, decimals=2, show_weights=False)
(KATALOG / "drzewo.txt").write_text(reguly, encoding="utf-8")
print(reguly)

def etykieta_wezla(tekst):
    """Skraca opis węzła: one-hot „cecha_wartosc <= 0.5” → „cecha ≠ wartosc” (gałąź „tak”), bez listy value."""
    if tekst.strip() in ("True", "False"):
        return tekst.replace("True", "tak").replace("False", "nie")
    linie = []
    for linia in tekst.split("\n"):
        if linia.startswith("value"):
            continue
        if linia.startswith("samples = "):
            linia = "n = " + linia.split("= ")[1]
        elif linia.startswith("class = "):
            linia = linia.split("= ")[1].upper()
        elif linia.endswith("<= 0.5"):
            nazwa = linia[:-len(" <= 0.5")]
            for c in SYMB:
                if nazwa.startswith(c + "_"):
                    linia = f"{c} ≠ {nazwa[len(c) + 1:]}"
        else:
            linia = linia.replace(".", ",").replace("<=", "≤")   # przecinek dziesiętny w progach liczbowych
        linie.append(linia)
    return "\n".join(linie)


fig, ax = plt.subplots(figsize=(15.5, 7))
wezly = plot_tree(drzewo, feature_names=cechy, class_names=list(drzewo.classes_), filled=True,
                  rounded=True, impurity=False, fontsize=10, ax=ax)
for w in wezly:
    w.set_text(etykieta_wezla(w.get_text()))
fig.savefig(KATALOG / "drzewo.png", dpi=200, bbox_inches="tight")
