# Diagnostyka usterek motocykla: system ekspertowy (Z2)

Autor: Jakub Gromowski

Baza wiedzy (`wiedza/moja_dziedzina.json`) zawiera 20 reguł i trzy poziomy atrybutów:
dane wejściowe (`rozrusznik`, `napiecie_v`, `kontrolka_fi`, `objaw`) → wnioski pośrednie
(`stan_akumulatora`, `etap_problemu`) → decyzja (`diagnoza`).

## Uruchamianie

```
python se/sprawdz_format.py wiedza/moja_dziedzina.json
python se/test_silnik.py            # wymaga wiedza/kawa.json od prowadzącego
python se/test_moja_dziedzina.py
python se/silnik.py
```

## Konsultacja wstecz (zapis z terminala)

```
=== Wnioskowanie wstecz ===
Jakie jest napięcie akumulatora na postoju (V)? 12,6
Czy rozrusznik kręci silnikiem? (kreci/nie_kreci) kreci
Czy świeci kontrolka usterki wtrysku (FI)? (swieci/nie_swieci) nie_swieci
Jaki jest objaw pracy silnika? (nie_odpala/gasnie/nierowno/brak_mocy) gasnie
Polecam: paliwo
Reguły: R4 -> R16
```

System zapytał najpierw o napięcie, bo pierwszą regułą dającą diagnozę jest R5 (napięcie ≤ 10,5 V
oznacza akumulator) i musiał sprawdzić jej warunek. Potem pytał o rozrusznik (żeby ustalić
`etap_problemu`), kontrolkę FI i objaw, bo tego wymagały kolejne reguły z diagnozą. Nie zapytał o
`stan_akumulatora`, bo ten wniosek wynika z napięcia i system ustala go sam.

Kolejność reguł: reguła nadrzędna R5 (bardzo niskie napięcie = akumulator) stoi przed regułami
szczegółowymi R6–R20, więc przy konflikcie wygrywa ona jako pierwsza pasująca.

## Z3: wyjaśnienia, weryfikacja i interfejs

```
python se/weryfikacja.py wiedza/kawa_z_bledami.json   # 10 problemów
python se/weryfikacja.py wiedza/kawa.json             # 0 problemów
python se/weryfikacja.py wiedza/moja_dziedzina.json   # 0 problemów
python -m streamlit run app.py
```

Weryfikator nie zgłasza dla mojej bazy ani błędów, ani uwag. Świadome ograniczenia, których nie wykrywa:
reguła R5 (napięcie ≤ 10,5 V) nakłada się z R1 (≤ 12,0 V) i konkuruje z R14, R17 i R19; konflikt
rozstrzyga kolejność reguł (R5 stoi wyżej). Wszystkie 96 kombinacji wejść (z wartościami progowymi) dają diagnozę.

## Z4: dane z silnika i drzewo decyzyjne

```
python ml/generator.py    # ml/dane.csv, ml/luki.json, ml/metadane.json
python ml/trening.py      # ml/podzial.json (tworzony raz), ml/drzewo.txt, ml/drzewo.png
```

Dane: 1000 losowań (ziarno 42), każde wejście równomiernie z dziedziny (napięcie 8,0–15,0 V z krokiem 0,1 V),
etykieta z silnika wnioskowaniem w przód (`label_source = rule_engine`), szum etykiet: brak.
662 unikalne wejścia, 0 przypadków bez decyzji. Podział: StratifiedGroupKFold(5, shuffle, random_state=42),
pierwszy fold jako test (200 przypadków); grupa = identyczna kombinacja wejść. Test zamknięty do Z6.

Drzewo: Pipeline(OneHotEncoder + DecisionTreeClassifier(criterion="entropy", random_state=42)),
głębokość 5, 12 liści, korzeń `napiecie_v <= 12.05`, zgodność z etykietami silnika na treningu 100%.

### Drzewo a reguły eksperta

| Nr | Fragment drzewa | Reguła | Zgodność | Wyjaśnienie |
|---|---|---|---|---|
| 1 | `napiecie_v <= 10.55` → akumulator | R5: `napiecie_v <= 10.5` → akumulator | zgodne | próg 10,55 leży w połowie między 10,5 a 10,6 (krok danych 0,1 V) |
| 2 | `napiecie_v > 12.05`, `rozrusznik_kreci <= 0.5`, `kontrolka_fi_nie_swieci > 0.5` → elektronika | R2 + R3 + R7 | zgodne | łańcuch trzech reguł jako jedna ścieżka; wnioski pośrednie zastąpione progiem i cechami one-hot |
| 3 | `10.55 < napiecie_v <= 12.05`, `kontrolka_fi_nie_swieci > 0.5`, `rozrusznik_nie_kreci <= 0.5`, `objaw_gasnie <= 0.5` → zaplon | R14, R17, R19 | zgodne, uogólnione | jeden liść scala trzy reguły: zamiast trzech objawów jeden warunek „objaw ≠ gasnie” |

Ograniczenia: etykiety pochodzą z silnika, więc zgodność mierzy tylko, jak drzewo odtwarza reguły, a nie
trafność diagnoz (to walidacja z etykietami eksperta). Rozkład klas wynika z reguł i równomiernego losowania
(akumulator 52%, zaplon 4%). Drzewo nie ma pojęć pośrednich (stan_akumulatora, etap_problemu) — progi i
one-hot zastępują je bezpośrednio.

## Z5: walidacja krzyżowa, strojenie i wybór modelu

```
python ml/strojenie.py    # ml/wyniki_cv.csv, ml/glebokosc.png, ml/roc.png, ml/raport_oof.json, ml/model.joblib, ml/model_info.json
```

Wszystkie wyniki pochodzą ze zbioru treningowego (800 przypadków); test z `ml/podzial.json` pozostaje zamknięty do Z6.
GridSearchCV: `max_depth` ∈ {2, 3, 4, 5, 6, 8, None} × `min_samples_leaf` ∈ {1, 5, 10} (21 ustawień),
StratifiedGroupKFold(5, shuffle=True, random_state=42), grupa = identyczne wejścia, metryka wyboru `f1_macro`.

**Wybrany model:** `max_depth=5`, `min_samples_leaf=5` (12 liści), macro-F1 w CV = 1,000 ± 0,000.
Osiem ustawień ma macro-F1 = 1,000 (etykiety pochodzą z silnika, bez szumu). Przy remisie wybrano prostszy
model: najmniejszą głębokość, a następnie większy `min_samples_leaf` (odporność na pojedyncze przypadki).
Głębokości 2–4 dają niedouczenie (0,35 / 0,62 / 0,93); od 5 trening i walidacja są równe, więc przeuczenia nie ma.

Dla porównania drzewo `max_depth=4` (też out-of-fold) myli `paliwo → zaplon` (12) i `dolot → elektronika` (11).
Najkosztowniejsza jest pomyłka `paliwo → zaplon`: silnik gaśnie w czasie jazdy z powodu paliwa, a naprawa
układu zapłonowego nie usuwa przyczyny. Model zapisany jako Pipeline (`ml/model.joblib`) z listą klas w `ml/model_info.json`.
