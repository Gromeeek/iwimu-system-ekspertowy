"""Generator danych uczących z własnego silnika (spotkanie 4).
Uruchom z katalogu głównego repozytorium:  python ml/generator.py

Wejścia losuje z dziedzin zadeklarowanych w wiedza/moja_dziedzina.json,
etykietę (diagnozę) ustala silnik wnioskowaniem w przód.
Wynik: ml/dane.csv, ml/luki.json, ml/metadane.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

KATALOG = Path(__file__).parent
sys.path.insert(0, str(KATALOG.parent / "se"))
from silnik import wczytaj_baze, wnioskuj_w_przod  # noqa: E402

PLIK_BAZY = KATALOG.parent / "wiedza" / "moja_dziedzina.json"
SEED = 42
N = 1000                 # liczba losowań
KROK_LICZBY = 0.1        # napięcie losowane z krokiem 0,1 V (progi reguł: 10,5 i 12,0 V)

baza = wczytaj_baze(PLIK_BAZY)
A = baza["atrybuty"]
CEL = baza["cel"]
WEJSCIA = [n for n, opis in A.items() if opis["typ"] != "wniosek"]   # tylko dane dostępne przy decyzji


def losuj_wejscie(rng):
    x = {}
    for n in WEJSCIA:
        if A[n]["typ"] == "liczba":
            lo, hi = A[n]["zakres"]
            kroki = int(round((hi - lo) / KROK_LICZBY))
            x[n] = round(lo + int(rng.integers(0, kroki + 1)) * KROK_LICZBY, 1)   # równomiernie: 8,0 … 15,0
        else:
            x[n] = str(rng.choice(A[n]["wartosci"]))                            # równomiernie z listy wartości
    return x


def main():
    rng = np.random.default_rng(SEED)
    wiersze, luki = [], []
    for i in range(1, N + 1):
        x = losuj_wejscie(rng)
        fakty, slad = wnioskuj_w_przod(baza, x)
        if fakty.get(CEL) is None:                       # brak decyzji → raport luk, nie do danych
            luki.append({"sample_id": i, **x})
            continue
        wiersze.append({"sample_id": i, **x, CEL: fakty[CEL], "label_source": "rule_engine"})

    kolumny = ["sample_id", *WEJSCIA, CEL, "label_source"]
    with open(KATALOG / "dane.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kolumny)
        w.writeheader()
        w.writerows(wiersze)
    (KATALOG / "luki.json").write_text(json.dumps(luki, ensure_ascii=False, indent=2), encoding="utf-8")

    unikalne = {tuple(r[n] for n in WEJSCIA) for r in wiersze}
    metadane = {
        "baza": PLIK_BAZY.name,
        "ziarno": SEED,
        "liczba_losowan": N,
        "probkowanie": {
            n: (f"równomiernie {A[n]['zakres'][0]}–{A[n]['zakres'][1]} z krokiem {KROK_LICZBY}"
                if A[n]["typ"] == "liczba" else "równomiernie z listy " + str(A[n]["wartosci"]))
            for n in WEJSCIA
        },
        "wiersze": len(wiersze),
        "unikalne_wejscia": len(unikalne),
        "przypadki_bez_decyzji": len(luki),
        "szum_etykiet": "brak",
        "liczebnosc_klas": {k: sum(r[CEL] == k for r in wiersze) for k in A[CEL]["wartosci"]},
    }
    (KATALOG / "metadane.json").write_text(json.dumps(metadane, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadane, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
