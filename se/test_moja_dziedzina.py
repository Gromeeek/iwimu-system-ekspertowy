"""Autotest bazy 'Diagnostyka usterek motocykla'. Uruchom: python se/test_moja_dziedzina.py"""
from pathlib import Path

from silnik import wczytaj_baze, wnioskuj_w_przod, wnioskuj_wstecz

baza = wczytaj_baze(Path(__file__).parent.parent / "wiedza" / "moja_dziedzina.json")
CEL = baza["cel"]

# (opis, fakty, oczekiwana diagnoza, oczekiwany ślad w przód albo None)
PRZYPADKI = [
    ("typowy: rozrusznik nie kręci, akumulator słaby",
     {"rozrusznik": "nie_kreci", "napiecie_v": 11.0, "kontrolka_fi": "swieci", "objaw": "nie_odpala"},
     "akumulator", ["R1", "R3", "R6"]),
    ("typowy: kręci, kontrolka świeci, gaśnie",
     {"rozrusznik": "kreci", "napiecie_v": 12.8, "kontrolka_fi": "swieci", "objaw": "gasnie"},
     "dolot", ["R2", "R4", "R11"]),
    ("PRÓG reguł R1/R2: dokładnie 12.0 V należy do 'slaby' (<=)",
     {"rozrusznik": "kreci", "napiecie_v": 12.0, "kontrolka_fi": "nie_swieci", "objaw": "nie_odpala"},
     "zaplon", ["R1", "R4", "R14"]),
    ("tuż za progiem: 12.1 V to już 'dobry'",
     {"rozrusznik": "kreci", "napiecie_v": 12.1, "kontrolka_fi": "nie_swieci", "objaw": "nie_odpala"},
     "paliwo", ["R2", "R4", "R15"]),
    ("PRÓG reguły R5: dokładnie 10.5 V",
     {"rozrusznik": "kreci", "napiecie_v": 10.5, "kontrolka_fi": "nie_swieci", "objaw": "nierowno"},
     "akumulator", ["R1", "R4", "R5"]),
    ("KONFLIKT: pasują R5 (akumulator) i R17 (zapłon), wygrywa R5 jako wyższa",
     {"rozrusznik": "kreci", "napiecie_v": 10.0, "kontrolka_fi": "nie_swieci", "objaw": "nierowno"},
     "akumulator", ["R1", "R4", "R5"]),
    ("tuż nad progiem R5 (10.6 V): R5 nie pasuje, działa R17",
     {"rozrusznik": "kreci", "napiecie_v": 10.6, "kontrolka_fi": "nie_swieci", "objaw": "nierowno"},
     "zaplon", ["R1", "R4", "R17"]),
    ("BEZ WYNIKU: objaw spoza bazy ('stuka')",
     {"rozrusznik": "kreci", "napiecie_v": 12.6, "kontrolka_fi": "nie_swieci", "objaw": "stuka"},
     None, None),
]

ok = 0
for opis, wejscie, oczekiwane, oczekiwany_slad in PRZYPADKI:
    fakty, slad = wnioskuj_w_przod(baza, wejscie)
    w_przod_ok = fakty.get(CEL) == oczekiwane and (oczekiwany_slad is None or slad == oczekiwany_slad)

    zadane = []
    def zapytaj(atrybut, powod):
        zadane.append(atrybut)
        return wejscie[atrybut]
    wstecz = wnioskuj_wstecz(baza, CEL, {}, zapytaj, [])
    wstecz_ok = wstecz == oczekiwane and len(zadane) == len(set(zadane))

    status = "OK " if w_przod_ok and wstecz_ok else "BŁĄD"
    ok += w_przod_ok and wstecz_ok
    print(f"[{status}] {opis}\n        w przód: {fakty.get(CEL)} {slad}, wstecz: {wstecz}, pytań: {len(zadane)}")

print(f"\nZaliczone przypadki: {ok}/{len(PRZYPADKI)}")
