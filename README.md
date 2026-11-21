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
