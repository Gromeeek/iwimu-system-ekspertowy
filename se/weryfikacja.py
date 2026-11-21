"""Weryfikacja bazy wiedzy (odpowiednik programu Cake z pakietu SPHINX).
Uruchom: python se/weryfikacja.py wiedza/kawa_z_bledami.json
"""
import sys
from pathlib import Path

from silnik import OPERATORY, wczytaj_baze


def weryfikuj(baza):
    """Zwraca listę (poziom, komunikat); poziom to 'BŁĄD' albo 'UWAGA'."""
    raport = []
    atrybuty = baza["atrybuty"]
    reguly = baza["reguly"]

    # Test 1 (gotowy): powtórzone identyfikatory reguł
    ids = [r["id"] for r in reguly]
    for id_ in sorted({i for i in ids if ids.count(i) > 1}):
        raport.append(("BŁĄD", f"Powtórzony identyfikator reguły {id_}"))

    # Test 2 (gotowy): niezadeklarowane atrybuty i nieznane operatory w warunkach
    for r in reguly:
        for atr, op, wart in r["jesli"]:
            if atr not in atrybuty:
                raport.append(("BŁĄD", f"{r['id']}: niezadeklarowany atrybut '{atr}'"))
            if op not in OPERATORY:
                raport.append(("BŁĄD", f"{r['id']}: nieznany operator '{op}'"))

    # Test 3: wartości spoza dziedziny (atrybuty z listą "wartosci", operatory == i !=)
    for r in reguly:
        for atr, op, wart in r["jesli"]:
            dziedzina = atrybuty.get(atr, {}).get("wartosci")
            if dziedzina is not None and op in ("==", "!=") and wart not in dziedzina:
                raport.append(("BŁĄD", f"{r['id']}: wartość {wart!r} spoza dziedziny '{atr}'"))

    # Test 4: nieużywane atrybuty wejściowe i nieużywane wartości atrybutów typu "wybor"
    uzyte_atrybuty = {atr for r in reguly for atr, _, _ in r["jesli"]}
    uzyte_wartosci = {(atr, wart) for r in reguly for atr, op, wart in r["jesli"] if op in ("==", "!=")}
    for nazwa, opis in atrybuty.items():
        if opis["typ"] == "wniosek":
            continue
        if nazwa not in uzyte_atrybuty:
            raport.append(("UWAGA", f"Atrybut '{nazwa}' nie jest używany w żadnej regule"))
        elif opis["typ"] == "wybor":
            for wart in opis["wartosci"]:
                if (nazwa, wart) not in uzyte_wartosci:
                    raport.append(("UWAGA", f"Wartość {wart!r} atrybutu '{nazwa}' nieużywana"))

    # Test 5: warunek niemożliwy do spełnienia (wniosek, którego żadna reguła nie ustala)
    wytwarzane = {tuple(r["to"]) for r in reguly}
    for r in reguly:
        for atr, op, wart in r["jesli"]:
            if atrybuty.get(atr, {}).get("typ") == "wniosek" and op == "==" and (atr, wart) not in wytwarzane:
                raport.append(("BŁĄD", f"{r['id']}: warunek {atr} == {wart!r} nigdy niespełniony"))

    # Test 6: pary reguł — duplikat, sprzeczność, pochłanianie
    for i, a in enumerate(reguly):
        for b in reguly[i + 1:]:
            wa = {tuple(w) for w in a["jesli"]}
            wb = {tuple(w) for w in b["jesli"]}
            ten_sam_wniosek = a["to"] == b["to"]
            if wa == wb and ten_sam_wniosek:
                raport.append(("UWAGA", f"{a['id']} i {b['id']} są identyczne (duplikat)"))
            elif wa == wb and a["to"][0] == b["to"][0]:
                raport.append(("BŁĄD", f"{a['id']} i {b['id']} są sprzeczne: te same warunki"))
            elif ten_sam_wniosek and wa < wb:
                raport.append(("UWAGA", f"{b['id']} jest pochłaniana przez {a['id']} (zbędna)"))
            elif ten_sam_wniosek and wb < wa:
                raport.append(("UWAGA", f"{a['id']} jest pochłaniana przez {b['id']} (zbędna)"))

    # Test 7: cel nieosiągalny
    if not any(r["to"][0] == baza["cel"] for r in reguly):
        raport.append(("BŁĄD", f"Cel '{baza['cel']}' nie jest wnioskiem żadnej reguły"))

    return raport


if __name__ == "__main__":
    plik = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent.parent / "wiedza" / "kawa.json"
    wyniki = weryfikuj(wczytaj_baze(plik))
    if not wyniki:
        print("Baza wiedzy nie zawiera wykrytych problemów.")
    for poziom, komunikat in wyniki:
        print(f"[{poziom}] {komunikat}")
    print(f"\nZnaleziono: {len(wyniki)} (w kawa_z_bledami.json jest 10 problemów do wykrycia)")
