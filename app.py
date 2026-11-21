"""Interfejs systemu ekspertowego (spotkanie 3). Uruchom: python -m streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "se"))
from silnik import coto, wczytaj_baze, wnioskuj_w_przod, wyjasnij_jak  # noqa: E402
from weryfikacja import weryfikuj  # noqa: E402

PLIK_BAZY = Path(__file__).parent / "wiedza" / "moja_dziedzina.json"
baza = wczytaj_baze(PLIK_BAZY)

st.set_page_config(page_title=baza["nazwa"])
st.title(baza["nazwa"])

with st.sidebar:
    st.header("Weryfikacja bazy wiedzy")
    raport = weryfikuj(baza)
    if not raport:
        st.success("Nie wykryto problemów.")
    for poziom, komunikat in raport:
        (st.error if poziom == "BŁĄD" else st.warning)(komunikat)

with st.form("konsultacja"):
    fakty = {}
    for nazwa, opis in baza["atrybuty"].items():
        if opis["typ"] == "liczba":
            lo, hi = opis.get("zakres", [-100, 100])
            # krok 0,1 — żeby dało się ustawić wartości progowe, np. 10,5 V albo 12,1 V
            wartosc = st.slider(opis["pytanie"], float(lo), float(hi), float(lo + hi) / 2, step=0.1)
            fakty[nazwa] = round(wartosc, 1)
        elif opis["typ"] == "wybor":
            fakty[nazwa] = st.radio(opis["pytanie"], opis["wartosci"], horizontal=True)
    wyslano = st.form_submit_button("Doradź")

if wyslano:
    wynik, slad = wnioskuj_w_przod(baza, fakty)
    cel = wynik.get(baza["cel"])
    if cel is None:
        st.error("Brak decyzji: żadna reguła w bazie wiedzy nie obejmuje tej kombinacji danych.")
    else:
        st.success(f"Polecam: {cel}")
        opis_coto = coto(baza, slad)
        if opis_coto:
            st.info(f"Co to jest? {opis_coto}")
    with st.expander("Jak do tego doszedłem?"):
        if not slad:
            st.write("Żadna reguła nie odpaliła.")
        for krok in wyjasnij_jak(baza, slad):
            st.text(krok)
