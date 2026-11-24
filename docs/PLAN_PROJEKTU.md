# Plan projektu ESL 2

## Cel

Zaprojektować i przeanalizować ostatnią warstwę klasyfikatora MNIST o wymiarze
256 -> 10. Warstwa przyjmuje 256 aktywacji uint8 z poprzednich operacji modelu,
współczynnikami wag są liczby int8, a wynikowa część jest int8/uint8 po
kwantyzacji.

W pierwszej wersji używamy:

- jednej jednostki MAC,
- 32-bitowego akumulatora,
- potoku po wejściach z II = 1,
- 2560 operacji MAC na obraz,
- wag zapisywanych w pamięci BRAM po syntezie.

## Etap 1: Pobranie i analiza modelu

- Pobrać model `mnist-12-int8.onnx`.
- Sprawdzić graf ONNX i znaleźć warstwę `256 -> 10`.
- Odczytać:
  - wagi i bias,
  - zero pointy wejścia i wag,
  - skale wejścia i wag,
  - parametr requantisation,
  - zero point i skalę wyniku.
- Potwierdzić prowadzącemu, że dokładnie ten model jest dozwolony.

## Etap 2: Wzorzec programowy

- Zaimplementować liczenie warstwy w Pythonie.
- Odwzorować kolejność operacji ONNX:
  - QLinearMatMul,
  - akumulacja int32,
  - requantisation,
  - bias przez QLinearAdd,
  - końcowa requantisation uint8.
- Porównać wynik z ONNX Runtime.
- Zapisać testy dla:
  - wejścia zerowego,
  - wejścia pełnego,
  - wejść losowych,
  - błędnego kształtu wejścia.
- Wymagany wynik: zgodność bit w bit.

## Etap 3: Implementacja MyHDL - wersja podstawowa

- Przyjąć 256 wejść uint8.
- Przygotować:
  - FSM `LOAD`,
  - FSM `MAC`,
  - fazę requantisation,
  - fazę dodania biasu,
  - stan przekazywania wyniku.
- Ładować wejścia do pamięci wejściowej.
- Przechowywać wagi w pamięci 2560 elementów.
- Wykonywać jedna operację MAC na takt po rozgrzaniu potoku.
- Użyć akumulatora 32-bitowego.
- Zaimplementować stałoprzecinkowe mnożniki requantisation.
- Zadbać o zgodne zaokrąglanie i saturację do zakresu uint8.

## Etap 4: Testy MyHDL i generowanie RTL

- Symulować implementację MyHDL.
- Porównać 10 wyników z wzorcem Pythonowym.
- Przeprowadzić testy dla wejść losowych i wartości skrajnych.
- Testowo sprawdzić requantisation w całym możliwym zakresie akumulatora.
- Wygenerować Verilog przez:

  ```bash
  source .venv/bin/activate
  python -m mnist_fc.convert_fc
  ```

- Wygenerowany RTL zapisuje się w `build/mnist_fc_serial.v`.

## Etap 5: Test RTL

- Zainstalować dostępny symulator RTL albo skorzystać z narzędzia uczelnianego.
- Napisać testbench podający reset i 256 wejść.
- Wykonywać wynik dla obrazów testowych.
- Porównać każdy z 10 wyników z wzorcem ONNX.
- Nie wystarcza porównanie tylko rozpoznanej cyfry; porównujemy bajty wyjść.

## Etap 6: Synteza w Vivado

- Wybierać dokładny układ Zynq-7000 wymagany na zajęciach.
- Dodać projekt Vivado z wygenerowanym Verilogiem.
- Uruchomić syntezę wersji podstawowej.
- Zapisać:
  - liczbę LUT,
  - liczbę FF,
  - liczbę DSP,
  - liczbę BRAM,
  - częstotliwość zegara,
  - liczba taktów na obraz.
- Potwierdzić, że tablica wag została zmapowana do BRAM.
- Sprawdzić, czy układ pomieści się w docelowym Zynq-7000.

## Etap 7: Wersja równoległa

Wybrać jeden wariant zgodny z wymaganiami:

- P jednostek MAC rozwiniętych po wyjściach, albo
- rozwinięcie po wejściach z drzewem sumatorów.

Dla każdej liczby MAC-ów zmierzyć:

- liczbę taktów na obraz,
- throughput,
- LUT,
- DSP,
- BRAM,
- zużycie rejestrów.

## Etap 8: Porównanie

Przygotować tabelę:

| Wersja | MAC | II | Taktów/obraz | LUT | DSP | BRAM | Zgodność |
|---|---:|---:|---:|---:|---:|---:|---|
| Pojedynczy MAC | 1 | 1 | 2600 obliczeń + 256 wejść | TBD | TBD | TBD | PASS |

Dodać wykresy lub krótki rozdział opisujący:

- czasowościową zajętość,
- koszt sprzętowy,
- wpływ równoległości,
- kompromis szybkość - zasoby,
- dokładność bitową.

## Etap 9: Dokumentacja i prezentacja

- Opisać cel i architekturę.
- Włożyć schemat blokowy.
- Dodać schemat FSM.
- Włożyć fragmenty generowanego Veriloga.
- Pokazać wyniki symulacji.
- Pokazać raporty syntezy Vivado.
- Dodać tabelę porównawczą.
- Opisać problemy i sposób ich rozwiązania.
- Przygotować krótką prezentację wyników.

## Aktualny stan

- Model ONNX pobrany i przeanalizowany.
- Zidentyfikowano warstwę 256 -> 10.
- Przygotowano wzorzec Pythonowy.
- Przygotowano implementację MyHDL z jednym MAC-em.
- Przygotowano testy ONNX i MyHDL.
- Wygenerowano Verilog.
- Wszystkie obecne testy Python/MyHDL przechodzą.
- Dodano testbench MyHDL i sprawdzono wygenerowany RTL w symulatorze Icarus.
- RTL zgadza się bit w bit z Pythonowym wzorcem dla wejścia zerowego,
  maksymalnego i losowego; sprawdzono też zatrzymanie odbioru wyników.
- Pierwszy raport Vivado pokazał, że pierwotny ROM wag był realizowany w LUT;
  sama wskazówka `rom_style` nie zmieniła mapowania.
- ROM wag rozdzielono na dwa banki i zmieniono ich inicjalizację na pliki
  `$readmemh`. Ponowna synteza Vivado potwierdziła dwa banki wag w BRAM oraz
  osobną BRAM dla bufora wejściowego: 3 x RAMB18E1 łącznie.
- Aktualny raport syntezy: 374 LUT, 64 FF, 6 DSP48E1 i 3 RAMB18E1 dla
  `xc7z020clg400-1`. Test RTL w Icarus nadal zgadza się bit w bit z wzorcem.

Następny krok:

1. Sprawdzić wymagania prowadzącego dotyczące RTL testbenchu i wariantu równoległego.
2. Przenieść wygenerowany Verilog i oba pliki `.mem` do projektu Vivado.
3. Uruchomić syntezę w projekcie kursowym i zachować jego raport.
4. Dodać constraints zegara i uruchomić implementację, jeśli wymagane są wyniki
   czasowe.
