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
- Testy ko-symulacji aktualnego RTL dla P=1 i P=2 przechodzą w Icarus Verilog
  12.0 z MyHDL 0.11.52. Dla wejścia zerowego, maksymalnego i losowego wszystkie
  bajty wyjścia zgadzają się bit w bit z wzorcem Pythonowym; sprawdzono też
  zatrzymanie odbioru wyników.
- Rejestr po requantyzacji wykorzystuje istniejący stan `REQUANTIZE`, więc
  liczba cykli na obraz się nie zmieniła.
- Pierwszy raport Vivado pokazał, że pierwotny ROM wag był realizowany w LUT;
  sama wskazówka `rom_style` nie zmieniła mapowania.
- ROM wag rozdzielono na dwa banki i zmieniono ich inicjalizację na pliki
  `$readmemh`. Ponowna synteza Vivado potwierdziła dwa banki wag w BRAM oraz
  osobną BRAM dla bufora wejściowego: 3 x RAMB18E1 łącznie.
- Syntezę P=1/P=2 i sweep routingu/timingu 10, 29, 30, 35, 40 i 45 ns
  wykonano w Vivado 2026.2 dla aktualnego RTL.
- Przy 45 ns oba warianty spełniają podstawowe kryterium zamknięcia timingu
  WNS >= 0 ns i TNS = 0: P=1 ma WNS 1.108 ns, a P=2 ma WNS 0.596 ns.
  Przy 40 ns (25 MHz) timing nadal nie jest zamknięty. Zapas 5 ns nie jest
  wymagany, ale też nie został osiągnięty.
- Rejestr po requantyzacji poprawił WNS przy 40 ns o około 13 ns względem
  poprzedniej wersji, bez zmiany liczby cykli.

Następny krok:

1. Sprawdzić wymagania prowadzącego dotyczące docelowego zegara, zapasu
   timingowego, RTL testbenchu i wariantu równoległego.
2. Przenieść aktualne źródła RTL i pliki `.mem` do projektu kursowego Vivado.
3. Jeśli wymagany zegar lub zapas jest większy niż obecny wynik, optymalizować
   ścieżkę z akumulatora do rejestru pośredniego requantyzacji i ponowić timing.
