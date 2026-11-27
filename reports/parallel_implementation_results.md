# Wyniki wariantu rownoleglego P=2

**Uwaga:** raporty syntezy, implementacji i timingu ponizej dotycza wersji
sprzed zmiany requantyzacji opisanej w sekcji MNIST. Nie sa potwierdzeniem
timingu ani zasobow aktualnego RTL; trzeba je ponownie wygenerowac w Vivado.

## Zakres

Wariant P=2 liczy dwa wyjscia warstwy jednoczesnie. Wyniki nadal sa emitowane
po jednym bajcie przez interfejs valid/ready. Pary wag sa zapisane w
16-bitowej synchronicznej pamieci ROM.

## Zgodnosc i liczba taktow

Testy MyHDL porownuja wszystkie 10 bajtow wyjscia z referencja Python dla
wejsc zerowych, maksymalnych i losowych. Sprawdzaja tez backpressure.

| Wariant | Taktow obliczen i emisji | Taktow ladowania | Razem |
|---|---:|---:|---:|
| P=1 | 2600 | 256 | 2856 |
| P=2 | 1305 | 256 | 1561 |

Pomiar nie obejmuje resetu ani dodatkowych taktow wynikajacych z backpressure.

## Wyniki syntezy Vivado

Cel: urzadzenie wskazane w planie projektu, `xc7z020clg400-1`.

Oba warianty syntezy wykonano w Vivado 2026.2 dla tego samego ukladu.

| Wariant | LUT | FF | DSP48E1 | BRAM |
|---|---:|---:|---:|---|
| P=1 | 373 | 64 | 6 | 3 RAMB18E1 |
| P=2 | 683 | 105 | 12 | 1 RAMB36E1 + 1 RAMB18E1 |

Raport P=1: [vivado_utilization_7020.rpt](./vivado_utilization_7020.rpt).

Raport P=2: [vivado_utilization_parallel2.rpt](./vivado_utilization_parallel2.rpt).

Raport P=2 potwierdza, ze ROM wag jest zaimplementowany w RAMB36E1, a bufor
wejsciowy w RAMB18E1. Synteza zakonczyla sie bez bledow.

## Implementacja i analiza timingowa Vivado 2026.2

Oba warianty zaimplementowano dla `xc7z020clg400-1` w Vivado 2026.2. Projekt
nie narzuca czestotliwosci zegara. Wybor wykonano na podstawie sweepu okresow
10, 29, 30, 35 i 40 ns. Kryterium akceptacji to jednoczesnie WNS >= 5 ns i
TNS = 0 dla obu wariantow; wybrano najwyzsza czestotliwosc spelniajaca te
warunki. Jest nia 25 MHz (40 ns). Punkt 100 MHz jest porownaniem, a nie
wymaganiem projektu.

| Okres | Czestotliwosc | P=1 WNS | P=1 TNS | P=2 WNS | P=2 TNS | Kryterium |
|---:|---:|---:|---:|---:|---:|:---:|
| 10 ns | 100 MHz | -18.582 ns | -550.049 ns | -18.326 ns | -1083.457 ns | FAIL |
| 29 ns | 34.483 MHz | +0.034 ns | 0.000 ns | -0.300 ns | -1.918 ns | FAIL |
| 30 ns | 33.333 MHz | +0.609 ns | 0.000 ns | +0.074 ns | 0.000 ns | FAIL |
| 35 ns | 28.571 MHz | +2.327 ns | 0.000 ns | +3.029 ns | 0.000 ns | FAIL |
| 40 ns | 25 MHz | +6.415 ns | 0.000 ns | +7.537 ns | 0.000 ns | PASS |

Okres 40 ns odpowiada najwyzszej przebadanej czestotliwosci, ktora spelnia
przyjete kryterium dla obu wariantow. Przy 35 ns oba maja TNS = 0, ale ich
WNS (2.327 ns i 3.029 ns) jest mniejszy od wymaganego marginesu 5 ns. Przy
30 ns WNS wynosi 0.609 ns i 0.074 ns; przy 29 ns P=2 ma ujemne WNS i TNS.
Nie wyznaczano dokladnej granicy Fmax. Przy integracji nalezy dostarczyc
blokowi zegar o okresie co najmniej 40 ns.

Szacowany throughput przy wybranym okresie 40 ns to ok. 8,754 obrazow/s dla
P=1 i 16,015 obrazow/s dla P=2, na podstawie odpowiednio 2,856 i 1,561
taktow na obraz. To szacunki dla bloku, nie pomiar na plytce.

Zasoby po trasowaniu przy wybranym okresie 40 ns:

| Wariant | LUT | FF | DSP48E1 | BRAM |
|---|---:|---:|---:|---|
| P=1 | 354 | 64 | 6 | 3 RAMB18E1 |
| P=2 | 652 | 105 | 12 | 1 RAMB36E1 + 1 RAMB18E1 |

Raporty dla wybranego okresu 40 ns:

- P=1: [timing](./vivado_timing_serial_40ns.rpt),
  [critical paths](./vivado_timing_serial_40ns_paths.rpt),
  [utilization](./vivado_timing_serial_40ns_utilization.rpt).
- P=2: [timing](./vivado_timing_parallel2_40ns.rpt),
  [critical paths](./vivado_timing_parallel2_40ns_paths.rpt),
  [utilization](./vivado_timing_parallel2_40ns_utilization.rpt).

Raporty dla porownawczego okresu 10 ns / 100 MHz (oba warianty nie spelniaja
timingu):

- P=1: [timing](./vivado_timing_serial_10ns.rpt),
  [critical paths](./vivado_timing_serial_10ns_paths.rpt).
- P=2: [timing](./vivado_timing_parallel2_10ns.rpt),
  [critical paths](./vivado_timing_parallel2_10ns_paths.rpt).

Raporty dla porownawczego okresu 30 ns:

- P=1: [timing](./vivado_timing_serial_30ns.rpt),
  [critical paths](./vivado_timing_serial_30ns_paths.rpt),
  [utilization](./vivado_timing_serial_30ns_utilization.rpt).
- P=2: [timing](./vivado_timing_parallel2_30ns.rpt),
  [critical paths](./vivado_timing_parallel2_30ns_paths.rpt),
  [utilization](./vivado_timing_parallel2_30ns_utilization.rpt).

Raporty sweepu: [P=1 przy 10 ns](./vivado_timing_serial_10ns.rpt),
[P=2 przy 10 ns](./vivado_timing_parallel2_10ns.rpt),
[P=1 przy 29 ns](./vivado_timing_serial_29ns.rpt),
[P=2 przy 29 ns](./vivado_timing_parallel2_29ns.rpt).

## Aktualna walidacja RTL

- `python -m unittest -v tests.test_fc_rtl`: 2 testy PASS dla P=1 i P=2.
- Wspolsymulacje aktualnego RTL wykonano na Icarus Verilog 12.0 z MyHDL
  0.11.52. Kazdy wariant sprawdzono dla wejscia zerowego, maksymalnego i
  deterministycznego losowego; wszystkie bajty wyjscia zgadzaja sie bit w bit
  z referencja Python. Testy obejmuja tez backpressure na wyjsciu.
- Synteza i timing aktualnego RTL nie zostaly jeszcze wykonane. Raporty
  Vivado opisane wyzej pochodza sprzed zmiany requantyzacji i nie sa aktualnym
  potwierdzeniem zasobow ani czestotliwosci.

## Test na zbiorze MNIST

Ewaluator:

```bash
python scripts/evaluate_mnist.py --myhdl-samples 10 \
  --json-out reports/mnist_test_set_results.json
```

Wykorzystano oficjalny podzial testowy MNIST `t10k`: 10 000 obrazow 28x28.
Skrypt pobiera dane w razie potrzeby i sprawdza MD5 plikow. Model float
otrzymuje piksele podzielone przez 255, a model int8 piksele 0-255 (skala 1,
zero point 0).

| Ewaluowany model | Dokladnosc top-1 |
|---|---:|
| Pelny model float ONNX | 98.90% |
| Pelny model int8 ONNX | 98.92% |
| Referencja Python warstwy 256->10 na cechach int8 | 98.92% |

Predykcje referencji Python i pelnego modelu int8 sa zgodne dla wszystkich
10 000 obrazow, a wyjscia bajtowe sa zgodne dla 10 000/10 000 wektorow
(100 000/100 000 bajtow). Wczesniejsza rozbieznosc dotyczyla obrazu o indeksie
4951, kanalu 6: iloczyn w `QLinearMatMul` jest blisko polowy miedzy liczbami.
Requantyzacja sprzetowa wykonywala zaokraglenie fixed-point bez posredniego
zaokraglenia iloczynu do float32, przez co wynik wynosil 77 zamiast 76. Obecna
wersja odtwarza zaokraglenie iloczynu float32 za pomoca arytmetyki calkowitej.

Symulacje cyklowe MyHDL dla P=1 i P=2 wykonano na 11 rzeczywistych wektorach:
dziesieciu rownomiernie rozmieszczonych probkach oraz probce 4951, wykrytej
jako przypadek graniczny float32. Oba warianty zgadzaja sie bajt w bajt z
referencja Python i ONNX Runtime na 11/11 wektorach; poprawnie klasyfikuja
11/11 probek. Nie jest to pomiar dokladnosci sprzetowej na calym zbiorze ani
pelny test RTL na 10 000 obrazach. Szczegolowe wyniki, sumy kontrolne i
hash'e modeli sa w [mnist_test_set_results.json](./mnist_test_set_results.json).

## Pozostale walidacje

- Historyczny sweep sprzed zmiany arytmetyki wskazywal okres 40 ns jako
  spelniajacy przyjete kryterium WNS >= 5 ns i TNS = 0. Nie jest to aktualne
  potwierdzenie timingu nowego RTL; wymagane jest ponowne uruchomienie Vivado.
- Raporty nie zawieraja opoznien wejscia/wyjscia, wiec nie potwierdzaja
  zamkniecia timingowego na poziomie calej plytki lub systemu.
