# Wyniki syntezy i timingu

Raporty wygenerowano 27 listopada 2026 w Vivado 2026.2 dla aktualnego RTL,
po zmianie requantyzacji zgodnej z zaokragleniem float32 i dodaniu rejestru
posredniego. Rejestr wykorzystuje istniejacy stan `REQUANTIZE`, wiec liczba
cykli sie nie zmienila. Cel syntezy i implementacji: `xc7z020clg400-1`.

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

Oba warianty zsyntetyzowano dla tego samego ukladu.

| Wariant | LUT | FF | DSP48E1 | BRAM |
|---|---:|---:|---:|---|
| P=1 | 1163 | 72 | 5 | 3 RAMB18E1 |
| P=2 | 2299 | 121 | 10 | 1 RAMB36E1 + 1 RAMB18E1 |

Raport P=1: [vivado_utilization_7020.rpt](./vivado_utilization_7020.rpt).

Raport P=2: [vivado_utilization_parallel2.rpt](./vivado_utilization_parallel2.rpt).

Synteza potwierdza mapowanie pamieci wag i bufora wejsciowego do BRAM.

## Implementacja i analiza timingowa Vivado 2026.2

Oba warianty zaimplementowano dla `xc7z020clg400-1`. Wykonano sweep okresow
10, 29, 30, 35, 40 i 45 ns. Podstawowe kryterium zamkniecia timingu to
WNS >= 0 ns i TNS = 0 ns; dodatkowy zapas 5 ns nie jest wymagany.
**Oba warianty spelniaja podstawowe kryterium przy 45 ns (22.222 MHz)**.
Przy 40 ns (25 MHz) timing nadal nie jest zamkniety. Przy 45 ns zapas wynosi
1.108 ns dla P=1 i 0.596 ns dla P=2, czyli nie ma dodatkowego zapasu 5 ns.

| Okres | Czestotliwosc | P=1 WNS | P=1 TNS | P=2 WNS | P=2 TNS | Kryterium |
|---:|---:|---:|---:|---:|---:|:---:|
| 10 ns | 100 MHz | -31.209 ns | -706.390 ns | -31.213 ns | -1375.367 ns | FAIL |
| 29 ns | 34.483 MHz | -11.840 ns | -184.556 ns | -11.901 ns | -362.845 ns | FAIL |
| 30 ns | 33.333 MHz | -10.745 ns | -166.336 ns | -10.942 ns | -337.749 ns | FAIL |
| 35 ns | 28.571 MHz | -5.977 ns | -92.388 ns | -6.088 ns | -182.393 ns | FAIL |
| 40 ns | 25 MHz | -1.724 ns | -22.266 ns | -2.297 ns | -56.001 ns | FAIL |
| 45 ns | 22.222 MHz | 1.108 ns | 0.000 ns | 0.596 ns | 0.000 ns | PASS |

Przy 45 ns krytyczna sciezka prowadzi z akumulatora do rejestru posredniego
requantyzacji. Opoznienie sciezki wynosi 43.820 ns dla P=1 (62 poziomy logiki)
i 44.403 ns dla P=2 (61 poziomow). Rejestr posredni poprawil WNS przy 40 ns
o ok. 13 ns wzgledem poprzedniej wersji, bez dodatkowych cykli.

Zasoby po trasowaniu przy okresie 45 ns:

| Wariant | LUT | FF | DSP48E1 | BRAM |
|---|---:|---:|---:|---|
| P=1 | 1072 | 72 | 5 | 3 RAMB18E1 |
| P=2 | 2164 | 121 | 10 | 1 RAMB36E1 + 1 RAMB18E1 |

Przy okresie 45 ns teoretyczna przepustowosc, bez resetu i backpressure, wynosi
ok. 7 781 obrazow/s dla P=1 (2856 taktow na obraz) i 14 236 obrazow/s dla P=2
(1561 taktow na obraz). To wyliczenie dotyczy samego bloku RTL, nie calej
plytki; raporty nie uwzgledniaja opoznien wejscia/wyjscia.

Raporty syntezy:

- P=1: [utilization](./vivado_utilization_7020.rpt).
- P=2: [utilization](./vivado_utilization_parallel2.rpt).

Raporty timingu, sciezek krytycznych i zasobow po trasowaniu dla calego
sweepu:

| Okres | P=1 timing | P=1 sciezki | P=1 zasoby | P=2 timing | P=2 sciezki | P=2 zasoby |
|---:|---|---|---|---|---|---|
| 10 ns | [rpt](./vivado_timing_serial_10ns.rpt) | [rpt](./vivado_timing_serial_10ns_paths.rpt) | [rpt](./vivado_timing_serial_10ns_utilization.rpt) | [rpt](./vivado_timing_parallel2_10ns.rpt) | [rpt](./vivado_timing_parallel2_10ns_paths.rpt) | [rpt](./vivado_timing_parallel2_10ns_utilization.rpt) |
| 29 ns | [rpt](./vivado_timing_serial_29ns.rpt) | [rpt](./vivado_timing_serial_29ns_paths.rpt) | [rpt](./vivado_timing_serial_29ns_utilization.rpt) | [rpt](./vivado_timing_parallel2_29ns.rpt) | [rpt](./vivado_timing_parallel2_29ns_paths.rpt) | [rpt](./vivado_timing_parallel2_29ns_utilization.rpt) |
| 30 ns | [rpt](./vivado_timing_serial_30ns.rpt) | [rpt](./vivado_timing_serial_30ns_paths.rpt) | [rpt](./vivado_timing_serial_30ns_utilization.rpt) | [rpt](./vivado_timing_parallel2_30ns.rpt) | [rpt](./vivado_timing_parallel2_30ns_paths.rpt) | [rpt](./vivado_timing_parallel2_30ns_utilization.rpt) |
| 35 ns | [rpt](./vivado_timing_serial_35ns.rpt) | [rpt](./vivado_timing_serial_35ns_paths.rpt) | [rpt](./vivado_timing_serial_35ns_utilization.rpt) | [rpt](./vivado_timing_parallel2_35ns.rpt) | [rpt](./vivado_timing_parallel2_35ns_paths.rpt) | [rpt](./vivado_timing_parallel2_35ns_utilization.rpt) |
| 40 ns | [rpt](./vivado_timing_serial_40ns.rpt) | [rpt](./vivado_timing_serial_40ns_paths.rpt) | [rpt](./vivado_timing_serial_40ns_utilization.rpt) | [rpt](./vivado_timing_parallel2_40ns.rpt) | [rpt](./vivado_timing_parallel2_40ns_paths.rpt) | [rpt](./vivado_timing_parallel2_40ns_utilization.rpt) |
| 45 ns | [rpt](./vivado_timing_serial_45ns.rpt) | [rpt](./vivado_timing_serial_45ns_paths.rpt) | [rpt](./vivado_timing_serial_45ns_utilization.rpt) | [rpt](./vivado_timing_parallel2_45ns.rpt) | [rpt](./vivado_timing_parallel2_45ns_paths.rpt) | [rpt](./vivado_timing_parallel2_45ns_utilization.rpt) |

## Aktualna walidacja RTL

- `python -m unittest -v tests.test_fc_rtl`: 2 testy PASS dla P=1 i P=2.
- Wspolsymulacje aktualnego RTL wykonano na Icarus Verilog 12.0 z MyHDL
  0.11.52. Kazdy wariant sprawdzono dla wejscia zerowego, maksymalnego i
  deterministycznego losowego; wszystkie bajty wyjscia zgadzaja sie bit w bit
  z referencja Python. Testy obejmuja tez backpressure na wyjsciu.
- Aktualna synteza P=1/P=2 i pelny sweep timingu zostaly wykonane w Vivado
  2026.2. Oba warianty spelniaja podstawowe kryterium WNS >= 0 ns i TNS = 0
  przy 45 ns; dodatkowy zapas 5 ns nie zostal osiagniety i nie jest wymagany.

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

- Raporty nie zawieraja opoznien wejscia/wyjscia, wiec nie potwierdzaja
  zamkniecia timingowego na poziomie calej plytki lub systemu.
