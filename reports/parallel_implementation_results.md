# Wyniki wariantu rownoleglego P=2

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

Cel: `xc7z020clg400-1`.

| Wariant | LUT | FF | DSP48E1 | BRAM |
|---|---:|---:|---:|---|
| P=1 | 374 | 64 | 6 | 3 RAMB18E1 |
| P=2 | 683 | 105 | 12 | 1 RAMB36E1 + 1 RAMB18E1 |

Raport P=1: [vivado_utilization_7020.rpt](./vivado_utilization_7020.rpt).  
Raport P=2: [vivado_utilization_parallel2.rpt](./vivado_utilization_parallel2.rpt).

Raport P=2 potwierdza, ze ROM wag jest zaimplementowany w RAMB36E1, a bufor
wejsciowy w RAMB18E1. Synteza zakonczyla sie bez bledow.

## Pozostale walidacje

- Wspolsymulacja wygenerowanego RTL P=2 z Icarus Verilog pozostaje do
  uruchomienia. W ostatnim uruchomieniu dwa testy RTL zostaly pominiete, bo
  srodowisko testowe nie mialo Icarus Verilog i MyHDL VPI.
- Nie mierzono maksymalnej czestotliwosci ani throughput w obrazach na sekunde;
  wymagaja constraints zegara i implementacji z analiza timingowa.
