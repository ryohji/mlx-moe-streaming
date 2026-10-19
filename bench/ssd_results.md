# SSD read bench (2048 MB file, 512 MB per cell)

- pread+NOCACHE: os.preadv with F_NOCACHE (page cache bypassed).
- mmap: mmap + madvise(WILLNEED) + touch one byte per page. **Not cache-bypassing**: the file was just written, so parts may already be in the page cache, and repeated passes over a 2 GB file can hit cache; treat mmap numbers as an upper bound. Page faults also hold the GIL, so mmap QD scaling is pessimistic.
- random offsets are 16KB-aligned; latency is per block read.

| mode | pattern | block | QD | MB/s | avg ms | p99 ms |
|---|---|---|---|---|---|---|
| pread+NOCACHE | random | 64KB | 1 | 394 | 0.16 | 0.68 |
| pread+NOCACHE | random | 64KB | 2 | 789 | 0.16 | 0.61 |
| pread+NOCACHE | random | 64KB | 4 | 925 | 0.27 | 0.84 |
| pread+NOCACHE | random | 64KB | 8 | 1193 | 0.42 | 1.06 |
| pread+NOCACHE | random | 256KB | 1 | 1035 | 0.24 | 0.84 |
| pread+NOCACHE | random | 256KB | 2 | 1842 | 0.27 | 0.78 |
| pread+NOCACHE | random | 256KB | 4 | 1918 | 0.52 | 1.08 |
| pread+NOCACHE | random | 256KB | 8 | 1899 | 1.05 | 1.62 |
| pread+NOCACHE | random | 1MB | 1 | 1938 | 0.51 | 0.99 |
| pread+NOCACHE | random | 1MB | 2 | 2859 | 0.70 | 0.93 |
| pread+NOCACHE | random | 1MB | 4 | 2849 | 1.40 | 1.71 |
| pread+NOCACHE | random | 1MB | 8 | 2730 | 2.90 | 3.70 |
| pread+NOCACHE | random | 4MB | 1 | 2633 | 1.51 | 2.24 |
| pread+NOCACHE | random | 4MB | 2 | 3264 | 2.43 | 2.91 |
| pread+NOCACHE | random | 4MB | 4 | 3227 | 4.87 | 5.90 |
| pread+NOCACHE | random | 4MB | 8 | 2208 | 14.03 | 22.20 |
| pread+NOCACHE | random | 8MB | 1 | 2558 | 3.11 | 4.13 |
| pread+NOCACHE | random | 8MB | 2 | 2904 | 5.45 | 6.55 |
| pread+NOCACHE | random | 8MB | 4 | 2367 | 13.19 | 17.06 |
| pread+NOCACHE | random | 8MB | 8 | 1586 | 38.97 | 58.00 |
| pread+NOCACHE | sequential | 64KB | 1 | 293 | 0.21 | 1.25 |
| pread+NOCACHE | sequential | 64KB | 2 | 676 | 0.18 | 0.83 |
| pread+NOCACHE | sequential | 64KB | 4 | 1652 | 0.15 | 0.55 |
| pread+NOCACHE | sequential | 64KB | 8 | 1855 | 0.27 | 0.68 |
| pread+NOCACHE | sequential | 256KB | 1 | 1250 | 0.20 | 0.75 |
| pread+NOCACHE | sequential | 256KB | 2 | 2069 | 0.24 | 0.39 |
| pread+NOCACHE | sequential | 256KB | 4 | 2522 | 0.40 | 0.80 |
| pread+NOCACHE | sequential | 256KB | 8 | 3140 | 0.63 | 0.81 |
| pread+NOCACHE | sequential | 1MB | 1 | 1791 | 0.56 | 1.13 |
| pread+NOCACHE | sequential | 1MB | 2 | 3166 | 0.63 | 0.95 |
| pread+NOCACHE | sequential | 1MB | 4 | 3195 | 1.24 | 1.93 |
| pread+NOCACHE | sequential | 1MB | 8 | 3081 | 2.57 | 3.21 |
| pread+NOCACHE | sequential | 4MB | 1 | 2204 | 1.80 | 9.52 |
| pread+NOCACHE | sequential | 4MB | 2 | 2671 | 2.96 | 5.45 |
| pread+NOCACHE | sequential | 4MB | 4 | 3225 | 4.85 | 6.09 |
| pread+NOCACHE | sequential | 4MB | 8 | 1995 | 15.64 | 20.51 |
| pread+NOCACHE | sequential | 8MB | 1 | 2754 | 2.87 | 3.94 |
| pread+NOCACHE | sequential | 8MB | 2 | 3079 | 5.11 | 5.58 |
| pread+NOCACHE | sequential | 8MB | 4 | 2779 | 11.20 | 15.45 |
| pread+NOCACHE | sequential | 8MB | 8 | 1971 | 30.77 | 37.18 |
| mmap | random | 64KB | 1 | 188 | 0.33 | 0.89 |
| mmap | random | 64KB | 2 | 45463 | 0.00 | 0.00 |
| mmap | random | 64KB | 4 | 47877 | 0.00 | 0.00 |
| mmap | random | 64KB | 8 | 48444 | 0.00 | 0.00 |
| mmap | random | 256KB | 1 | 680 | 0.37 | 1.14 |
| mmap | random | 256KB | 2 | 61600 | 0.00 | 0.01 |
| mmap | random | 256KB | 4 | 73155 | 0.00 | 0.00 |
| mmap | random | 256KB | 8 | 78057 | 0.00 | 0.00 |
| mmap | random | 1MB | 1 | 1027 | 0.97 | 2.17 |
| mmap | random | 1MB | 2 | 77332 | 0.01 | 0.02 |
| mmap | random | 1MB | 4 | 86824 | 0.01 | 0.02 |
| mmap | random | 1MB | 8 | 96289 | 0.01 | 0.01 |
| mmap | random | 4MB | 1 | 1157 | 3.45 | 8.77 |
| mmap | random | 4MB | 2 | 10450 | 0.70 | 7.44 |
| mmap | random | 4MB | 4 | 98773 | 0.04 | 0.06 |
| mmap | random | 4MB | 8 | 96012 | 0.04 | 0.04 |
| mmap | random | 8MB | 1 | 1056 | 7.57 | 40.71 |
| mmap | random | 8MB | 2 | 6193 | 2.37 | 16.00 |
| mmap | random | 8MB | 4 | 99818 | 0.08 | 0.11 |
| mmap | random | 8MB | 8 | 101550 | 0.07 | 0.11 |
| mmap | sequential | 64KB | 1 | 476 | 0.13 | 0.86 |
| mmap | sequential | 64KB | 2 | 10688 | 0.01 | 0.15 |
| mmap | sequential | 64KB | 4 | 49702 | 0.00 | 0.00 |
| mmap | sequential | 64KB | 8 | 48254 | 0.00 | 0.00 |
| mmap | sequential | 256KB | 1 | 81785 | 0.00 | 0.00 |
| mmap | sequential | 256KB | 2 | 81645 | 0.00 | 0.00 |
| mmap | sequential | 256KB | 4 | 74159 | 0.00 | 0.01 |
| mmap | sequential | 256KB | 8 | 74546 | 0.00 | 0.00 |
| mmap | sequential | 1MB | 1 | 98516 | 0.01 | 0.01 |
| mmap | sequential | 1MB | 2 | 101469 | 0.01 | 0.01 |
| mmap | sequential | 1MB | 4 | 97582 | 0.01 | 0.01 |
| mmap | sequential | 1MB | 8 | 94457 | 0.01 | 0.01 |
| mmap | sequential | 4MB | 1 | 105926 | 0.04 | 0.04 |
| mmap | sequential | 4MB | 2 | 105705 | 0.04 | 0.04 |
| mmap | sequential | 4MB | 4 | 104470 | 0.04 | 0.04 |
| mmap | sequential | 4MB | 8 | 95937 | 0.04 | 0.06 |
| mmap | sequential | 8MB | 1 | 101029 | 0.08 | 0.16 |
| mmap | sequential | 8MB | 2 | 107188 | 0.07 | 0.08 |
| mmap | sequential | 8MB | 4 | 102375 | 0.07 | 0.11 |
| mmap | sequential | 8MB | 8 | 89640 | 0.07 | 0.09 |
