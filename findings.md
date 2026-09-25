# Lab 1 — measurements and findings

Source material for the part-four report. Every number here was measured on the
programs as delivered, on an otherwise idle machine, and re-measured from
scratch after an earlier session turned out to have been sharing the CPU with
other work.

## 1. Summary

On 2,000,000 proteins the multiprocessing version is **5.1x** faster than the
serial one and the threaded version **3.8x**. All three print identical
results. The ceiling is set by the ~2 s that is never parallel — the CSV read
and the final label pass — which by Amdahl's law caps the speedup at **8.9x**
on 16 workers and **18.7x** on infinitely many.

| Version         | Time (s) | Speedup | Efficiency (S/16) |
| --------------- | -------- | ------- | ----------------- |
| serial          | 37.35    | 1.00    | —                 |
| multiprocessing |  7.37    | 5.07    | 32%               |
| threads         |  9.84    | 3.80    | 24%               |

## 2. Test setup and method

| Item        | Value                                                            |
| ----------- | ---------------------------------------------------------------- |
| CPU         | Intel Core i5-13500H: 12 cores (4 P-cores with SMT, 8 E-cores), 16 logical |
| Software    | Linux, Python 3.14.7, NumPy 2.5.3, pandas 3.0.6, matplotlib 3.11.2 |
| Dataset     | `proteins.csv`, 2,000,000 rows (280 MB), `python proteins-generator.py 2000000 42` |
| Seed        | 42 in all three programs, so every run clusters from the same initial centroids |
| Timing      | `time.perf_counter()` from before the CSV read to after the results are computed; figures are built after the clock stops, as the lab requires |
| Repetitions | Headline: median of 3 interleaved runs after a discarded warm-up run. Sweeps: median of 2 runs per point |

Two precautions are worth stating in the report, because both changed the
numbers:

- **Nothing else ran during the measurements.** An earlier set was taken while
  other copies of the programs were running, and its sweep values were
  unusable.
- **Every sweep carries its own serial reference,** measured inside the same
  block. Under sustained load the CPU settles into a lower clock: the same
  program takes 35.05 s on a cold machine and 37.35 s in steady state, a 6%
  drift. Speedups are unaffected — the cold block gives 5.19x and 3.77x against
  the warm block's 5.07x and 3.80x, within run-to-run noise — but absolute
  times may only be compared inside a block.

The machine is also **heterogeneous**: 4 fast performance cores (8 hardware
threads) and 8 slower efficiency cores. That drives the result in section 5.

## 3. What was parallelized, and how

Both parallel versions apply the same **data parallelism (SPMD)** to the inner
loop of k-means, following the four steps of the parallelization process:

| Step          | Decision                                                            |
| ------------- | ------------------------------------------------------------------- |
| Decomposition | The dataset is split into chunks of equal size; one k-means iteration over one chunk is a task |
| Assignment    | Each worker runs the same map step on its own chunk: assign its points to the closest centroid, then reduce them to points-per-cluster, feature sums per cluster, and chunk inertia |
| Orchestration | Only the centroids travel out (k x 2 floats) and only the partial sums come back (k x 3 values); the master adds them up and moves the centroids — a map-reduce round per iteration |
| Mapping       | One worker per logical core; the operating system places them on the cores |

The two versions differ only in the mechanism:

- **`lab1-proteins-mp.py`** — one `mp.Pool(mp.cpu_count())` created once, whose
  `initializer` gives each process the dataset a single time, then
  `pool.starmap` per iteration over `16 x 4` chunks. Separate address spaces,
  no GIL.
- **`lab1-proteins-th.py`** — `os.cpu_count()` `threading.Thread`s started and
  joined on every iteration. Shared address space: each thread receives its
  chunk as a NumPy view, with no copy, and appends its partial result to one
  shared list protected by a `threading.Lock`. The speedup exists only because
  NumPy releases the GIL while it computes.

The dataset is read once, by the master, in both versions. Nothing else is
serial except the final label pass and the printing.

## 4. Where the time goes

Phase breakdown on 2,000,000 proteins with 16 workers, from instrumented runs
(figure 2):

| Phase                     | Serial (s) | Multiprocessing (s) | Threads (s) |
| ------------------------- | ---------- | ------------------- | ----------- |
| CSV read (never parallel) |  1.97      | 1.94                | 1.91        |
| Worker startup            | —          | 0.62                | ~0          |
| Elbow, k = 1..10          | 35.17      | 4.75                | 7.79        |
| Final k-means + report    |  0.74      | 0.20                | 0.29        |
| **Total**                 | **37.89**  | **7.52**            | **9.99**    |

Taken alone, the parallelized section — the elbow — speeds up **7.4x** with
processes (46% of 16) and **4.5x** with threads (28%). The whole-program
speedups of 5.1x and 3.8x are lower because the serial read and the worker
startup are charged to the total.

## 5. Choosing the number of workers

### Processes and threads

Each column is measured against a serial reference taken in the same block:
35.41 s for the process sweep, 37.50 s for the thread sweep.

| Workers | Multiprocessing (s) | Speedup | Threads (s) | Speedup | Amdahl bound |
| ------- | ------------------- | ------- | ----------- | ------- | ------------ |
|  1      | 36.83               | 0.96    | 36.55       | 1.03    | 1.00         |
|  2      | 21.10               | 1.68    | 21.72       | 1.73    | 1.90         |
|  4      | 12.73               | 2.78    | 14.43       | 2.60    | 3.45         |
|  8      |  9.00               | 3.93    | 11.34       | 3.31    | 5.82         |
| 12      |  7.73               | 4.58    | 10.25       | 3.66    | 7.56         |
| 16      |  7.39               | 4.79    | 10.16       | 3.69    | 8.88         |

With a single worker both versions land on the serial time to within the
measurement noise (0.96x and 1.03x): one worker doing all the chunks is the
serial program plus the cost of the machinery around it, and that cost is small
enough to disappear into the +-4% spread between runs.

Beyond 12 workers the curves flatten — the last logical cores are SMT siblings
and efficiency cores, and the k-means steps are limited by memory bandwidth as
much as by arithmetic. Threads past one per core only get worse: 24 threads
11.07 s, 32 threads 10.38 s, 48 threads 11.89 s, 64 threads 13.75 s. Threads
are created and joined on every iteration, so extra threads add overhead
without adding work.

### Chunks per process — the one tuning that paid

Giving each process *several* chunks instead of exactly one is worth 16% of the
total runtime (figure 3). Serial reference for this block: 37.06 s.

| Chunks per process | Time (s) | Speedup |
| ------------------ | -------- | ------- |
| 1                  | 9.21     | 4.02    |
| 2                  | 8.21     | 4.52    |
| **4 (chosen)**     | **7.98** | **4.64**|
| 6                  | 7.91     | 4.69    |
| 8                  | 8.07     | 4.59    |
| 12                 | 8.16     | 4.54    |

The cause is the heterogeneous CPU. With one equally sized chunk per process,
every iteration ends only when the slowest efficiency core finishes, and the
performance cores idle at the barrier. Cutting the data finer lets the pool
hand the spare chunks to whichever worker is free, so the fast cores simply do
more of them. The optimum is flat between 2 and 8 chunks per process; 4 was
chosen as the middle of that range rather than a machine-specific peak.

The threaded version cannot use this: there a chunk *is* a thread, so extra
chunks mean extra thread creations, which is what the 24-64 thread measurements
above show.

## 6. Amdahl's law

The serial fraction is the part no number of workers can shrink: the 1.94 s CSV
read plus the ~0.08 s final label pass, against a 37.89 s total.

    s = 2.02 / 37.89 = 0.053   (5.3%)
    p = 0.947

    S(N)  = 1 / (s + p/N)
    S(16) = 1 / (0.053 + 0.947/16) = 8.88x
    S(inf) = 1 / 0.053             = 18.7x

Measured against that bound: multiprocessing reaches 5.07x, **57% of the 8.88x
Amdahl bound**; threads reach 3.80x, 43% of it. The gap between the bound and
the measurement is *not* serial code — it is parallel overhead, load imbalance
on unequal cores, and memory bandwidth.

The Karp-Flatt metric makes that visible. It derives the *effective* serial
fraction from each measured speedup, `e = (1/S - 1/N) / (1 - 1/N)`:

| Workers | Multiprocessing e | Threads e |
| ------- | ----------------- | --------- |
|  2      | 19.2%             | 15.8%     |
|  4      | 14.6%             | 18.0%     |
|  8      | 14.8%             | 20.3%     |
| 12      | 14.7%             | 20.7%     |
| 16      | 15.6%             | 22.2%     |

For multiprocessing `e` settles around 15% against the 5.3% of genuinely serial
code: roughly ten points of the loss are overhead, and because `e` stays flat
as workers are added, that overhead is proportional, not growing — the
per-iteration dispatch and barrier, not a bottleneck that worsens with scale.
For threads `e` climbs steadily from 15.8% to 22.2%, which is the GIL: the
parts of each iteration that run Python rather than NumPy cannot overlap, and
their relative weight grows as the parallel part shrinks.

Gustafson's law gives the optimistic reading for a growing dataset: with the
same 5.3% serial fraction, a problem scaled to 16 workers would show a scaled
speedup of `s + p*N = 15.2x`. The lab's dataset is fixed, so Amdahl is the
relevant bound here, but the elbow-only figure of 7.4x shows what the
implementation reaches when the serial read is taken out of the picture.

## 7. Why threads are slower than processes

1. **The GIL.** NumPy releases it inside its kernels, which is why threads gain
   anything at all, but the Python around them — the loop over chunks, the
   reduction, the convergence test — holds it, and that fraction cannot overlap.
   Karp-Flatt puts it at 22% effective serial fraction at 16 workers, against
   16% for processes.
2. **Thread lifetime.** Threads are created and joined on every one of the
   ~350 iterations, while the process pool is created once.
3. **No load balancing.** Processes can be given more chunks than workers;
   threads cannot, so every iteration waits for the slowest core.

Threads pay no data-transfer cost at all, which is their advantage: the chunks
are views, nothing is pickled, and there is no 0.62 s pool startup. It is not
enough to make up for the three points above.

## 8. When parallelism does not pay

On the 50,000-protein development dataset both parallel versions are **slower**
than the serial one:

| Version         | 50,000 rows (s) | Speedup |
| --------------- | --------------- | ------- |
| serial          | 0.82            | 1.00    |
| multiprocessing | 1.39            | 0.59    |
| threads         | 1.29            | 0.64    |

The work per iteration is too small to cover process creation, dispatch and
synchronization — the textbook case of overhead dominating a fine-grained
workload. Parallelism pays here only at the 2,000,000-row scale the lab asks
for.

## 9. Correctness

The three programs print identical results on both datasets, which is what
makes the timings comparable:

```
Proteins read: 2000000
Optimum number of clusters (k): 3
Cluster with the highest sequence length:
  id: 1
  highest sequence length: 263
  average sequence length: 125.92
  centroid: enzyme=8.50, hydrofob=185.23
```

They share the seed, the initial centroids and the arithmetic; the parallel
versions only change the order in which the partial sums are added, which does
not move a centroid at this precision.

## 10. Figures

| File                       | Shows                                                      |
| -------------------------- | ---------------------------------------------------------- |
| `figures/fig1-speedup.png` | Speedup vs. workers for both versions, against the Amdahl bound and linear scaling |
| `figures/fig2-phases.png`  | Where the time goes in each version                        |
| `figures/fig3-chunks.png`  | Execution time vs. chunks per process                      |

`findings.html` is the same report as a standalone web page with the figures
embedded; open it in a browser.

## 11. Reproducing the measurements

```bash
python proteins-generator.py 2000000 42
python lab1-proteins-serial.py
python lab1-proteins-mp.py
python lab1-proteins-th.py
```

Run them on an idle machine, discard the first run, and keep the median of
three. Worker counts are the constants `PROCESSES` and `CHUNKS_PER_PROCESS` in
`lab1-proteins-mp.py` and `NUM_THREADS` in `lab1-proteins-th.py`; the sweeps
above were produced by overriding them and re-running the same programs.
