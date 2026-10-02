# Lab 1 — k-means parallelization in Python

Master in Big Data — *Technological Fundamentals in the Big Data World*.

Parts one to three are implemented: the serial program and, for each of the
two parallel parts, two programs that solve it under a different parallel
paradigm. Part four, the written report, is not part of this repository.

| Part | Program                   | Tool              | Paradigm                               |
| ---- | ------------------------- | ----------------- | -------------------------------------- |
| 1    | `lab1-proteins-serial.py` | —                 | none                                   |
| 2    | `lab1-proteins-mp.py`     | `multiprocessing` | data parallel (SPMD)                   |
| 2    | `lab1-proteins-mp-ep.py`  | `multiprocessing` | task farming (embarrassingly parallel) |
| 3    | `lab1-proteins-th.py`     | `threading`       | data parallel (SPMD)                   |
| 3    | `lab1-proteins-th-ep.py`  | `threading`       | task farming (embarrassingly parallel) |

Every program prints exactly the same results — same seed, same initial
centroids, same arithmetic — and they differ only in how the work is spread.

## Dataset

The dataset is produced by `proteins-generator.py`, which is course material
and is kept byte for byte as it was handed out (the lint and format hooks skip
it). It writes `proteins.csv` into the current directory:

```bash
python proteins-generator.py 50000 42      # development
python proteins-generator.py 2000000 42    # performance measurements
```

`proteins.csv` is not committed.

## Running

```bash
python -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
python -m pip install --group run
python lab1-proteins-serial.py    # part one
python lab1-proteins-mp.py        # part two,   data parallel
python lab1-proteins-mp-ep.py     # part two,   task farming
python lab1-proteins-th.py        # part three, data parallel
python lab1-proteins-th-ep.py     # part three, task farming
```

LAB1.pdf accepts exactly three programs in the delivery, named
`lab1-proteins-serial.py`, `lab1-proteins-mp.py` and `lab1-Proteins-th.py`
(with a capital `P` for the threaded one, and all lowercase here). The `-ep`
pair is kept in the repository for the comparison the report needs: zip
whichever pair of parallel programs is delivered under those two names.

The lab forbids absolute or relative paths to the dataset, so the program
reads `proteins.csv` from the working directory: run it from the directory
that holds the file.

## What the programs do

1. Starts the clock.
2. Reads `proteins.csv`, keeping `enzyme`, `hydrofob` and the length of every
   `sequence`.
3. Runs k-means for `k = 1..10` and builds the elbow curve.
4. Picks the optimum `k` as the point of the curve furthest from the straight
   line joining its two ends, with both axes scaled to `[0, 1]` first.
5. Clusters the proteins by `enzyme` and `hydrofob` with that `k`.
6. Finds the cluster holding the longest sequence — ties broken by the
   `hydrofob` of the centroid — and averages the sequence lengths in it.
7. Stops the clock, prints the results and the execution time.
8. Only then builds the three figures and shows them: elbow graph, clusters
   with their centroids, and a heat map of the centroid values. The elbow
   graph draws the criterion of step 4 as well as the curve — the line joining
   the two ends, and the perpendicular dropped from the chosen point onto it.

Nothing blocks while the measured work is running, as the lab requires: no
figure is created before the clock has been stopped.

k-means is implemented from scratch on top of NumPy: euclidean distance,
random initial centroids drawn from the data, and Lloyd's iteration until the
centroids move less than `TOLERANCE` or `MAX_ITERATIONS` is reached. The
random generator is always seeded with `SEED = 42`, so runs are reproducible.

## How the parallel versions work

### Data parallel — `lab1-proteins-mp.py`, `lab1-proteins-th.py`

Both follow the same data-parallel (SPMD) decomposition, applied to the inner
loop of k-means, which is where all the time goes:

- **Decomposition.** The dataset is split once into one chunk per worker.
- **Map.** Every worker runs the same step over its own chunk: assign its
  points to the closest centroid and reduce them to three small partial
  results — points per cluster, sum of the features per cluster, and inertia.
- **Reduce.** The master adds the partial results up and moves the centroids.
  Only the centroids travel out and only the partial sums come back, so the
  data stays where it is for the whole run.

`lab1-proteins-mp.py` creates one `mp.Pool(mp.cpu_count())`, whose
`initializer` hands the dataset to every process once, and then calls
`pool.starmap` per iteration over `CHUNKS_PER_PROCESS` chunks per worker. `lab1-proteins-th.py` starts `os.cpu_count()`
`threading.Thread`s per iteration; they share the address space, so each one
receives its chunk as a NumPy view with no copy, and they collect their
partial results in a shared list protected by a lock. NumPy releases the GIL
while it works, which is what makes the threaded version gain anything at all.

### Task farming — `lab1-proteins-mp-ep.py`, `lab1-proteins-th-ep.py`

The other two programs parallelize the outer loop instead, the one that runs
k-means once per candidate `k` to build the elbow curve. Those runs are
completely independent of each other, which makes this the embarrassingly
parallel decomposition of the problem:

- **Decomposition.** One task per candidate `k`: a whole k-means run.
- **Assignation.** The ten tasks are farmed out to ten workers.
- **Orchestration.** None. No task reads what another task writes, so there is
  nothing to synchronize while they run — the threaded version needs a lock
  only to drop each finished model into the shared result table.
- **Combination.** Once every task has finished, the master reads the
  inertias off the collected models and picks the optimum `k`. The model for
  that `k` was already fitted by its task, so the final clustering costs no
  extra k-means run — only one pass to label the points.

`lab1-proteins-mp-ep.py` creates a `mp.Pool` and calls `pool.map` once, over
`range(1, 11)`; the pool `initializer` hands the dataset to every process at
startup so that no task carries it. `lab1-proteins-th-ep.py` starts one
`threading.Thread` per `k`, all sharing the one copy of the dataset, and joins
them all before reading the results. Each task seeds its own generator with
`SEED`, so which worker picks a task up, and in which order the tasks finish,
cannot change the result.

Task farming is the simpler of the two paradigms — one `map` call against a
per-iteration dispatch — but it is the slower one here, for two reasons that
the report covers. The tasks are very unequal: k-means with `k = 10` takes
4.1 s on its own and k-means with `k = 1` takes 0.07 s, and a farm can never
finish before its longest task, which caps the speedup of the clustering phase
at about 10. And the ten tasks each stream the whole 16 MB dataset from main
memory, whereas the data-parallel versions give every worker a chunk small
enough to stay in cache, so the farm saturates the memory bandwidth first.

## Measured times

Median of three interleaved runs on 2,000,000 proteins, `SEED = 42`, on an
idle i5-13500H (12 cores — 4 performance cores with SMT and 8 efficiency cores
— 16 logical):

| Version                          | Time    | Speedup |
| -------------------------------- | ------- | ------- |
| serial                           | 41.60 s | 1.00    |
| multiprocessing, data parallel   |  8.30 s | 5.01    |
| threads, data parallel           | 10.79 s | 3.86    |
| multiprocessing, task farming    | 14.38 s | 2.89    |
| threads, task farming            | 14.01 s | 2.97    |

Measure on an idle machine: under sustained load the CPU settles into a lower
clock, and the same program takes 35.0 s cold against 37.4 s in steady state.
Speedups are unaffected; absolute times only compare within one measurement
session.

The full measurement set for the two data-parallel programs — scaling curves,
phase breakdown, Amdahl and Karp-Flatt analysis, figures — is in
[`findings.md`](findings.md), and as a web page in
[`findings.html`](findings.html). Its absolute times come from an earlier
session and run about 10 % faster than the table above, which was measured in
one go with the task-farming programs; the speedups are what compare.

Reading the CSV takes about 1.8 s in every version and is not parallelized,
which bounds the speedup from above; the rest of the gap comes from the
per-iteration dispatch and from memory bandwidth, since the k-means steps move
much more data than they compute.

### Choosing the number of workers

Both counts were measured rather than guessed, three runs per setting on the
2,000,000-protein dataset.

Processes, at four chunks per process: 12.7 s with 4, 9.0 s with 8, 7.7 s with
12, 7.4 s with 16. One process per logical core it is — past that the machine
has nothing left to give.

Chunks per process, with 16 processes: 9.2 s with one chunk each, 8.2 s with
two, 8.0 s with four, 7.9 s with six, 8.1 s with eight, 8.2 s with twelve. The
cores are not equal — four performance cores and eight efficiency ones — so
one equally sized chunk per process makes every iteration wait for the slowest
core. Cutting the data finer lets the pool hand the extra chunks to whoever is
free first; four per process sits in the middle of the flat optimum.

Threads: 11.3 s with 8, 10.2 s with 16, 10.4 s with 32, 11.9 s with 48, 13.8 s
with 64. One thread per logical core again. The threaded version cannot use
the chunking trick — a chunk is a thread there, and threads are created and
joined on every iteration, so extra chunks only add overhead instead of
balancing the load. That, plus the parts of NumPy that keep the GIL, is why
threads end up slower than processes.

The task-farming programs have nothing to choose: the farm holds exactly one
task per candidate `k`, so it runs ten workers whatever the machine offers.
The pool is capped at ten processes for that reason — a sixteenth process
would only pay for its own startup and for a copy of the dataset it would
never cluster, which measured 1.6 s of the run.

## Layout

```
.
├── findings.md               # measurements and analysis for the report
├── findings.html             # the same report as a standalone web page
├── figures/                  # the report figures, 160 dpi
├── lab1-proteins-serial.py   # part one, serial
├── lab1-proteins-mp.py       # part two, multiprocessing, data parallel
├── lab1-proteins-mp-ep.py    # part two, multiprocessing, task farming
├── lab1-proteins-th.py       # part three, threads, data parallel
├── lab1-proteins-th-ep.py    # part three, threads, task farming
├── proteins-generator.py     # course material, untouched
├── proteins.csv              # generated, not committed
├── authors.txt               # one line per author (NIA, SURNAMES, NAME)
├── pyproject.toml            # every tool is configured here
├── .pre-commit-config.yaml
└── .github/workflows/ci.yml
```

The delivery expects standalone `.py` files, so each part is a single script
at the root of the repository rather than a package. The programs therefore
repeat the code they have in common — reading the dataset, choosing
the optimum `k`, reporting and plotting — instead of importing it from a
shared module that could not be delivered.

## Everyday commands

| Task                | Command                      |
| ------------------- | ---------------------------- |
| Lint                | `ruff check .`               |
| Lint and autofix    | `ruff check --fix .`         |
| Format              | `ruff format .`              |
| Check formatting    | `ruff format --check .`      |
| Type-check          | `pyright`                    |
| Run every hook      | `pre-commit run --all-files` |
| Update hook pins    | `pre-commit autoupdate`      |
