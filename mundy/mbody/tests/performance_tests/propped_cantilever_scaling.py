#!/usr/bin/env python3
"""Run ProppedCantileverConditioning over the number of segments N, unpreconditioned and preconditioned by
self-mobility Jacobi and algebraic multigrid, cold and warm-started, at dt = 1e-3; then plot the CG iterations, the
condition number of P A, and the time of one preconditioned solve against N on three y axes.

    python propped_cantilever_scaling.py [--exe PATH] [--out DIR] [--jobs J] [--threads T] [--max-n N] [--plot-only]
                                         [--show]

Every (preconditioner, N) is two runs, each writing its own CSVs under --out:
  - cold: --conditioning --iterations, the condition number at the initial configuration and cold solves;
  - warm: --iterations --warm-start, solves started from the last step's solution.
Iterations and times are those of the solve at step STEPS, so cold and warm solve the same system after STEPS steps.
A run whose CSVs exist is skipped, so an interrupted sweep resumes; failures go to <out>/failures.txt.
The figure is <out>/propped_cantilever_scaling.png.
"""
import argparse
import concurrent.futures
import csv
import glob
import os
import subprocess

DT = 1e-3
STEPS = 2
N_ALL = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
N_AMG = N_ALL + [2000, 5000, 10000, 20000, 50000]
PRECONDITIONERS = {'none': N_ALL, 'jacobi': N_ALL, 'amg': N_AMG}
LABELS = {'none': 'unpreconditioned', 'jacobi': 'self-mobility Jacobi', 'amg': 'self-mobility AMG'}
COLORS = {'none': 'tab:gray', 'jacobi': 'tab:blue', 'amg': 'tab:red'}


def prefix(out, preconditioner, n, warm):
    return os.path.join(out, f'{preconditioner}_N{n}_{"warm" if warm else "cold"}')


def run(exe, out, threads, preconditioner, n, warm):
    p = prefix(out, preconditioner, n, warm)
    if os.path.exists(p + '_iterations.csv'):
        return None
    command = [exe, '--num-segments', str(n), '--preconditioner', preconditioner, '--dt', str(DT), '--iterations',
               '--steps', str(STEPS), '--timings', '--prefix', p]
    command += ['--warm-start'] if warm else ['--conditioning']
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND='false')
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        for path in (p + '_iterations.csv', p + '_conditioning.csv'):
            if os.path.exists(path):
                os.remove(path)
        return f'{" ".join(command)}\n  {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""}\n'
    print(f'done {preconditioner} N={n} {"warm" if warm else "cold"}', flush=True)
    return None


def sweep(exe, out, jobs, threads, max_n):
    os.makedirs(out, exist_ok=True)
    cases = [(p, n, warm) for p, ns in PRECONDITIONERS.items() for n in ns if n <= max_n for warm in (False, True)]
    # Small N first, so every curve fills in from the left
    cases.sort(key=lambda c: c[1])
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for failure in pool.map(lambda c: run(exe, out, threads, *c), cases):
            if failure:
                print('FAILED ' + failure, flush=True)
                failures.append(failure)
    if failures:
        with open(os.path.join(out, 'failures.txt'), 'a') as f:
            f.writelines(failures)


def read_rows(pattern):
    rows = []
    for path in glob.glob(pattern):
        with open(path) as f:
            rows.extend(csv.DictReader(f))
    return rows


def collect(out):
    """{preconditioner: {'kappa': {N: k}, 'iterations': {warm: {N: i}}, 'seconds': {warm: {N: s}}}}."""
    data = {p: {'kappa': {}, 'iterations': {False: {}, True: {}}, 'seconds': {False: {}, True: {}}}
            for p in PRECONDITIONERS}
    for r in read_rows(os.path.join(out, '*_conditioning.csv')):
        data[r['Preconditioner']]['kappa'][int(r['NumSegments'])] = float(r['ConditionNumber'])
    for r in read_rows(os.path.join(out, '*_iterations.csv')):
        if int(r['Step']) != STEPS:
            continue
        p, n, warm = r['Preconditioner'], int(r['NumSegments']), r['WarmStart'] == '1'
        data[p]['iterations'][warm][n] = int(r['Iterations'])
        data[p]['seconds'][warm][n] = float(r['SetupSeconds']) + float(r['SolveSeconds'])
    return data


def plot(data, out, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax_iterations = plt.subplots(figsize=(11, 6.5))
    ax_kappa = ax_iterations.twinx()
    ax_seconds = ax_iterations.twinx()
    ax_seconds.spines['right'].set_position(('axes', 1.12))
    for ax in (ax_iterations, ax_kappa, ax_seconds):
        ax.set_xscale('log')
        ax.set_yscale('log')

    def curve(ax, points, **style):
        ns = sorted(points)
        if ns:
            ax.plot(ns, [points[n] for n in ns], **style)

    for p in PRECONDITIONERS:
        color = COLORS[p]
        curve(ax_kappa, data[p]['kappa'], color=color, marker='s', linestyle='-.', linewidth=1.2)
        for warm in (False, True):
            linestyle = '--' if warm else '-'
            curve(ax_iterations, data[p]['iterations'][warm], color=color, marker='o', linestyle=linestyle)
            curve(ax_seconds, data[p]['seconds'][warm], color=color, marker='^', linestyle=linestyle, alpha=0.7)

    ax_iterations.set_xlabel('number of segments N')
    ax_iterations.set_ylabel(f'CG iterations at step {STEPS} (o)')
    ax_kappa.set_ylabel('condition number of P A (s)')
    ax_seconds.set_ylabel(f'setup + solve time at step {STEPS} [s] (^)')
    ax_iterations.grid(True, which='major', alpha=0.3)
    handles = [Line2D([], [], color=COLORS[p], label=LABELS[p]) for p in PRECONDITIONERS]
    handles += [Line2D([], [], color='k', marker='o', linestyle='', label='iterations'),
                Line2D([], [], color='k', marker='s', linestyle='', label='condition number'),
                Line2D([], [], color='k', marker='^', linestyle='', label='time'),
                Line2D([], [], color='k', linestyle='-', label='cold start'),
                Line2D([], [], color='k', linestyle='--', label=f'warm start after {STEPS} steps'),
                Line2D([], [], color='k', linestyle='-.', label='condition number (initial configuration)')]
    ax_iterations.set_title(f'Propped cantilever, dt = {DT:g}')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8, frameon=False)
    path = os.path.join(out, 'propped_cantilever_scaling.png')
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')
    if show:
        plt.show()


def print_table(data):
    ns = sorted({n for d in data.values() for n in d['kappa']} |
                {n for d in data.values() for w in (False, True) for n in d['iterations'][w]})
    print(f'{"N":>7s}' + ''.join(f'{p + " " + q:>22s}' for p in PRECONDITIONERS for q in ('kappa', 'its c/w')))
    for n in ns:
        line = f'{n:>7d}'
        for p in PRECONDITIONERS:
            k = data[p]['kappa'].get(n)
            cold = data[p]['iterations'][False].get(n, '-')
            warm = data[p]['iterations'][True].get(n, '-')
            kappa = '%.3e' % k if k else '-'
            line += '%22s%22s' % (kappa, '%s/%s' % (cold, warm))
        print(line)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exe', default=os.path.join(here, 'MundyMBody_ProppedCantileverConditioning.exe'))
    parser.add_argument('--out', default='propped_cantilever_scaling')
    parser.add_argument('--jobs', type=int, default=1, help='runs at once')
    parser.add_argument('--threads', type=int, default=1, help='OpenMP threads per run')
    parser.add_argument('--max-n', type=int, default=N_AMG[-1], help='skip runs with more segments')
    parser.add_argument('--plot-only', action='store_true', help='plot the CSVs already in --out')
    parser.add_argument('--show', action='store_true', help='also open the figure')
    args = parser.parse_args()
    if not args.plot_only:
        sweep(args.exe, args.out, args.jobs, args.threads, args.max_n)
    data = collect(args.out)
    print_table(data)
    plot(data, args.out, args.show)


if __name__ == '__main__':
    main()
