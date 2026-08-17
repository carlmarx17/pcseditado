# PSC with ADIOS2 on COSMA7

Operational procedure to build, validate, run and restart the PSC
anisotropy cases with ADIOS2 checkpoints.

## Validated state

Configuration verified on June 24, 2026:

```text
Repository:  /cosma7/data/dp433/dc-mart18/pcseditado
Single build: /cosma7/data/dp433/dc-mart18/pcseditado/build
ADIOS2:      $HOME/adios2 (version 2.12.0.182)
Partition:   cosma7-rp
Account:     dp433
Compiler:    gnu_comp/14.1.0
MPI:         openmpi/5.0.3
HDF5:        parallel_hdf5/1.14.4
Launcher:    mpirun
```

ADIOS2 is actually enabled because:

```bash
grep -n PSC_HAVE_ADIOS2 build/src/include/PscConfig.h
ldd build/src/psc_mirror_bikappa3 | grep adios2
```

The expected output contains:

```text
#define PSC_HAVE_ADIOS2
libadios2_cxx_mpi.so
libadios2_core_mpi.so
```

The following were also validated:

- writing BP5 checkpoints;
- reading a checkpoint and continuing the computation;
- MPI run with 28 processes on one node;
- MPI run with 56 processes on two nodes;
- production run with 1024 processes on 37 nodes.

## Important files

```text
src/cosma_adios2_env.sh
src/cosma_adios2_setup.sh
src/cosma_build_psc_adios2.sh
src/submit_anisotropy_adios2.slurm
adios2cfg.xml
build/
```

Only one build folder named `build` should exist. Do not use
`build-adios2` or `build-adios2-nohdf5`.

## 1. Log in and load the environment

```bash
ssh dc-mart18@login7.cosma.dur.ac.uk
cd /cosma7/data/dp433/dc-mart18/pcseditado
source src/cosma_adios2_env.sh
```

Check the environment:

```bash
adios2-config --version
command -v mpirun
command -v h5pcc
module list
```

The script cleans Conda, loads the compatible modules and sets
`ADIOS2_DIR`, `PATH` and `LD_LIBRARY_PATH`.

## 2. Install ADIOS2 if missing

This step is not necessary as long as the following exists:

```text
$HOME/adios2/bin/adios2-config
```

Check:

```bash
test -x "$HOME/adios2/bin/adios2-config"
$HOME/adios2/bin/adios2-config --version
```

If it does not exist:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
BUILD_JOBS=4 src/cosma_adios2_setup.sh
```

Do not mix this installation with old ADIOS2, OpenMPI or HDF5 modules.

## 3. Create the single build

For a clean rebuild:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
rm -rf build
BUILD_JOBS=4 src/cosma_build_psc_adios2.sh
```

The script configures CMake with:

```text
PSC_USE_ADIOS2=ON
USE_CUDA=OFF
USE_VPIC=OFF
BUILD_TESTING=OFF
```

COSMA7 note: `cmake` is currently available on the login node, but not
necessarily on the compute nodes. A build job may fail with:

```text
cmake: command not found
```

Do not confuse this environment issue with an ADIOS2 failure. If you want
to build entirely on a compute node, a version of CMake accessible from
that node must first be installed or exposed.

## 4. Verify the build

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
source src/cosma_adios2_env.sh

grep -n PSC_HAVE_ADIOS2 build/src/include/PscConfig.h
ldd build/src/psc_mirror_bikappa3 | grep -i adios
```

Check the executables:

```bash
ls -l \
  build/src/psc_mirror_bikappa3 \
  build/src/psc_mirror_bikappa5 \
  build/src/psc_firehose_bikappa3 \
  build/src/psc_firehose_bikappa5 \
  build/src/psc_mirror_bimaxwellian_strong \
  build/src/psc_firehose_bimaxwellian_strong \
  build/src/psc_whistler_bimaxwellian_strong
```

Do not submit production runs if `PSC_HAVE_ADIOS2` is missing or if `ldd`
shows `not found`.

## 5. Why mpirun is used

The validated launcher is:

```text
mpirun -np $SLURM_NTASKS
```

Do not use `srun --mpi=pmi2` with OpenMPI 5. In an earlier test it
produced:

```text
No PMIx server was reachable, but a PMI1/2 was detected.
1024 singletons will be started.
```

That starts independent MPI ranks, consumes memory massively and ends up
in OOM. `src/cosma_adios2_env.sh` uses `mpirun` by default.

## 6. Submit a case

The default target is `psc_mirror_bikappa3`:

```bash
cd /cosma7/data/dp433/dc-mart18/pcseditado
sbatch src/submit_anisotropy_adios2.slurm
```

Other cases:

```bash
sbatch --export=ALL,PSC_TARGET=psc_mirror_bikappa5 \
  src/submit_anisotropy_adios2.slurm

sbatch --export=ALL,PSC_TARGET=psc_firehose_bikappa3 \
  src/submit_anisotropy_adios2.slurm

sbatch --export=ALL,PSC_TARGET=psc_firehose_bikappa5 \
  src/submit_anisotropy_adios2.slurm

sbatch --export=ALL,PSC_TARGET=psc_mirror_bimaxwellian_strong \
  src/submit_anisotropy_adios2.slurm
```

The script requests:

```text
37 nodes
28 processes per node
1024 MPI processes
48 hours
partition cosma7-rp
account dp433
```

It does not set a `--nodelist`: Slurm selects free nodes.

## 7. Slurm prolog failures

If a job terminates immediately with:

```text
State=CANCELLED
Reason=Prolog
ExitCode=0:0
```

and does not generate `.out` or `.err`, the PSC script never got to run.
It is a node prolog failure, not an ADIOS2 failure.

Check:

```bash
scontrol show job -dd JOBID
sacct -j JOBID \
  --format=JobID,State,ExitCode,Elapsed,NodeList,Reason -X
```

If COSMA has not yet fixed the affected nodes, they can be temporarily
excluded when submitting:

```bash
sbatch --exclude='m[7031-7043]' src/submit_anisotropy_adios2.slurm
```

The exclusion should be temporary and based on observed prolog failures;
it should not become a permanent fixed list.

## 8. Confirm that the job is using ADIOS2

```bash
JOBID=12345678
LOG=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_aniso_${JOBID}.out

grep -E '^(target|job|nodes|ntasks|adios2_dir|adios2_config|launcher)=' "$LOG"
```

It should show:

```text
adios2_dir=/cosma/home/dp433/dc-mart18/adios2
adios2_config=/cosma/home/dp433/dc-mart18/adios2/bin/adios2-config
launcher=mpirun
launcher=mpirun -np 1024 ./psc_mirror_bikappa3
```

Also verify the binary copied to the run directory:

```bash
TARGET=psc_mirror_bikappa3
RUN=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/${TARGET}_${JOBID}

ldd "$RUN/$TARGET" | grep -i adios
```

## 9. Confirm checkpoint writing

For all mirror/firehose/whistler anisotropy cases:

```text
fields and moments: every 500 steps
particles:          every 10000 steps
checkpoint:         every 5000 steps
grid:               1024x1024
particles/cell:     1500
mi/me:              200
nmax:               depends on saturation; default 1200000
load balancing:     every 2500 steps
continuity:         every 5000 steps
energy:             every 5000 steps in diag.asc
```

Use `PSC_NMAX` in `--export` to set the step cap for each run according
to the observed saturation.

Before the first interval there will be no `checkpoint_*.bp` folder. That
does not mean ADIOS2 is disabled.

Once the interval is reached:

```bash
find "$RUN" -maxdepth 1 -type d -name 'checkpoint_*.bp' -print
find "$RUN" -maxdepth 2 -type f -path '*.bp/*' -ls | head
```

A valid BP5 folder contains files such as:

```text
data.0
md.0
md.idx
profiling.json
```

## 10. Monitor

```bash
squeue -j "$JOBID" -o '%.18i %.16P %.24j %.2t %.10M %.4D %R'
tail -f "$LOG"
```

Diagnostics and results:

```bash
tail -f "$RUN/diag.asc"
ls -ltr "$RUN" | tail
du -sh "$RUN"
```

Do not run the simulation directly on the login node.

## 11. Restart from a checkpoint

Example:

```bash
export PSC_RESTART=/cosma7/data/dp433/dc-mart18/anisotropy_adios2/psc_mirror_bikappa3_JOBID/checkpoint_5000.bp

sbatch \
  --export=ALL,PSC_TARGET=psc_mirror_bikappa3,PSC_RESTART="$PSC_RESTART" \
  src/submit_anisotropy_adios2.slurm
```

The executable should print:

```text
**** Reading checkpoint...
```

and continue from the stored step.

## 12. Optional short test

The environment variables allow the problem to be reduced:

```bash
sbatch \
  --export=ALL,PSC_TARGET=psc_mirror_bikappa3,PSC_NMAX=4,PSC_NGRID=16,PSC_NP_Y=1,PSC_NP_Z=1,PSC_NICELL=16,PSC_CHECKPOINT_EVERY=2,PSC_FIELDS_EVERY=2,PSC_PARTICLES_EVERY=2 \
  src/submit_anisotropy_adios2.slurm
```

For a test, the Slurm resources must also be adjusted. Do not submit this
configuration with 37 nodes.

A correct test should:

1. finish with exit code zero;
2. create `checkpoint_2.bp`;
3. contain `data.0`, `md.0` and `md.idx`;
4. allow a restart with `PSC_RESTART`.

## Common issues

### `write_checkpoint not available without adios2`

The executable was built without ADIOS2 or was taken from a different
build:

```bash
grep PSC_HAVE_ADIOS2 build/src/include/PscConfig.h
ldd build/src/psc_mirror_bikappa3 | grep -i adios
```

### `libadios2_*.so => not found`

```bash
source src/cosma_adios2_env.sh
echo "$ADIOS2_DIR"
echo "$LD_LIBRARY_PATH"
```

### Immediate OOM with 1024 processes

Search the error for:

```text
1024 singletons will be started
```

If it appears, `srun --mpi=pmi2` was used. Switch back to `mpirun`.

### Job cancelled with no logs

Check `Reason=Prolog`. If it appears, review or temporarily exclude the
affected nodes.

### `checkpoint_*.bp` has not appeared yet

Check the current step and the checkpoint interval. For example,
`psc_mirror_bikappa3` does not write the first checkpoint until step 5000.
