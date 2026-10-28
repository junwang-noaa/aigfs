#!/usr/bin/env python3
                    
import os   
import subprocess
from time import time
import json
import gc
import multiprocessing as mp
from multiprocessing import shared_memory
import concurrent.futures
                
import xarray as xr
import datetime
import grib2io
import numpy as np
import pandas as pd
        
        
SECTION3 = np.array([0, 1038240, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 1440, 721, 0, -1, 90000000, 0, 48, -90000000, 359750000,250000, 250000, 0])
            
            
class Grib2Writer:
    def __init__(self, start_date, case_name="aigfs", json_path=None):
        self.case_name = case_name
                
        if self.case_name == "aigfs":
            table_file = f"{json_path}/tables_aigfs.json"
        elif self.case_name.startswith("aige"):
            table_file = f"{json_path}/tables_aigefs.json"
        else:   
            raise ValueError(f"name {self.case_name} is not supported!")
            
        with open(table_file, "r") as f:
            self.attrs = json.load(f)
        self.start_date = start_date
        
    def create_grib2_message(self, var, da, lead, level=None):

        # Set duration. NOTE: the duration attr exists for all Grib2Message objects.
        # For Grib2Messages that are instantaneous, the duration is just 0.
        duration = datetime.timedelta(hours=0)
        if var == "total_precipitation_6hr":
            duration = datetime.timedelta(hours=6)
        elif var == "total_precipitation_cumsum":
            duration = datetime.timedelta(hours=lead)
    
        # Create GRIB2 message.
        msg = grib2io.Grib2Message(
            section3=SECTION3,
            pdtn=self.attrs[var]["templates"]["pdtn"],
            drtn=self.attrs[var]["templates"]["drtn"],
        )

        # Set GRIB2 attributes from json table.
        for k,v in self.attrs[var]["attrs"].items():
            setattr(msg, k, v)

        # Set GRIB2 attributes for ensemble members
        if self.case_name.startswith("aige"):
            number = int(self.case_name[-2:])
            msg.perturbationNumber = number

        # update decScaleFactor for specific humidity
        # 12 for [5000, 10000]Pa, 10 for [15000, ..., 40000]Pa, 8 for [50000, ..., 100000]Pa
        if var == "specific_humidity":
            if level >= 5000 and level <= 10000:
                msg.decScaleFactor = 12
            elif level >= 15000 and level <= 40000:
                msg.decScaleFactor = 10
            elif level >= 50000 and level <= 100000:
                msg.decScaleFactor = 8
            else:
                raise ValueError(f"level {level} Pa is not included in this model!")
            
        # Set GRIB2 attributes unique to each iteration.
        msg.refDate = self.start_date
        msg.duration = duration
        msg.unitOfForecastTime = 1 # Hour
        msg.leadTime = datetime.timedelta(hours=lead)
        if level is not None:
            msg.scaledValueOfFirstFixedSurface = level

        return msg

    def save_grib2(self, xarray_ds, outdir):

        prefix = "aigefs" if self.case_name.startswith("aige") else "aigfs"

        # Convert geopotential to geopotential height.
        xarray_ds["geopotential"] = xarray_ds["geopotential"] / 9.80665

        # Update total_precipitation_6h unit to (kg/m^2) and set min to zero
        if "total_precipitation_6hr" in xarray_ds:
            xarray_ds["total_precipitation_6hr"] = xarray_ds["total_precipitation_6hr"].clip(min=0) * 1000

        # Drop total_precipitation_cumsum for AIGEFS. Otherwise update unit to (kg/m^2) and set min to zero
        if "total_precipitation_cumsum" in xarray_ds:
            if self.case_name.startswith("aige"):
                xarray_ds = xarray_ds.drop_vars("total_precipitation_cumsum")
            else:
                xarray_ds["total_precipitation_cumsum"] = xarray_ds["total_precipitation_cumsum"].clip(min=0) * 1000

        # Set min spfh to zero
        if "specific_humidity" in xarray_ds:
            xarray_ds["specific_humidity"] = xarray_ds["specific_humidity"].clip(min=0)

        # Convert levels values from mb to Pa.
        xarray_ds["level"] = xarray_ds["level"] * 100 # Convert mb to Pa
        xarray_ds = xarray_ds.squeeze(dim="batch")

        # Reverse lat
        xarray_ds = xarray_ds.reindex(lat = xarray_ds.lat[::-1])
        

        # Set output GRIB2 file.
        cycle = self.start_date.hour
        lead = int(xarray_ds.time.dt.total_seconds()//3600)
        outfile_sfc = os.path.join(outdir, f"{prefix}.t{cycle:02d}z.sfc.f{lead:03d}.grib2")
        outfile_pres = os.path.join(outdir, f"{prefix}.t{cycle:02d}z.pres.f{lead:03d}.grib2")

        ## save to netcdf file
        #xarray_ds.to_netcdf(f"{prefix}.t{cycle:02d}z.f{lead:03d}.nc")

        # Delete the old file.
        for outfile in [outfile_sfc, outfile_pres]:
            if os.path.isfile(outfile):
                os.remove(outfile)

        # Open GRIB2 file.
        grib2_out_sfc = grib2io.open(outfile_sfc, mode="w")
        print(f" Opening GRIB2 File for surface variables: {outfile_sfc}")

        grib2_out_pres = grib2io.open(outfile_pres, mode="w")
        print(f" Opening GRIB2 File for pressure level variables: {outfile_pres}")

        # Iterate over the variable name keys in JSON file.
        for var in sorted(xarray_ds.data_vars):


            # Get variable as DataArray.
            da = xarray_ds[var]

            # Iterate over level...
            if "level" in da.coords.keys():
                for level in da.coords["level"]:
                    msg = self.create_grib2_message(var, da, lead, level=level)
                    msg.data = np.squeeze(da.sel(level=level).values)
                    msg.pack()
                    print(f"\t{msg}")
                    grib2_out_pres.write(msg)
            else:
                msg = self.create_grib2_message(var, da, lead)
                msg.data = np.squeeze(da.values)
                msg.pack()
                print(f"\t{msg}")
                grib2_out_sfc.write(msg)

        # Close GRIB2 file
        grib2_out_sfc.close()
        grib2_out_pres.close()

        # Use wgrib2 to generate index files
        for outfile in [outfile_sfc, outfile_pres]:
            output_idx_file = f"{outfile}.idx"
            
            # Construct the wgrib2 command
            wgrib2_command = ['wgrib2', '-s', outfile]
            
            try:
                # Open the output file for writing
                with open(output_idx_file, "w") as f_out:
                    # Execute the wgrib2 command and redirect stdout to the output file
                    subprocess.run(wgrib2_command, stdout=f_out, check=True)
            
                print(f"Index file created successfully: {output_idx_file}")
            
            except subprocess.CalledProcessError as e:
                print(f"Error running wgrib2 command: {e}")


# Each worker packs one lead time single-threaded; without this every worker (and the
# wgrib2 it runs) starts an OpenMP team sized to the whole node and they thrash.
_WORKER_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

# Per-worker Grib2Writer, built once by the pool initializer so the json table
# is not reloaded for every lead time.
_WORKER_WRITER = None


def _init_worker(start_date, case_name, json_path):
    global _WORKER_WRITER
    _WORKER_WRITER = Grib2Writer(start_date, case_name=case_name, json_path=json_path)


def _to_shared_memory(xarray_ds):
    """Copy the data variables of xarray_ds into one shared memory block.

    Returns (block name, layout, coords) where coords is xarray_ds without its data
    variables (small, sent by pickle) and layout lists, per data variable,
    (name, dims, shape, dtype, offset, attrs) inside the block.
    """
    layout, offset = [], 0
    for name, da in xarray_ds.data_vars.items():
        layout.append((name, da.dims, da.shape, da.dtype.str, offset, dict(da.attrs)))
        offset += -(-da.nbytes // 64) * 64  # keep every array 64-byte aligned
    shm = shared_memory.SharedMemory(create=True, size=max(offset, 1))
    try:
        for name, dims, shape, dtype, off, attrs in layout:
            dst = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=off)
            np.copyto(dst, xarray_ds[name].values)
            del dst
    except BaseException:
        shm.close()
        shm.unlink()
        raise
    shm.close()  # the block stays alive until the worker unlinks it
    return shm.name, layout, xarray_ds.drop_vars(list(xarray_ds.data_vars))


def _unlink_shared_memory(name):
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return
    shm.close()
    shm.unlink()


def _save_grib2_task(shm_name, layout, coords, outdir, lead):
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        # Rebuild the dataset on views of the shared block (no copy).
        xarray_ds = coords.copy()
        for name, dims, shape, dtype, offset, attrs in layout:
            xarray_ds[name] = xr.Variable(
                dims, np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=offset), attrs)
        t0 = time()
        _WORKER_WRITER.save_grib2(xarray_ds, outdir)
        elapsed = time() - t0
        del xarray_ds
    finally:
        try:
            shm.close()
        except BufferError:  # a view is still referenced somewhere; collect and retry
            gc.collect()
            try:
                shm.close()
            except BufferError:
                pass
        shm.unlink()
    print(f" [grib2 worker {os.getpid()}] f{lead:03d} written in {elapsed:.1f} s", flush=True)
    return lead, elapsed


def default_num_workers():
    """CPUs available to this process (honors Slurm/numactl affinity), minus one
    left for the main process that drives the GPU."""
    try:
        ncpu = len(os.sched_getaffinity(0))
    except AttributeError:
        ncpu = os.cpu_count() or 1
    return max(1, ncpu - 1)


class AsyncGrib2Writer:
    """Writes GRIB2 files on a pool of CPU worker processes, one lead time per task,
    so GRIB2 packing and wgrib2 indexing overlap with the GPU forecast.

    Each lead time's arrays go to the worker through a shared memory block rather
    than being pickled through a pipe: one memcpy in the caller, no copy in the worker.

    submit() returns as soon as the dataset is queued; it only blocks when
    max_pending lead times are already in flight, which bounds host memory.
    close() waits for all writes and raises if any of them failed.
    """

    def __init__(self, start_date, case_name="aigfs", json_path=None,
                 num_workers=None, max_pending=None):
        # Validate case_name/table in the parent so a bad config fails fast.
        Grib2Writer(start_date, case_name=case_name, json_path=json_path)

        self.num_workers = num_workers if num_workers else default_num_workers()
        self.max_pending = max_pending if max_pending else self.num_workers + 2

        # forkserver, not fork: the parent runs JAX/CUDA (multithreaded) and forking it
        # is unsafe. The fork server imports __main__ and this module once, and
        # workers are forked from that clean process.
        ctx = mp.get_context("forkserver")
        ctx.set_forkserver_preload(["__main__", __name__])
        self.pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(start_date, case_name, json_path),
        )
        self.pending = set()
        self.failures = []
        self.t0 = time()
        print(f" AsyncGrib2Writer: {self.num_workers} worker processes, "
              f"up to {self.max_pending} lead times in flight", flush=True)

        # Start the workers now so the fork server's imports overlap with model setup.
        # The fork server is launched by the first submit and its environment is
        # inherited by all workers, so the thread limits are set only around this.
        saved_env = {k: os.environ.get(k) for k in _WORKER_THREAD_ENV}
        os.environ.update({k: "1" for k in _WORKER_THREAD_ENV})
        try:
            for _ in range(self.num_workers):
                self.pending.add(self.pool.submit(os.getpid))
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _collect(self, return_when):
        done, self.pending = concurrent.futures.wait(self.pending, return_when=return_when)
        for fut in done:
            exc = fut.exception()
            if exc is not None:
                print(f" AsyncGrib2Writer: write failed: {exc!r}", flush=True)
                self.failures.append(exc)

    def submit(self, xarray_ds, outdir):
        """Queue one lead time (dataset with a single time step) for writing.

        Returns the seconds spent blocked waiting for a free slot (0 if the
        writers are keeping up).
        """
        lead = int(xarray_ds.time.dt.total_seconds().values.ravel()[0] // 3600)
        t0 = time()
        while len(self.pending) >= self.max_pending:
            self._collect(concurrent.futures.FIRST_COMPLETED)
        wait = time() - t0
        shm_name, layout, coords = _to_shared_memory(xarray_ds)
        try:
            fut = self.pool.submit(_save_grib2_task, shm_name, layout, coords, outdir, lead)
        except BaseException:
            _unlink_shared_memory(shm_name)
            raise
        # The worker frees the block; free it here if the task never ran or failed early.
        fut.add_done_callback(
            lambda f, name=shm_name: (f.cancelled() or f.exception() is not None)
            and _unlink_shared_memory(name))
        self.pending.add(fut)
        return wait

    def close(self):
        self._collect(concurrent.futures.ALL_COMPLETED)
        self.pool.shutdown(wait=True)
        print(f" AsyncGrib2Writer: all GRIB2 writes finished, "
              f"{time() - self.t0:.1f} s after pool start", flush=True)
        if self.failures:
            raise RuntimeError(f"{len(self.failures)} GRIB2 write(s) failed; first: {self.failures[0]!r}")

    def abort(self):
        """Drop queued writes and stop the workers (used when the forecast fails)."""
        self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False


if __name__ == "__main__":
    
    table_file = "tables.json"
    
    start_date = pd.to_datetime("2025-07-30 06:00:00")
    ds = xr.open_dataset("forecasts_levels-13_steps-64.nc")
    g2prefix = "aigec00"

    t0 = time()
    outdir = "./"
    os.makedirs(outdir, exist_ok=True)
    converter = Grib2Writer(start_date)
    converter.save_grib2(ds, g2prefix, outdir)

    print(f"It took {(time()-t0)/60} mins")
