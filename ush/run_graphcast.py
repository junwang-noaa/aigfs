#!/usr/bin/env python3
            
'''     
Description: Script to call the graphcast model using gdas products
Author: Sadegh Sadeghi Tabas (sadegh.tabas@noaa.gov)
Revision history:
    -20240307: Sadegh Tabas, initial code
    -20250506: Linlin Cui, moved hard coded model weights to a json file
    -20261026: write GRIB2 through an asynchronous multi-process CPU pipeline
               (AsyncGrib2Writer) so encoding overlaps with the GPU rollout
    -20261026: host-side work for each step runs on a background thread so the
               GPU starts the next step immediately; --monitor timing report
'''     
import os
import argparse
import queue
import threading
from time import time
from datetime import timedelta
import dataclasses
import functools
import re
import haiku as hk 
import jax
import numpy as np
import xarray
import boto3
import pandas as pd
import pickle
from absl import logging
        
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout 
            
from utils.grib2writer import AsyncGrib2Writer
from utils.resource_monitor import ResourceMonitor, GpuStepTimer, gpu_memory_stats, nvidia_smi_used, basic_memory_summary
            
class GraphCastModel:
    def __init__(
        self,   
        pretrained_model_path,  
        gdas_data_path,
        case_name: str, 
        config_file = None,
        output_dir = None, 
        num_pressure_levels = 13,
        forecast_length = 64,
        grib_workers = None,
    ):  
        self.pretrained_model_path = pretrained_model_path
        self.gdas_data_path = gdas_data_path
        self.forecast_length = forecast_length
        self.case_name = case_name
        self.num_pressure_levels = num_pressure_levels
        self.config_file_path = config_file
        self.grib_workers = grib_workers
        self.writer = None
        self.monitor = None
        
        if output_dir is None:
            self.output_dir = os.getcwd()
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.params = None
        self.state = {}
        self.model_config = None
        self.task_config = None
        self.diffs_stddev_by_level = None
        self.mean_by_level = None
        self.stddev_by_level = None
        self.current_batch = None
        self.inputs = None
        self.targets = None
        self.forcings = None
        self.s3_bucket_name = "noaa-nws-graphcastgfs-pds"
        self.dates = None
        

    def load_pretrained_model(self):
        """Load pre-trained GraphCast model."""
        if self.num_pressure_levels==13:
            #model_weights_path = f"{self.pretrained_model_path}/params/GCGFSv2_finetuned_GDAS-ERA5_0p25_13pl_mesh2to6_tp_output_only.npz"
            model_weights_path = f"{self.pretrained_model_path}/params/AIGFSv1.1.npz"
        else:
            model_weights_path = f"{self.pretrained_model_path}/params/GraphCast_ERA5_1979-2017_0p25_37pl_mesh2to6_tp-input-and-output.npz"

        with open(model_weights_path, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
            #self.params = ckpt.params
            self.state = {}
            self.model_config = ckpt.model_config
            self.task_config = ckpt.task_config

            #update params
            if self.config_file_path is not None:
                print(f'Loading params for member {self.case_name}')
                with open(self.config_file_path, 'rb') as f:
                    self.params = pickle.load(f)
            else:
                self.params = ckpt.params

    def load_gdas_data(self):
        """Load GDAS data."""
        #with open(gdas_data_path, "rb") as f:
        #    self.current_batch = xarray.load_dataset(f).compute()
        #warning: self.current_batch = xarray.load_dataset(self.gdas_data_path).compute()
        self.current_batch = xarray.load_dataset(self.gdas_data_path, decode_timedelta=True).compute()
        self.dates =  pd.to_datetime(self.current_batch.datetime.values)
        
        if (self.forecast_length + 2) > len(self.current_batch['time']):
            print('Updating batch dataset to account for forecast length')
            
            diff = int(self.forecast_length + 2 - len(self.current_batch['time']))
            ds = self.current_batch

            # time and datetime update
            curr_time_range = ds['time'].values.astype('timedelta64[ns]')
            new_time_range = (np.arange(len(curr_time_range) + diff) * np.timedelta64(6, 'h')).astype('timedelta64[ns]')
            ds = ds.reindex(time = new_time_range)
            curr_datetime_range = ds['datetime'][0].values.astype('datetime64[ns]')
            new_datetime_range = curr_datetime_range[0] + np.arange(len(curr_time_range) + diff) * np.timedelta64(6, 'h')
            ds['datetime'][0]= new_datetime_range

            self.current_batch = ds
            print('batch dataset updated')
            
        
    def extract_inputs_targets_forcings(self):
        """Extract inputs, targets, and forcings from the loaded data."""
        self.inputs, self.targets, self.forcings = data_utils.extract_inputs_targets_forcings(
            self.current_batch, target_lead_times=slice("6h", f"{self.forecast_length*6}h"), **dataclasses.asdict(self.task_config)
        )

    def load_normalization_stats(self):
        """Load normalization stats."""
        
        diffs_stddev_path = f"{self.pretrained_model_path}/stats/diffs_stddev_by_level.nc"
        mean_path = f"{self.pretrained_model_path}/stats/mean_by_level.nc"
        stddev_path = f"{self.pretrained_model_path}/stats/stddev_by_level.nc"
        with open(diffs_stddev_path, "rb") as f:
            self.diffs_stddev_by_level = xarray.load_dataset(f).compute()
        with open(mean_path, "rb") as f:
            self.mean_by_level = xarray.load_dataset(f).compute()
        with open(stddev_path, "rb") as f:
            self.stddev_by_level = xarray.load_dataset(f).compute()
    
    # Jax doesn't seem to like passing configs as args through the jit. Passing it
    # in via partial (instead of capture by closure) forces jax to invalidate the
    # jit cache if you change configs.
    def _with_configs(self, fn):
        return functools.partial(fn, model_config=self.model_config, task_config=self.task_config,)

    # Always pass params and state, so the usage below are simpler
    def _with_params(self, fn):
        return functools.partial(fn, params=self.params, state=self.state)

    # Deepmind models aren't stateful, so the state is always empty, so just return the
    # predictions. This is requiredy by the rollout code, and generally simpler.
    @staticmethod
    def _drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    def load_model(self):
        def construct_wrapped_graphcast(model_config, task_config):
            """Constructs and wraps the GraphCast Predictor."""
            # Deeper one-step predictor.
            predictor = graphcast.GraphCast(model_config, task_config)

            # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
            # from/to float32 to/from BFloat16.
            predictor = casting.Bfloat16Cast(predictor)

            # Applying normalization to the inputs/targets.
            predictor = normalization.InputsAndResiduals(
                predictor, 
                diffs_stddev_by_level=self.diffs_stddev_by_level, 
                mean_by_level=self.mean_by_level, 
                stddev_by_level=self.stddev_by_level,
            )

            # Wraps everything so the one-step model can produce trajectories.
            predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True,)
            return predictor

        @hk.transform_with_state
        def run_forward(model_config, task_config, inputs, targets_template, forcings,):
            predictor = construct_wrapped_graphcast(model_config, task_config)
            return predictor(inputs, targets_template=targets_template, forcings=forcings,)
        
        t0 = time()
        jax.jit(self._with_configs(run_forward.init))
        self.model = self._drop_state(self._with_params(jax.jit(self._with_configs(run_forward.apply))))
        elapsed_time = time() - t0
        logging.info(f"Elapsed time for compiling the model: {elapsed_time} seconds")
        logging.flush()
    
 
    def get_predictions(self):
        """Run GraphCast and save forecasts to a NetCDF file."""

        print (f"start running GraphCast for {self.forecast_length} steps --> {self.forecast_length*6} hours.")
        self.load_model()
           
        # Call and save f000 in grib2
        ds = self.current_batch
        ds = ds.drop_vars(['geopotential_at_surface','land_sea_mask', 'total_precipitation_6hr'])
        for var in ds.data_vars:
            if 'long_name' in ds[var].attrs:
                del ds[var].attrs['long_name']
        ds = ds.isel(time=slice(1, 2))
        ds['time'] = ds['time'] - pd.Timedelta(hours=6)

        try:
            self.writer.submit(ds, self.output_dir)

            # Rollout on the GPU. The main thread only drives the generator, which queues
            # each GPU step and then waits for it while building the next inputs. Each
            # finished step is handed to a host thread that pulls it to the CPU, adds the
            # precip cumsum and queues it to the GRIB2 writer pool, so the main thread
            # can queue the next GPU step right away instead of the GPU idling during
            # the host work. A single host thread keeps lead times (and tp_cum) in order.

            # Per-step timing, in seconds (logged per step only with --monitor yes):
            #   gpu compute:  time the GPU spent computing the step (measured by GpuStepTimer)
            #   dispatch:     generator step (queue the GPU step, build the next inputs); building
            #                 the next inputs waits for the GPU, so this includes gpu compute
            #                 for every step but the last
            #   handoff wait: main thread blocked because the host thread is behind (GPU stalls)
            #   gpu wait:     host thread in device_get, copying the result to the host
            #   writer wait:  host thread blocked because all writer slots are busy
            #   host:         host thread precip cumsum and copying the data to shared memory for the pool
            totals = {"gpu compute": 0.0, "dispatch": 0.0, "handoff wait": 0.0,
                      "gpu wait": 0.0, "writer wait": 0.0, "host": 0.0}
            first_step = {}
            detailed = self.monitor is not None
            gpu_timer = GpuStepTimer() if detailed else None

            # The generator waits for the GPU while building the next step's inputs,
            # so the step must be timestamped right after the model call queues it,
            # not when next() returns.
            predictor_fn = self.model
            if detailed:
                launched = 0

                def predictor_fn(**kwargs):
                    nonlocal launched
                    predictions = self.model(**kwargs)
                    launched += 1
                    gpu_timer.launched(launched, predictions)
                    return predictions

            # Small bound: each queued step holds one prediction in GPU memory.
            host_queue = queue.Queue(maxsize=2)
            host_errors = []

            def host_worker():
                tp_cum = np.zeros_like(self.targets['total_precipitation_6hr'].isel(time=slice(0, 1)).values)
                try:
                    while True:
                        item = host_queue.get()
                        if item is None:
                            return
                        step, prediction_chunk, t_dispatch = item
                        del item
                        # Labelled ranges for the profiler timeline (no-ops unless a trace is recorded).
                        t0 = time()
                        with jax.profiler.TraceAnnotation(f"host step {step}: device_get"):
                            prediction = jax.device_get(prediction_chunk)
                        t_gpu = time() - t0
                        del prediction_chunk
                        t_compute = (gpu_timer.compute_time(step) or 0.0) if detailed else 0.0
                        t0 = time()
                        with jax.profiler.TraceAnnotation(f"host step {step}: precip cumsum"):
                            tp_cum = tp_cum + prediction['total_precipitation_6hr'].values
                            prediction['total_precipitation_cumsum'] = (['batch', 'time', 'lat', 'lon'], tp_cum)
                        with jax.profiler.TraceAnnotation(f"host step {step}: writer submit"):
                            t_writer = self.writer.submit(prediction, self.output_dir)
                        t_host = time() - t0 - t_writer
                        lead = int(prediction.time.dt.total_seconds().values.ravel()[0] // 3600)
                        del prediction
                        for k, v in (("gpu compute", t_compute), ("dispatch", t_dispatch),
                                     ("gpu wait", t_gpu), ("writer wait", t_writer), ("host", t_host)):
                            totals[k] += v
                        if step == 1:
                            first_step.update(dispatch=t_dispatch, compute=t_compute)
                        if detailed:
                            logging.info(f"Step f{lead:03d}: gpu compute {t_compute:.2f} s, dispatch {t_dispatch:.2f} s, "
                                         f"gpu wait {t_gpu:.2f} s, writer wait {t_writer:.2f} s, host {t_host:.2f} s")
                            logging.flush()
                except BaseException as e:
                    host_errors.append(e)
                    # Keep draining so the main thread never blocks on a full queue.
                    while host_queue.get() is not None:
                        pass

            chunks = rollout.chunked_prediction_generator(
                predictor_fn=predictor_fn,
                rng=jax.random.PRNGKey(0),
                inputs=self.inputs,
                targets_template=self.targets * np.nan,
                forcings=self.forcings,
            )
            if self.monitor:
                self.monitor.phase("rollout (GPU + writers)")
            host_thread = threading.Thread(target=host_worker, name="rollout-host", daemon=True)
            host_thread.start()
            t_rollout = time()
            step = 0
            try:
                while not host_errors:
                    t0 = time()
                    chunk = next(chunks, None)
                    t_dispatch = time() - t0
                    if chunk is None:
                        break
                    step += 1
                    t0 = time()
                    host_queue.put((step, chunk, t_dispatch))
                    totals["handoff wait"] += time() - t0
                    del chunk
            finally:
                host_queue.put(None)
                host_thread.join()
            if host_errors:
                raise host_errors[0]
            t_rollout = time() - t_rollout
            if not detailed:
                logging.info(f"Rollout loop: {t_rollout:.1f} s total for {step} steps")
            else:
                gpu_timer.close()
                logging.info(f"Rollout loop: {t_rollout:.1f} s total; "
                             + ", ".join(f"{k} {v:.1f} s" for k, v in totals.items()))
                # The first step's dispatch includes JIT compilation, so also report without it.
                rest_wall = t_rollout - first_step.get("dispatch", 0.0)
                rest_compute = totals["gpu compute"] - first_step.get("compute", 0.0)
                logging.info(f"Actual GPU running time: {totals['gpu compute']:.1f} s for {step} steps "
                             f"({totals['gpu compute'] / max(step, 1):.2f} s/step), GPU busy "
                             f"{100 * totals['gpu compute'] / max(t_rollout, 1e-9):.0f}% of the rollout loop, "
                             f"{100 * rest_compute / max(rest_wall, 1e-9):.0f}% excluding the first step (compilation)")
            logging.flush()
        except BaseException:
            self.writer.abort()
            raise

        if self.monitor:
            self.monitor.phase("GRIB2 writers drain")
        t0 = time()
        self.writer.close()
        logging.info(f"Elapsed time waiting for GRIB2 writers after rollout: {time() - t0} seconds")
        logging.flush()

    def start_grib2_writer(self):
        """Start the CPU GRIB2 writer pool (needs self.dates, i.e. after load_gdas_data)."""
        self.writer = AsyncGrib2Writer(
            self.dates[0][1],
            case_name=self.case_name,
            json_path=f'{self.pretrained_model_path}/tables',
            num_workers=self.grib_workers,
        )

    def upload_to_s3(self, keep_data):
        s3 = boto3.client('s3')
        
        # Extract date and time information from the input file name
        input_file_name = os.path.basename(self.gdas_data_path)
        
        date_start = input_file_name.find("date-")

        # Check if "date-" is found in the input_file_name
        if date_start != -1:
            date_start += len("date-")  # Move to the end of "date-"
            date = input_file_name[date_start:date_start + 8]  # Extract 8 characters as the date
        
            time_start = date_start + 8  # Move to the character after the date
            time = input_file_name[time_start:time_start + 2]  # Extract 2 characters as the time


        # Define S3 key paths for input and output files
        input_s3_key = f'graphcastgfs.{date}/{time}/input/{self.gdas_data_path}'

        # Upload input file to S3
        s3.upload_file(self.gdas_data_path, self.s3_bucket_name, input_s3_key)
        
        # Upload output files to S3
        # Iterate over all files in the local directory and upload each one to S3
        s3_prefix = f'graphcastgfs.{date}/{time}/forecasts_{self.num_pressure_levels}_levels'
        
        for root, dirs, files in os.walk(self.output_dir):

            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, self.output_dir)
                s3_path = os.path.join(s3_prefix, relative_path)
            
                # Upload the file
                s3.upload_file(local_path, self.s3_bucket_name, s3_path)

        print("Upload to s3 bucket completed.")

        # Delete local files if keep_data is False
        if not keep_data:
            # Remove forecast data from the specified directory
            print("Removing input and forecast data from the specified directory...")
            try:
                os.system(f"rm -rf {self.output_dir}")
                os.remove(self.gdas_data_path)
                print("Local input and output files deleted.")
            except Exception as e:
                print(f"Error removing input and forecast data: {str(e)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GraphCast model.")
    parser.add_argument("-i", "--input", help="input file path (including file name)", required=True)
    parser.add_argument("-w", "--weights", help="parent directory of the graphcast params and stats", required=True)
    parser.add_argument("-l", "--length", help="length of forecast (6-hourly), an integer number in range [1, 40]", required=True)
    parser.add_argument("-n", "--case_name", help="aigfs, or gefs member [aigec00, aigep01, ..., aigep30]", required=True)
    #parser.add_argument("-c", "--config", help="GC weight member file", required=True)
    parser.add_argument("-c", "--config", help="GC weight member file", default=None)
    parser.add_argument("-o", "--output", help="output directory", default=None)
    parser.add_argument("-p", "--pressure", help="number of pressure levels", default=13)
    parser.add_argument("-u", "--upload", help="upload input data as well as forecasts to noaa s3 bucket (yes or no)", default = "no")
    parser.add_argument("-k", "--keep", help="keep input and output after uploading to noaa s3 bucket (yes or no)", default = "no")
    parser.add_argument("-t", "--grib_workers", help="number of CPU processes writing GRIB2 (0 = available CPUs - 1)", default=0)
    parser.add_argument("-m", "--method", help="GRIB2 output method (grib2io)", default="grib2io")
    parser.add_argument("--monitor", help="yes: per-step GPU/CPU timing and memory report; no: total timing and peak memory only (yes or no)", default="no")
    
    print("GPU testing, Available devices:", jax.devices())

    logging.set_verbosity(logging.INFO)

    args = parser.parse_args()
    monitor = ResourceMonitor("setup (load model/data)") if args.monitor.lower() == "yes" else None
    runner = GraphCastModel(args.weights, args.input, args.case_name, args.config, args.output, int(args.pressure), int(args.length), int(args.grib_workers))
    
    t0 = time()
    runner.load_pretrained_model()
    elapsed_time = time() - t0
    logging.info(f"Elapsed time for loading model: {elapsed_time} seconds")
    logging.flush()

    t0 = time()
    runner.load_gdas_data()
    elapsed_time = time() - t0
    logging.info(f"Elapsed time for loading input data: {elapsed_time} seconds")
    logging.flush()

    # Start GRIB2 writer processes early so their startup overlaps with model setup.
    runner.start_grib2_writer()

    t0 = time()
    runner.extract_inputs_targets_forcings()
    elapsed_time = time() - t0
    logging.info(f"Elapsed time for extracting inputs, targets, and forcings: {elapsed_time} seconds")
    logging.flush()


    t0 = time()
    runner.load_normalization_stats()
    elapsed_time = time() - t0
    logging.info(f"Elapsed time for loading normalization stats: {elapsed_time} seconds")
    logging.flush()

    t0 = time()
    runner.monitor = monitor
    runner.get_predictions()
    elapsed_time = time() - t0
    logging.info(f"Elapsed time for running the model: {elapsed_time} seconds")
    logging.flush()

    # GPU memory is still held by this process here, so nvidia-smi can see it.
    if monitor:
        monitor.report(gpu_stats=gpu_memory_stats(), gpu_smi=nvidia_smi_used(os.getpid()))
    else:
        basic_memory_summary()
    
    upload_data = args.upload.lower() == "yes"
    keep_data = args.keep.lower() == "yes"
    
    if upload_data:
        runner.upload_to_s3(keep_data)
