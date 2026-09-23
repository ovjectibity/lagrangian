import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
from cupti import cupti
from cupti import pm_sampling
from cupti import profiler_host


METRICS_OUTPUT_PATH = Path("pm_sampling_metrics.jsonl")
CONFIGURED_METRICS = [
    "gr__cycles_elapsed.max",
    "sm__cycles_elapsed.sum",
]


def full_metric_name(base_metric, sub_metric):
    if sub_metric == base_metric or sub_metric.startswith(f"{base_metric}."):
        return sub_metric
    if sub_metric.startswith("."):
        return f"{base_metric}{sub_metric}"
    return f"{base_metric}.{sub_metric}"


def get_pm_sampling_metrics(chip_name):
    host = profiler_host.ProfilerHost(
        chip_name, profiler_host.ProfilerType.PM_SAMPLING
    )
    host.initialize()

    try:
        metrics = {}
        for metric_type in (
            profiler_host.MetricType.COUNTER,
            profiler_host.MetricType.RATIO,
            profiler_host.MetricType.THROUGHPUT,
        ):
            base_metrics = sorted(host.get_base_metrics(metric_type))
            sub_metrics = {
                metric: sorted(host.get_sub_metrics(metric, metric_type))
                for metric in base_metrics
            }
            metrics[metric_type.name.lower()] = {
                "base_metrics": base_metrics,
                "metrics": sorted(
                    {
                        full_metric_name(metric, sub_metric)
                        for metric, sub_metric_names in sub_metrics.items()
                        for sub_metric in sub_metric_names
                    }
                ),
                "sub_metrics": sub_metrics,
            }

        configured_metric_passes = {}
        for metric in CONFIGURED_METRICS:
            try:
                config_image = host.create_config_image([metric])
                configured_metric_passes[metric] = (
                    profiler_host.get_num_of_passes(config_image)
                )
            except ValueError:
                configured_metric_passes[metric] = None

        return {
            "configured_metric_passes": configured_metric_passes,
            "metric_types": metrics,
        }
    finally:
        host.deinitialize()

@jax.jit
def matmulk(a,b): 
    c = a @ b
    return c

def setup_cupti(): 
    def func_buffer_requested():
        buffer_size = 8 * 1024 * 1024  # 8MB buffer
        max_num_records = 0
        return buffer_size, max_num_records

    def func_buffer_completed(activities: list):
        for activity in activities:
            if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                print(f"kernel name = {activity.name}")
                print(f"kernel duration (ns) = {activity.end - activity.start}")
    
    #Step 1: Register CUPTI callbacks
    cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)

    #Step 2: Enable CUPTI Activity Collection
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)

    collector = pm_sampling.Collector(device_index=0)

    collector.enable()

    collector.configure(
        metrics=[
            "gr__cycles_elapsed.max",
            "sm__cycles_elapsed.sum",
            # "tpc__warps_inactive_sm_idle_realtime.min.pct_of_peak_sustained_elapsed",
            # "tpc__warps_inactive_sm_idle_realtime.avg.pct_of_peak_sustained_elapsed",
            # "tpc__warps_inactive_sm_idle_realtime.max.pct_of_peak_sustained_elapsed",
            # "sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed",
            # "sm__pipe_tensor_cycles_active_realtime.min.pct_of_peak_sustained_elapsed",
            # "sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed",
            # "sm__pipe_tensor_cycles_active_realtime.max.pct_of_peak_sustained_elapsed"
        ],
        hardware_buffer_size=512 * 1024 * 1024,
        sampling_interval=10000,
        # trigger_mode=pm_sampling.TriggerMode.GPU_TIME_INTERVAL,
        trigger_mode=pm_sampling.TriggerMode.GPU_SYSCLK_INTERVAL,
    )

    return collector


def main():
    chip_names = profiler_host.get_supported_chips()
    with METRICS_OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for chip_name in chip_names:
            print("Printing the PM sampling metrics for: " + chip_name)
            metrics = get_pm_sampling_metrics(chip_name)
            output_file.write(
                json.dumps(
                    {"chip_name": chip_name, **metrics}, sort_keys=True
                )
                + "\n"
            )

    print(f"PM sampling metrics saved to {METRICS_OUTPUT_PATH}")

    i = 50
    j = 50
    k = 50
    key = random.key(42)
    a = random.uniform(key, shape=(i, j))
    b = random.uniform(key, shape=(j, k))

    # collector = setup_cupti()
    # collector.start() 

    # c = matmulk(a,b).block_until_ready()
    
    # collector.stop() 

    # ##Print up the sampled metrics: 
    # counter_data = collector.decode(max_samples=200)
    # print(f"Number of completed samples: {counter_data.num_completed_samples}")
    # for sample_idx, sample in enumerate(counter_data):
    #     print(
    #         f"Sample {sample_idx}: "
    #         f"start={sample.start_timestamp} end={sample.end_timestamp}"
    #     )
    #     for metric_name, value in zip(counter_data.metrics, sample.metric_values):
    #         print(f"  {metric_name}: {value}")

    # collector.disable()
    # cupti.activity_flush_all(1)
    # cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
    
    return

if __name__ == "__main__":
    main()
