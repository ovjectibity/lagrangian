import jax
import jax.numpy as jnp
import jax.random as random
from cupti import cupti
from cupti import pm_sampling

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
    i = 50
    j = 50
    k = 50
    key = random.key(42)
    a = random.uniform(key, shape=(i, j))
    b = random.uniform(key, shape=(j, k))

    collector = setup_cupti()
    collector.start() 

    c = matmulk(a,b).block_until_ready()
    
    collector.stop() 

    ##Print up the sampled metrics: 
    counter_data = collector.decode(max_samples=200)
    print(f"Number of completed samples: {counter_data.num_completed_samples}")
    for sample_idx, sample in enumerate(counter_data):
        print(
            f"Sample {sample_idx}: "
            f"start={sample.start_timestamp} end={sample.end_timestamp}"
        )
        for metric_name, value in zip(counter_data.metrics, sample.metric_values):
            print(f"  {metric_name}: {value}")

    collector.disable()
    cupti.activity_flush_all(1)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
    
    return

if __name__ == "__main__":
    main()