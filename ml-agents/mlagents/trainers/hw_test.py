import sys

import GPUtil
import psutil



def get_hardware_stats(interval_seconds : float = 0.1 ) -> dict:


    stats: dict = {
        'cpu_frequency_mhz': None,
        'cpu_usage_percent': None,
        'gpu_usage_percent': None,
        'vram_usage_mb': None,
        'ram_usage_mb': None,
    }

    stats['cpu_frequency_mhz'] = psutil.cpu_freq()
    stats['cpu_usage_percent'] = psutil.cpu_percent(interval = interval_seconds)
    stats['ram_usage_mb'] = round(psutil.virtual_memory().used / 1024 / 1024, 0)

    nvidia = False

    try:

        gpus = GPUtil.getGPUs()

        if gpus:
         gpu = gpus[0]

         stats['gpu_usage_percent'] = gpu.load * 100
         stats['vram_usage_mb'] = gpu.memoryUsed
         nvidia = True
    except:
         pass

    if not nvidia:
         stats['gpu_usage_percent'] = -1
         stats['vram_usage_mb'] = -1

   # print(stats)
    return stats

if __name__ == '__main__':

    get_hardware_stats()
