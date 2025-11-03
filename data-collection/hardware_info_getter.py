import csv
import platform
import subprocess
from platform import uname
import psutil
import GPUtil
import sys

def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def get_cpu_name(system):
    if system == "Windows":
        try:
            import wmi
            c = wmi.WMI()
            return c.Win32_Processor()[0].Name.strip()
        except Exception:
            return platform.processor()
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.strip().split(": ")[1]
        except Exception:
            return platform.processor()
    elif system == "Darwin":  # macOS
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).strip().decode()
        except Exception:
            return platform.processor()
    return platform.processor()

def collect_system_info(system):
    data = {}
    uname = platform.uname()
    if system == "Windows":

        if sys.getwindowsversion().build >= 22000:
            data["os"] = f"{uname.system} 11"
        else:
            data["os"] = f"{uname.system} 10"

        #data["Version"] = uname.version
        #data["Machine"] = uname.machine
    elif system == "Linux":
        data["os"] = f"{uname.system} {uname.release}"
        #data["Version"] = uname.version
        #data["Machine"] = uname.machine
    elif system == "Darwin":
        data["os"] = "macOS"
        #data["Version"] = uname.version
        #data["Machine"] = uname.machine
    return data



def collect_cpu_info(system):
    data = {}

    data["cpu"] = get_cpu_name(system)
    data["physical_cores"] = psutil.cpu_count(logical=False)
    #data["Total Cores"] = psutil.cpu_count(logical=True)
    freq = psutil.cpu_freq()
    return data

def collect_memory_info():
    data = {}
    mem = psutil.virtual_memory()
    data["system_ram_gb"] = mem.total
    return data


def collect_gpu_info(system):
    data = {}

    if system == "Windows":
        try:
            if GPUtil:
                gpus = GPUtil.getGPUs()
                if gpus:
                    for i, gpu in enumerate(gpus):
                        data["gpu"] = gpu.name
                        data["GPU_ram_mb"] = gpu.memoryTotal
                    return data

            import wmi
            c = wmi.WMI()
            for i, gpu in enumerate(c.Win32_VideoController()):
                data["gpu"] = gpu.Name
                try:
                    vram_gb = int(gpu.AdapterRAM) / 1024**3
                    data["GPU_ram_mb"] = round(vram_gb, 2)
                except Exception:
                    data["GPU_ram_mb"] = "Unknown"
        except Exception as e:
            data["GPU Error"] = str(e)

    elif system == "Linux":
        try:
            import pyamdgpuinfo
            if pyamdgpuinfo.detect_gpus() > 0:
                for i in range(pyamdgpuinfo.detect_gpus()):
                    data[f"gpu"] = pyamdgpuinfo.get_gpu_name(i)
                    data[f"GPU_ram_mb"] = round(pyamdgpuinfo.get_vram_size(i) / 1024**2, 2)
                return data
        except ImportError:
            pass

        if GPUtil:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                data[f"gpu"] = gpu.name
                data[f"GPU_ram_mb"] = gpu.memoryTotal

        else:
            data["gpu"] = "No GPU detected"

    elif system == "Darwin":
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], text=True
            )
            gpus = [line.strip() for line in out.split("\n") if "Chipset Model" in line or "VRAM" in line]
            for i, info in enumerate(gpus):
                data[f"gpu"] = info
        except Exception:
            data["gpu"] = "macOS GPU detection failed"

    else:
        data["gpu"] = "Unsupported OS"

    return data

def get_all_data():
    system = platform.system()
    all_data = {}
    all_data.update(collect_system_info(system))
    all_data.update(collect_cpu_info(system))
    all_data.update(collect_memory_info())
    all_data.update(collect_gpu_info(system))

    return all_data

def export_to_csv(filename="hardware_info.csv"):
    system = platform.system()
    all_data = {}
    all_data.update(collect_system_info(system))
    all_data.update(collect_cpu_info(system))
    all_data.update(collect_memory_info())
    all_data.update(collect_gpu_info(system))


    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Component", "Value"])
        for key, value in all_data.items():
            writer.writerow([key, value])

    print(f"Hardware information exported to {filename}")

if __name__ == "__main__":
    export_to_csv()
