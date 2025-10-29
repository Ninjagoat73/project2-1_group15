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
            data["OS"] = f"{uname.system} 11"
        else:
            data["OS"] = f"{uname.system} 10"

        data["Version"] = uname.version
        data["Machine"] = uname.machine
    elif system == "Linux":
        data["OS"] = f"{uname.system} {uname.release}"
        data["Version"] = uname.version
        data["Machine"] = uname.machine
    elif system == "Darwin":
        data["OS"] = "macOS"
        data["Version"] = uname.version
        data["Machine"] = uname.machine
    return data



def collect_cpu_info(system):
    data = {}

    data["CPU Name"] = get_cpu_name(system)
    data["Physical Cores"] = psutil.cpu_count(logical=False)
    data["Total Cores"] = psutil.cpu_count(logical=True)
    freq = psutil.cpu_freq()
    if freq:
        data["Max Frequency (MHz)"] = f"{freq.max:.2f}"
        data["Min Frequency (MHz)"] = f"{freq.min:.2f}"
    return data

def collect_memory_info():
    data = {}
    mem = psutil.virtual_memory()
    data["Total RAM"] = get_size(mem.total)
    return data


def collect_gpu_info(system):
    data = {}

    if system == "Windows":
        try:
            if GPUtil:
                gpus = GPUtil.getGPUs()
                if gpus:
                    for i, gpu in enumerate(gpus):
                        data[f"GPU {i} Name"] = gpu.name
                        data[f"GPU {i} Total Memory (MB)"] = gpu.memoryTotal
                    return data

            import wmi
            c = wmi.WMI()
            for i, gpu in enumerate(c.Win32_VideoController()):
                data[f"GPU {i} Name"] = gpu.Name
                data[f"GPU {i} Driver Version"] = gpu.DriverVersion
                try:
                    vram_gb = int(gpu.AdapterRAM) / 1024**3
                    data[f"GPU {i} VRAM (GB)"] = round(vram_gb, 2)
                except Exception:
                    data[f"GPU {i} VRAM (GB)"] = "Unknown"
        except Exception as e:
            data["GPU Error"] = str(e)

    elif system == "Linux":
        try:
            import pyamdgpuinfo
            if pyamdgpuinfo.detect_gpus() > 0:
                for i in range(pyamdgpuinfo.detect_gpus()):
                    data[f"GPU {i} Name"] = pyamdgpuinfo.get_gpu_name(i)
                    data[f"GPU {i} VRAM Total (MB)"] = round(pyamdgpuinfo.get_vram_size(i) / 1024**2, 2)
                    data[f"GPU {i} Core Clock (MHz)"] = pyamdgpuinfo.get_gpu_clock(i)
                    data[f"GPU {i} Memory Clock (MHz)"] = pyamdgpuinfo.get_mem_clock(i)
                return data
        except ImportError:
            pass

        if GPUtil:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                data[f"GPU {i} Name"] = gpu.name
                data[f"GPU {i} Total Memory (MB)"] = gpu.memoryTotal

        else:
            data["GPU"] = "No GPU detected"

    elif system == "Darwin":
        try:
            out = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], text=True
            )
            gpus = [line.strip() for line in out.split("\n") if "Chipset Model" in line or "VRAM" in line]
            for i, info in enumerate(gpus):
                data[f"GPU {i} Info"] = info
        except Exception:
            data["GPU"] = "macOS GPU detection failed"

    else:
        data["GPU"] = "Unsupported OS"

    return data

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

export_to_csv()
