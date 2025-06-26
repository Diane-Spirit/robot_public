import psutil
import matplotlib.pyplot as plt
from jtop import jtop  # jetson-stats

class PerformanceMonitor:
    def __init__(self):
        self.fps = []
        self.time_broadcast = []
        self.cpu_usage = []
        self.gpu_usage = []
        self.ram_usage = []
        self.compression_ratio = []

    def record(self, start_time, end_time, time_broadcast,raw_size, encoded_size, jtop_stats):
        # Calcola FPS
        elapsed_time = end_time - start_time
        self.fps.append(1.0 / elapsed_time if elapsed_time > 0 else 0)

        # Usa psutil per monitorare CPU e RAM
        self.cpu_usage.append(psutil.cpu_percent())
        self.ram_usage.append(psutil.virtual_memory().percent)
        self.time_broadcast.append(time_broadcast)

        # Usa jetson-stats per monitorare la GPU
        if "GPU" in jtop_stats:
            self.gpu_usage.append(jtop_stats["GPU"])
        else:
            self.gpu_usage.append(0)

        # Calcola rapporto di compressione
        self.compression_ratio.append(raw_size / encoded_size if encoded_size > 0 else 0)

    def plot(self, filename="performance_plot.png", window_size=200):
        # Grafici delle performance
        fps = self.fps[-window_size:]
        cpu_usage = self.cpu_usage[-window_size:]
        gpu_usage = self.gpu_usage[-window_size:]
        ram_usage = self.ram_usage[-window_size:]
        compression_ratio = self.compression_ratio[-window_size:]
        broadcast = self.time_broadcast[-window_size:]
        fig, axs = plt.subplots(6, 1, figsize=(10, 15))

        axs[0].plot(fps, label="FPS", color="blue")
        axs[0].set_title("FPS")
        axs[0].set_ylabel("Frames/sec")
        axs[0].legend()

        axs[1].plot(broadcast, label="Broadcast", color="blue")
        axs[1].set_title("Broadcast")
        axs[1].set_ylabel("msec")
        axs[1].legend()

        axs[2].plot(cpu_usage, label="CPU Usage", color="orange")
        axs[2].set_title("CPU Usage")
        axs[2].set_ylabel("%")
        axs[2].legend()

        axs[3].plot(gpu_usage, label="GPU Usage", color="green")
        axs[3].set_title("GPU Usage")
        axs[3].set_ylabel("%")
        axs[3].legend()

        axs[4].plot(ram_usage, label="RAM Usage", color="red")
        axs[4].set_title("RAM Usage")
        axs[4].set_ylabel("%")
        axs[4].legend()

        axs[5].plot(compression_ratio, label="Compression Ratio", color="purple")
        axs[5].set_title("Compression Ratio")
        axs[5].set_ylabel("Raw/Encoded")
        axs[5].legend()

        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)