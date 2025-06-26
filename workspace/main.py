import argparse
import struct
import time
from typing import Any

import numpy as np
from encoding import Encoder, Encoding
from evaluate import PerformanceMonitor
from jtop import jtop  # jetson-stats
from shared_memory import SharedMemoryManager

import pyzed.sl as sl  # Import ZED SDK here since it can be delayed until needed


def online_mode(save_data: bool = False, save_plot: bool = False, save_path: str = "./saved_frames/test.npz") -> None:
    """
    Process online point cloud data from the ZED camera, encode and broadcast via shared memory.

    Args:
        save_data (bool): If True, save raw point cloud data to disk.
        save_plot (bool): If True, generate and save plots for analysis.
        save_path (str): File path for saving the raw data.
    """
    # Get the singleton shared memory manager instance.
    sm_manager = SharedMemoryManager.get_instance()

    # Initialize performance monitor.
    monitor = PerformanceMonitor()

    # Create and configure ZED camera.
    zed: sl.Camera = sl.Camera()
    init_params = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        coordinate_units=sl.UNIT.CENTIMETER,
        depth_maximum_distance=200000.0,
        camera_resolution=sl.RESOLUTION.HD1080,
        camera_fps=30,
        camera_image_flip=sl.FLIP_MODE.ON,
    )

    # Define a custom resolution (480x480 instead of HD1080 for performance).
    custom_resolution = sl.Resolution(480, 480)

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open Error: {repr(status)}")
        exit(1)

    runtime_params = sl.RuntimeParameters(confidence_threshold=100)
    point_cloud = sl.Mat()

    # Initialize the encoder with chosen encoding type.
    encoder = Encoder(custom_resolution, Encoding.DIANE_MULTINOMIAL_i16)
    first_iteration = True
    fps: float = 18.0

    # Start the Jetson performance monitor using context managers.
    with jtop() as jetson:
        # Set flag to indicate readiness of shared memory.
        sm_manager.flag_read_mem[0] = 1
        # Variables to hold bandwidth target.
        target_bw: int = 0  
        while True:
            start_time = time.time()

            # On iterations after the first, update the target bandwidth and squid flag.
            if not first_iteration:
                # Wait until flag_read_mem[0] is cleared.
                while sm_manager.flag_read_mem[0] == 1:
                    time.sleep(0.001)
                # Extract 4-byte bandwidth target from shared memory (little-endian unsigned int).
                target_bytes = sm_manager.flag_read_mem[2:6]
                target: int = struct.unpack('<I', target_bytes)[0]
                # Set target_bw based on retrieved target value.
                target_bw = target
                print("Measured target (Mbps):", target // 1000000)
                print("Filtered target (Mbps):", target_bw // 1000000)
                squid: bool = (sm_manager.flag_read_mem[1] == 1)
            else:
                # During the first iteration, use default values.
                target = 30 * 1000000  # 30 Mbps default target
                target_bw = target
                squid = False
                first_iteration = False

            # Grab a frame from the camera.
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # Retrieve the XYZRGBA point cloud with the custom resolution.
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, resolution=custom_resolution)
                numpy_pc = point_cloud.get_data()
                raw_size: int = numpy_pc.nbytes

                time_to_grab = time.time() - start_time

                if save_data:
                    np.savez(save_path, point_cloud=numpy_pc)

                # Pass bandwidth and squid flag to the encoder.
                encoded_bytes: bytes = encoder.encode(numpy_pc, bw=target_bw, fps=fps, squid=squid)
                encoded_size: int = len(encoded_bytes)

                time_to_encode = time.time() - start_time - time_to_grab

                # Pack the encoded size (first 3 bytes of 4-byte unsigned int) as header.
                header: bytes = struct.pack('<I', encoded_size)[:3]

                # Write header and encoded bytes to shared memory.
                sm_manager.shared_mem.seek(0)
                sm_manager.shared_mem.write(header + encoded_bytes)
                # Reset flag to indicate new frame is ready.
                sm_manager.flag_read_mem[0] = 1

                # Calculate remaining time to meet desired FPS.
                intergrab_time = (1 / fps) - time_to_grab - time_to_encode

                # Retrieve and log Jetson stats if needed.
                jtop_stats: Any = jetson.stats

                if intergrab_time > 0:
                    time.sleep(intergrab_time)
                else:
                    print("Too many FPS requested: camera grab cannot keep up.")

    # Close the camera after exiting the loop.
    zed.close()


def main() -> None:
    """
    Parse command line arguments and run the online mode.
    """
    parser = argparse.ArgumentParser(description="Online Point Cloud Encoding and Broadcasting")
    parser.add_argument("--save", action="store_true", help="Save raw data from the camera")
    parser.add_argument("--plot", action="store_true", help="Save performance plot")
    args = parser.parse_args()

    online_mode(save_data=args.save, save_plot=args.plot)


if __name__ == "__main__":
    main()