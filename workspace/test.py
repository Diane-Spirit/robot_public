import os
import mmap
import time
import numpy as np
import struct 

HEADER_SIZE = 3
DATA_MAX_SIZE = 1843200  
SHARED_PC_SIZE = HEADER_SIZE + DATA_MAX_SIZE
SEMAPHORE_SIZE = 6

SHARED_PC = "/dev/shm/shared_pc"
SEMAPHORE_READ = "/dev/shm/semaphore_read"

def create_shared_memory():
    for path, size in [
        (SHARED_PC, SHARED_PC_SIZE),
        (SEMAPHORE_READ, SEMAPHORE_SIZE),
    ]:
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(b"\x00" * size)
        else:
            with open(path, "r+b") as f:
                f.truncate(size)

def map_shared_memory():
    create_shared_memory()  
    
    shared_pc = open(SHARED_PC, "r+b")
    semaphore_read = open(SEMAPHORE_READ, "r+b")

    shared_mem = mmap.mmap(shared_pc.fileno(), SHARED_PC_SIZE, 
                         mmap.MAP_SHARED, mmap.PROT_WRITE)
    flag_read_mem = mmap.mmap(semaphore_read.fileno(), SEMAPHORE_SIZE, 
                            mmap.MAP_SHARED, mmap.PROT_WRITE)


    return shared_mem, flag_read_mem


shared_mem, flag_read_mem = map_shared_memory()

target = 100 * 1000000  

fps = 18

try:
    count = 0
    while True:
        start_time = time.time()

        PSZ = 4 * 16 # bits per point
        MFS = target / fps # Maximum frame size in bits

        max_points = int(MFS / PSZ) 

        max_points = (max_points // 150) * 150

        data_size = min(max_points * 8, DATA_MAX_SIZE)
        #data_size = DATA_MAX_SIZE

        # print(f"Send {data_size} byte")

        squid = flag_read_mem[1] == 1

        if squid:
            print("Squid mode active")
            data = np.full(int(data_size), 0, dtype=np.uint8).tobytes()
        else:
            # print("Normal mode active")
            data = np.random.randint(0, 256, int(data_size), dtype=np.uint8).tobytes()
        
        header = struct.pack('<I', int(data_size))[:3]

        shared_mem.seek(0)
        shared_mem.write(header + data)

        flag_read_mem[0] = 1  


        
        
        while flag_read_mem[0] == 1:
            time.sleep(0.001)

        target_bytes = flag_read_mem[2:6]
        target = struct.unpack('<I', target_bytes)[0]  
        # print(f"BitRate: {target}")

        elapsed_time = time.time() - start_time
        
        time.sleep(1/fps)


except KeyboardInterrupt:
    print("Closing...")
finally:
    shared_mem.close()
    flag_read_mem.close()
