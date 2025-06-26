import os
import mmap

# Constants for shared memory sizes and file paths.
HEADER_SIZE = 3
DATA_MAX_SIZE = 1843200  # Maximum data size in bytes (e.g., 480x480x8 bytes).
SHARED_PC_SIZE = HEADER_SIZE + DATA_MAX_SIZE
SEMAPHORE_SIZE = 6

SHARED_PC = "/dev/shm/shared_pc"
SEMAPHORE_READ = "/dev/shm/semaphore_read"


def create_shared_memory() -> None:
    """
    Ensure that the shared memory files exist with the correct sizes.
    
    For each defined shared memory file, if the file does not exist, it is created and
    filled with zeros. If it exists, it is truncated to the expected size.
    """
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
    """
    Create (if necessary) and map the shared memory regions.
    
    Returns:
        A tuple containing:
          - shared_mem: mmap object for the shared picture buffer.
          - flag_read_mem: mmap object for the semaphore flag region.
    """
    # Ensure the shared memory files are created and sized correctly.
    create_shared_memory()

    # Open the files in read+write binary mode.
    shared_pc_file = open(SHARED_PC, "r+b")
    semaphore_read_file = open(SEMAPHORE_READ, "r+b")

    # Memory-map the files with shared access and write permissions.
    shared_mem = mmap.mmap(shared_pc_file.fileno(), SHARED_PC_SIZE,
                           mmap.MAP_SHARED, mmap.PROT_WRITE)
    flag_read_mem = mmap.mmap(semaphore_read_file.fileno(), SEMAPHORE_SIZE,
                              mmap.MAP_SHARED, mmap.PROT_WRITE)

    return shared_mem, flag_read_mem


class SharedMemoryManager:
    """
    Singleton class to manage shared memory mappings.
    
    This class ensures that a single instance is used to map and access the shared
    memory regions defined by SHARED_PC and SEMAPHORE_READ.
    """
    _instance = None

    def __init__(self) -> None:
        if SharedMemoryManager._instance is not None:
            raise Exception("SharedMemoryManager singleton instance already exists!")
        # Map the shared memory regions.
        self.shared_mem, self.flag_read_mem = map_shared_memory()
        SharedMemoryManager._instance = self

    @staticmethod
    def get_instance() -> "SharedMemoryManager":
        """
        Get the single instance of SharedMemoryManager.
        
        Returns:
            SharedMemoryManager: The singleton instance.
        """
        if SharedMemoryManager._instance is None:
            SharedMemoryManager()
        return SharedMemoryManager._instance