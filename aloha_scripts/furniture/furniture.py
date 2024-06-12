import multiprocessing as mp
import time
from abc import ABC
from multiprocessing import shared_memory, Manager
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from furniture.parts.part import Part
from utils.detection import detection_loop
from robot_utils import Recorder, ImageRecorder


class Furniture(ABC):
    def __init__(self):
        self.parts: List[Part] = []
        self.num_parts = len(self.parts)

        self.tag_size = None

        # Multiprocessing for pose detection.
        self.detection_started = False

    def start_detection(self):
        self.ctx = mp.get_context("spawn")

        if self.detection_started:
            return

        self.shm = self.create_shared_memory()
        self.lock = self.ctx.Lock()
        manager = Manager()
        shared_dict = manager.dict() 
        # shared_dict["image_recorder"] = ImageRecorder(init_node=False)
        
        
        self.proc = self.ctx.Process(
            target=detection_loop,
            args=(
                self.parts,
                self.num_parts,
                self.tag_size,
                self.lock,
                self.shm,
                shared_dict,
            ),
            daemon=True,
        )
        self.proc.start()
        self.detection_started = True
        self._wait_detection_start()
        # return shared_dict["image_recorder"]

    def _wait_detection_start(self):
        max_wait = 20  # 20 seconds
        while True:
            start = time.time()
            while (time.time() - start) < max_wait:
                _, founds = self.get_parts_poses_founds()
                if founds.any():
                    # Heuristic to check whether the detection started. (At least one part is found.)
                    return
                time.sleep(0.03)

            input(
                "Could not find any furniture parts from the cameras\n"
                "Press enter after putting the furniture in the workspace."
            )

    def create_shared_memory(self):
        """Create shared memory to save the parts poses and images."""
        parts_poses = np.zeros(shape=(self.num_parts * 7,), dtype=np.float32)
        parts_poses_shm = shared_memory.SharedMemory(
            create=True, size=parts_poses.nbytes
        )
        parts_founds = np.zeros(shape=(self.num_parts,), dtype=bool)
        parts_founds_shm = shared_memory.SharedMemory(
                    create=True, size=parts_founds.nbytes
                )
        return (
            parts_poses_shm.name,
            parts_founds_shm.name,
        )

    def get_parts_poses(self):
        if not self.detection_started:
            raise Exception("First call `start_detection` to get part poses")
        with self.lock:
            return self.get_array()

    def get_parts_poses_founds(
        self,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        """Get parts poses and founds only."""
        parts_poses, founds = self.get_parts_poses()
        return parts_poses, founds

    def get_array(
            self,
    ) -> Tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.bool_],
    ]:
        """Get the shared memory of parts poses and images."""
        parts_poses_shm = shared_memory.SharedMemory(name=self.shm[0])
        parts_founds_shm = shared_memory.SharedMemory(name=self.shm[1])

        parts_poses = np.ndarray(
            shape=(self.num_parts * 7,), dtype=np.float32, buffer=parts_poses_shm.buf
        )
        parts_found = np.ndarray(
            shape=(self.num_parts,), dtype=bool, buffer=parts_founds_shm.buf
        )

        return (
            parts_poses.copy(),
            parts_found.copy(),
        )
