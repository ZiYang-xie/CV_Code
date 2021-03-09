import json
import os
import random
import shutil
import unittest
from typing import Dict

import numpy as np
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.constants import DETECTION_NAMES
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.detection.utils import category_to_detection_name, detection_name_to_rel_attributes
from nuscenes.utils.splits import create_splits_scenes


splits = create_splits_scenes()
print(splits["train"])
