import os
import json
import csv
import logging
import subprocess
import numpy as np

from smoke.utils.miscellaneous import mkdir
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.utils import detection_name_to_rel_attributes


ID_TYPE_CONVERSION = {
    0: 'car',
    1: 'truck',
    2: 'bus',
    3: 'trailer',
    4: 'construction_vehicle',
    5: 'pedestrian',
    6: 'bicycle',
    7: 'traffic_cone',
    8: 'barrier'
}

def euler_to_quaternion(yaw, pitch, roll):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def random_attr(name: str) -> str:
            """
            This is the most straight-forward way to generate a random attribute.
            Not currently used b/c we want the test fixture to be back-wards compatible.
            """
            # Get relevant attributes.
            rel_attributes = detection_name_to_rel_attributes(name)

            if len(rel_attributes) == 0:
                # Empty string for classes without attributes.
                return ''
            else:
                # Pick a random attribute otherwise.
                return rel_attributes[np.random.randint(0, len(rel_attributes))]

def nuScenes_evaluation(
        eval_type,
        dataset,
        predictions,
        output_folder,
):
    logger = logging.getLogger(__name__)
    if "detection" in eval_type:
        logger.info("performing nuScenes detection evaluation: ")
        do_nuScenes_detection_evaluation(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger
        )


def do_nuScenes_detection_evaluation(dataset,
                                  predictions,
                                  output_folder,
                                  logger
                                  ):
    mock_meta = {
                'use_camera': True,
                'use_lidar': False,
                'use_radar': False,
                'use_map': False,
                'use_external': False,
    }
    mock_results= {}
    for image_token, prediction in predictions.items():
        sample_res = []
        for p in prediction:
            p = p.numpy()
            p = p.round(4)
            type = ID_TYPE_CONVERSION[int(p[0])]
            sample_res.append(
                {
                    'sample_token': image_token,
                    'translation': p[9:12].tolist(),
                    'size': p[6:9].tolist(),
                    'rotation': euler_to_quaternion(p[12], 0, 0),
                    'velocity': [0,0],
                    'detection_name': type,
                    'attribute_name': random_attr(type)
                })
        mock_results[image_token] = sample_res
    mock_submission = {
            'meta': mock_meta,
            'results': mock_results
    }

    logger.info("Evaluate on nuScenes dataset")
    output_file = output_folder + ".json"
    with open(output_file, 'w') as f:
        json.dump(mock_submission, f, indent=2)

    cfg = config_factory('detection_cvpr_2019')
    nusc_eval = DetectionEval(dataset.nusc, cfg, output_file, eval_set='val', output_dir=output_folder,
                                verbose=False)
    metrics, md_list = nusc_eval.evaluate()
    print(md_list)


