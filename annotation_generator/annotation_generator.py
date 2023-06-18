import os
import json
import shutil
import cv2
import numpy as np
import copy

path = "../data/all_gt_mask"
dest_filepath = "./annotation_1617181920.json"

info = {"year": 2022, "version": 1.0}

json_imgs = []
json_annot = []
license = [{"id": 0, "name": "Unknown", "url": ""}]
categories = [
    {"supercategory": "type", "id": 1, "name": "point_source"},
    {"supercategory": "type", "id": 2, "name": "diffused_source"},
    {"supercategory": "type", "id": 3, "name": "Unknown"},
]
"""
img_template = {"file_name" : ,
				"height" : ,
				"width" : ,
				"id" : file_name}

annot_template = {"image_id" : image_name,
				  "id" : annotation number in that image
				  "segmentation" : [],
				  "bbox" :
				  "category_id" : category}
"""


def getMask(_dir_path, all_files):
    for _file in all_files:
        file_path = os.path.join(_dir_path, _file)
        if _file.split(".")[-1] != "npy":
            img = cv2.imread(file_path)
        else:
            img = np.load(file_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = np.zeros((img_gray.shape))
        mask[img_gray > 0] = 255
        # find all the independent plumes
        plume_mask = []
        plume_bbox = []
        plume_cat = []

        _temp_patch = _file.split(".")[0]
        patch_name = f"{_temp_patch}"
        _temp_patch = _temp_patch.split("_")
        patch_id = f"{_temp_patch[-2]}_{_temp_patch[-1]}"
        file_name = _file[:18]
        contours, hier = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            # cv2.drawContours(img, contours, -1, (0,255,0), 3)
            # cv2.imwrite(f"{dest_seg}/{_file.split('.')[0]}.png", img)
            for _contour in contours:
                if len(_contour) < 8:
                    continue  # plume too small
                sqz_contour = _contour.squeeze(axis=1)  # (width, height) OR (y, x)
                # bbox : (top, left, bottom, right) OR (x.min, ymin, x.max, y.max)
                plume_bbox.append(
                    [
                        int(sqz_contour[:, 1].min()),
                        int(sqz_contour[:, 0].min()),
                        int(sqz_contour[:, 1].max()),
                        int(sqz_contour[:, 0].max()),
                    ]
                )
                plume_mask.append(sqz_contour.tolist())
                plume_cat.append(1)
            annot_template = {
                "patch_name": patch_name,
                "patch_id": patch_id,
                "image_id": file_name,
                "segmentation": plume_mask,
                "bbox": plume_bbox,
                "category_id": plume_cat,
            }
            json_annot.append(annot_template)

    # create json data
    json_data = {"info": info, "licenses": license, "annotations": json_annot, "categories": categories}
    # write data to json file
    if os.path.isfile(dest_filepath):
        os.remove(dest_filepath)
    with open(dest_filepath, mode="w") as f:
        f.write(json.dumps(json_data, indent=4))


def main(path):
    all_dirs = os.listdir(path)
    all_dir_path = [os.path.join(path, _dir_name) for _dir_name in all_dirs]
    for _dir_path in all_dir_path:
        sub_dir_files = os.listdir(_dir_path)
        print("Computing annotations for file", _dir_path)
        getMask(_dir_path, sub_dir_files)


if __name__ == "__main__":
    main(path)
