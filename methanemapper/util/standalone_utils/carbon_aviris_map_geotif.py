import rasterio
from affine import Affine
import numpy as np
import cv2
from pyproj import Transformer
from pyproj import crs as crs_type
import argparse
import glob
from joblib import Parallel, delayed

import os
import sys
import spectral as spy
import spectral.io.envi as envi

sys.path.append("./../")
from HSI2RGB import *


def imageObj(hdr, img):
    "Create an object of the image corresponding to certain header"
    head = envi.read_envi_header(hdr)
    param = envi.gen_params(head)
    param.filename = img  # spectral data file corresponding to .hdr file

    if "bbl" in head:
        head["bbl"] = [float(b) for b in head["bbl"]]
        print(len(head["bbl"]))
        good_bands = np.nonzero(head["bbl"])[0]  # get good bands only for computation
        bad_bands = np.nonzero(np.array(head["bbl"]) == 0)[0]
        print("calculating target signature...")

    interleave = head["interleave"]
    if interleave == "bip" or interleave == "BIP":
        print("it is a bip")
        from spectral.io.bipfile import BipFile

        img_obj = BipFile(param, head)

    if interleave == "bil" or interleave == "BIL":
        print("It is a bil file")
        from spectral.io.bilfile import BilFile

        img_obj = BilFile(param, head)

    return img_obj


def rectifyLocFile(glt, img, b_val=0.0, output=None):
    if output is None:
        output = np.zeros((glt.shape[0], glt.shape[1], 2))
    if not np.array_equal(output.shape, [glt.shape[0], glt.shape[1], 2]):
        print("image dimension of output arrary do not match the GLT file")
    # Get absolute even if GLT has negative values
    glt_mag = np.absolute(glt)
    # GLT value of zero means no data, extract this because python has zero-indexing.
    glt_mask = np.all(glt_mag == 0, axis=2)
    output[glt_mask] = b_val
    glt_mag[glt_mag > (img.shape[0] - 1)] = 0
    # now check the lookup and fill in the location, -1 to map to zero-indexing
    output[~glt_mask] = img[glt_mag[~glt_mask, 1] - 1, glt_mag[~glt_mask, 0] - 1, 0:2]

    return output


def get_coordinates_and_data_from_tif(src):
    # Function that generates EPSG 3857 coords for each pixel in tif -> stripped down version of read_tif
    if isinstance(src.transform, Affine):
        transform = src.transform
    else:
        transform = src.affine

    N = src.width
    M = src.height
    dx = transform.a
    dy = transform.e
    minx = transform.c
    maxy = transform.f

    data_in = src.read()
    if dy < 0:
        print(f"Origin in top left, dy value: {dy}")
        dy = -dy

    # Generate X and Y grid locations
    xdata = minx + dx / 2 + dx * np.arange(N)  # longitudes
    ydata = maxy - dy / 2 - dy * np.arange(M - 1, -1, -1)  # latitudes

    extent = [xdata[0], xdata[-1], ydata[0], ydata[-1]]

    return xdata, ydata, extent, data_in[0:3]


def get_gt_layer_idxs_of_gt_patch_pxls(rect_loc: np.ma.MaskedArray, pxl_coords, scope_corners=None):
    pxl_locs = np.zeros(pxl_coords.shape, dtype=np.uint32)
    rect_loc_scope = rect_loc

    for i in range(pxl_locs.shape[0]):
        pxl_coord_finder = rect_loc_scope - pxl_coords[i].reshape(1, 1, 2)
        pxl_coord_finder = np.linalg.norm(pxl_coord_finder, axis=2)
        pxl_coord_finder = np.ma.masked_array(pxl_coord_finder, mask=rect_loc_scope.mask[:, :, 0])
        pxl_idx = np.unravel_index(np.ma.argmin(pxl_coord_finder, fill_value=9999), pxl_coord_finder.shape)
        pxl_locs[i] = np.array(pxl_idx, dtype=np.int32)  #  pxl_loc[j,i,0] -> y-vals, pxl_loc[j,i,1] -> x-vals

    return pxl_locs


def computeHomography(src_pts, dst_pts, patch_data=None, dst_img=None):
    dst_h, dst_w, dst_ch = dst_img.shape
    if patch_data.shape[0] > 0:
        patch_data = np.transpose(patch_data, (1, 2, 0))
    h, status = cv2.findHomography(src_pts, dst_pts)
    img_out = cv2.warpPerspective(patch_data, h, (dst_w, dst_h))

    return img_out


def getCorners(h, w, box_size):
    # Create a box b_size x b_size around center coordinate, use those corners pixels
    c_x, c_y, b_size = h // 2, w // 2, box_size // 2
    t, b, l, r = c_x - b_size, c_x + b_size, c_y - b_size, c_y + b_size
    all_x, all_y = [], []
    all_x1, all_y1 = [], []
    for i in (t, b):
        for j in (l, r):
            all_x1.append(i)
            all_y1.append(j)
    all_x1.append(c_x)
    all_y1.append(c_y)

    all_x = [0, 0, h - 1, h - 1]
    all_y = [0, w - 1, 0, w - 1]

    return (np.array(all_x), np.array(all_y), np.array(all_x1), np.array(all_y1))


def generate_gt_layer_idxs_of_gt_patch(rect_loc, pxl_coords):
    # Generate mask of rect_loc
    rect_loc_w_mask = np.ma.masked_equal(rect_loc, 0)
    h, w, c = pxl_coords.shape
    # get corners of a box around center
    all_coords = getCorners(h, w, box_size=50)
    all_coord_vals = pxl_coords[(all_coords[2], all_coords[3])]
    # reading more pixels from to get better homogrphy matrix
    corner_pxl_idxs = get_gt_layer_idxs_of_gt_patch_pxls(rect_loc_w_mask, all_coord_vals)
    # compute homography matrix, format [width(y), height(x)]
    src_pts = np.stack((all_coords[3], all_coords[2]), axis=1)
    dst_pts = np.fliplr(corner_pxl_idxs)

    return (src_pts, dst_pts)


def main(tif_path, loc_hdr_path, glt_hdr_path, img_hdr_path, gt_mask_name, gt_clr_dir):
    gt_clr_path = os.path.join(gt_clr_dir, gt_mask_name)

    if os.path.isfile(gt_clr_path):
        print(gt_mask_name, "already exists")
        return None
    # -------------------------------------------------
    # 	  Read tif file and get patch coordinates
    # -------------------------------------------------
    raster = rasterio.open(tif_path)
    print("Tif Metadata:")
    for x in raster.meta:
        print(x, raster.meta[x])

    crs = str(raster.meta["crs"])

    # Get coordinates and raw patch data
    x, y, extent, patch_data = get_coordinates_and_data_from_tif(raster)
    print(f"Patch coordinates in {crs}: {extent}")

    # Convert min,max coords from EPSG 3857 to EPSG 4326
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # Start with last element in y and first element in x
    tif_coords = np.zeros([len(y), len(x), 2])
    tif_coords[:, :, 1] = tif_coords[:, :, 1] + np.expand_dims(y[::-1], axis=1)
    tif_coords[:, :, 0] = tif_coords[:, :, 0] + np.expand_dims(x, axis=0)

    #  Convert tif crs to epsg 4326
    for i in range(tif_coords.shape[0]):
        for j in range(tif_coords.shape[1]):
            tif_coords[i, j, 0], tif_coords[i, j, 1] = transformer.transform(tif_coords[i, j, 0], tif_coords[i, j, 1])

    patch_corners_longs = tif_coords[0, 0, 0], tif_coords[0, -1, 0], tif_coords[-1, 0, 0], tif_coords[-1, -1, 0]
    patch_corners_lats = tif_coords[0, 0, 1], tif_coords[0, -1, 1], tif_coords[-1, 0, 1], tif_coords[-1, -1, 1]
    print(
        f"Patch coordinates in EPSG:4326 (tl, tr, bl, br): {[(x,y) for x,y in zip(patch_corners_longs, patch_corners_lats)]}"
    )

    # ------------------------------------------------------------
    # 	  Load loc and glt file to compute orthocorrected loc
    # ------------------------------------------------------------
    try:
        loc_rw, loc_col, loc_ch = spy.envi.open(loc_hdr_path).shape
    except:
        return loc_hdr_path.split("/")[-2]

    glt_obj = imageObj(glt_hdr_path, f'{glt_hdr_path.split(".")[0]}')
    loc_obj = imageObj(loc_hdr_path, f'{loc_hdr_path.split(".")[0]}')
    img_obj = imageObj(img_hdr_path, f'{img_hdr_path.split(".")[0]}')

    glt_data = glt_obj.read_bands([0, 1])
    loc_data = loc_obj.read_subregion((0, loc_rw), (0, loc_col))
    viz_bands = np.arange(80).tolist()
    img_data = img_obj.read_bands(viz_bands)
    img_mask = np.ma.masked_less(img_data[:, :, 10], -9990.0).mask

    # Get color image of background
    rgb_img = createRGB(img_data[:, :, 0:80], mask=img_mask)
    rgb_img = np.uint8(rgb_img * 255)

    # Rectified loc matrix.
    rect_loc = rectifyLocFile(glt_data, loc_data)

    # ------------------------------------------------------------
    # 	  Get pixel indices of corresponding gt patches
    # ------------------------------------------------------------

    src_pts, dst_pts = generate_gt_layer_idxs_of_gt_patch(rect_loc, tif_coords)
    im_out = computeHomography(src_pts, dst_pts.astype(np.int64), patch_data, rect_loc)
    im_gry = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)
    try:
        im_clr = rgb_img + im_out
    except:
        print("size mismatch for file:", gt_mask_name)
        return None
    im_gry[im_gry > 10] = 255
    kernel = np.ones((5, 5), np.uint8)
    im_gry = cv2.dilate(im_gry, kernel, iterations=1)
    im_gry = cv2.erode(im_gry, kernel, iterations=1)
    tiff_image = np.concatenate((rgb_img, np.expand_dims(im_gry, axis=-1)), axis=2)

    tiff_h, tiff_w, tiff_c = tiff_image.shape
    tiff_meta = raster.meta
    tiff_meta.update(nodata=0, width=tiff_w, height=tiff_h, count=tiff_c, crs=crs_type.CRS.from_epsg(4326))

    with rasterio.open(gt_clr_path, "w", **tiff_meta) as dst:
        dst.write(np.transpose(tiff_image, (2, 0, 1)))

    print("saved file:", gt_mask_name)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--avaris_data_dir", help="Path to avaris sample data directory")
    parser.add_argument("-tif", "--tif_file_path", help="Path to tiff file")
    parser.add_argument("-t", "--test_file", help="name of test file")
    parser.add_argument("-gtd", "--gt_dir", help="Root dir of GT.")
    args = parser.parse_args()

    # Dev paths
    all_tifs = [args.test_file] if args.test_file else os.listdir(args.tif_file_path)
    all_tifs_path = [os.path.join(args.tif_file_path, _tif_name) for _tif_name in all_tifs]

    file_not_found = []

    def _call_main(_tif_path):
        aviris_dir_part = f'{_tif_path.split("/")[-1][0:18]}*'
        gt_mask_name = f'{_tif_path.split("/")[-1].split(".")[0]}.tif'
        try:
            aviris_dir_path = glob.glob(os.path.join(args.avaris_data_dir, aviris_dir_part))[0]
        except:
            print("file not found", aviris_dir_part)
            return
        aviris_dir = aviris_dir_path.split("/")[-1]
        aviris_loc_hdr_path = os.path.join(aviris_dir_path, f"{aviris_dir}_loc.hdr")
        aviris_glt_hdr_path = os.path.join(aviris_dir_path, f"{aviris_dir}_glt.hdr")
        aviris_img_hdr_path = os.path.join(aviris_dir_path, f"{aviris_dir}_img.hdr")
        out = main(_tif_path, aviris_loc_hdr_path, aviris_glt_hdr_path, aviris_img_hdr_path, gt_mask_name, args.gt_dir)
        if out is not None:
            file_not_found.append(out)

    Parallel(n_jobs=8)(delayed(_call_main)(_tif_path) for _tif_path in all_tifs_path)
    print(file_not_found)
