import argparse

# Inputs
parser = argparse.ArgumentParser()

parser.add_argument("-vis", "--visualize", action="store_true", help="visualize the land cover in color scale")
parser.add_argument("-rgb", "--color_img", action="store_true", help="visualize the ground in RGB domain")
parser.add_argument(
    "-sbs", "--side_by_side", action="store_true", help="visualize land cover and RGB image side by side"
)
parser.add_argument(
    "-smf",
    "--segmentation_mf",
    choices=["row_wise", "column_wise"],
    help="compute matched filter based on segmentation, picks pixels row wise or column wise",
)
parser.add_argument(
    "-snfmf",
    "--sensor_noisefree_mf",
    choices=["row_wise", "column_wise"],
    help="compute matched filter based on segmentation, computes segmentation on each set of sensors",
)
parser.add_argument(
    "-pxl", "--pxl_batch_size", default=100000, type=int, help="Pixel batch size for segmented matched filter"
)
parser.add_argument(
    "-sensors",
    "--num_of_sensor",
    default=8,
    type=int,
    help="number of sensors to take in one set to supress sensor noise",
)
parser.add_argument(
    "-cols", "--num_columns", default=5, type=int, help="Number of columns to pick for match filter computation"
)
args = parser.parse_args()
