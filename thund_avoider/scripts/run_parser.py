from datetime import datetime, timedelta

from wakepy.modes import keep

from thund_avoider.parser.parser import Parser, collect_data

BASE_URL = "http://s3-eu-west-1.amazonaws.com/fmi-opendata-radar-geotiff/{year}/{month}/{day}/FIN-DBZ-3067-250M/{year}{month}{day}{hour}{minute}_FIN-DBZ-3067-250M.tif"
PROGRESS_FILE = "progress.json"
FIRST_DATE = datetime(2020, 12, 31, 23, 55)
LAST_DATE = datetime(2024, 7, 31, 23, 55)
DELTA = timedelta(minutes=5)
DELTA_HOUR = timedelta(hours=1)
INTENSITY_THRESHOLD = 100
MIN_PIXELS = 9000
ANGLES = [
    (60, 120),
    (150, 210),
    (240, 300)
]


if __name__ == "__main__":
    parser = Parser(
        base_url=BASE_URL,
        progress_file=PROGRESS_FILE,
        first_date=FIRST_DATE,
        last_date=LAST_DATE,
        delta=DELTA,
        delta_hour=DELTA_HOUR,
        intensity_threshold=INTENSITY_THRESHOLD,
        min_pixels=MIN_PIXELS,
        angles=ANGLES
    )
    with keep.presenting():
        parser.get_data()
        collect_data("../parser/data")
