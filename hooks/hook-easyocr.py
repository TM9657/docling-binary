# hook-easyocr.py
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

# Tell PyInstaller to collect the metadata for easyocr, which includes
# the entry_points.txt file that registers the 'easyocr' class.
hiddenimports = collect_submodules('easyocr')
datas = copy_metadata('easyocr') + collect_data_files('easyocr')