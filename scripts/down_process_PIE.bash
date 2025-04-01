mkdir -p datasets
cd datasets
git clone https://github.com/aras62/PIE.git
cd PIE
unzip -qq annotations_attributes.zip -d '.'
unzip -qq annotations_vehicle.zip -d '.'
unzip -qq annotations.zip -d '.'
sh download_clips.sh
sh split_clips_to_frames.sh
python utils/generatePIEAnnotation.py 