# mkdir -p datasets/nuscenes
# wget https://www.nuscenes.org/public/nuscenes-prediction-challenge-trajectory-sets.zip
# unzip nuscenes-prediction-challenge-trajectory-sets.zip -d datasets/nuscenes
# rm nuscenes-prediction-challenge-trajectory-sets.zip

mkdir -p datasets/sdd
wget http://vatic2.stanford.edu/stanford_campus_dataset.zip
unzip stanford_campus_dataset.zip -d datasets/sdd
rm stanford_campus_dataset.zip