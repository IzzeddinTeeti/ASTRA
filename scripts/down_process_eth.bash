

## download ETH videos and annotations
# download ETH videos
gdown 1qcWaVpyeosiirNP7pVyxHmeiJfzwnkKr

# unzip
mkdir -p datasets/eth_ucy
unzip eth_dataset.zip -d datasets/eth_ucy

# Fetching Homographic Matrix for each data
# Link Ref: https://github.com/HarshayuGirase/Human-Path-Prediction/issues/27
# Note: Students and Univ has same homography matrix
mkdir -p datasets/eth_ucy/homography
gdown 15hDOhjSJbKtV9ys1D_9tGFyKpt6Bv2eJ -O datasets/eth_ucy/homography/H_eth.txt
gdown 1LwQCCpjBGaHpIQBNU1b5iA7d8CfxzhoE -O datasets/eth_ucy/homography/H_hotel.txt
gdown 1Mnb9sJB7TmDHWYDgYxN-lvIQSw9p4gJH -O datasets/eth_ucy/homography/H_students003.txt
gdown 1FYGpoFAoIzmkpSjv5jSJfsfjWfnBlmqK -O datasets/eth_ucy/homography/H_zara01.txt
gdown 1x9VWY2H3IV-6MT60TKOFb0A5Tp8yaSMP -O datasets/eth_ucy/homography/H_zara02.txt

# convert videos to images
python utils/video2images.py 

# process the annotations
for dataset in 'eth' 'hotel' 'univ' 'zara01' 'zara02'; do
    python data/process_eth.py --subset $dataset
done

# remove the downloaded zip file
rm eth_dataset.zip
