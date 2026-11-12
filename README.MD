# FedHydra: a federated classifier based on Hydra

## Abstract
FedHydra builds upon Fed2RC (https://github.com/CasellaJr/Fed2RC). Our goal was to use the Hydra algorithm for time series classification, in Federated Learning. Hydra is a dictionary-based method using competing convolutional kernels, incorporating aspects of both Rocket
and conventional dictionary methods.

## Usage
* Clone this repo.
* Download and unzip the UCRArchive Time Series Classification dataset: UCRArchive, and place it in the same directory of the repository.
* Install the requirements.
* Run experiments:
python fedhydra.py 


Note: To change the configurations and experimental setups, please look at or edit the code/fedhydra.py file directly.

## Results
The results reports the accuracy and F1-score for all the 128 datasets of the UCR Archive. Inside the results/ directory, you can find experiments evaluated in both non-i.i.d. and i.i.d. settings.
Other experiments regarding unsual configurations of FedHydra are available in the results/sensitivity_analysis directory.
