# Multi Class Deep SVDD(machine learning) for SHM

## Introduction

![Multi Deep svdd model pic - 복사본](https://github.com/happyleeyi/DeepSVDD/assets/173021832/47b45b16-2101-4a44-9d1e-1401a5ae94df)

The research to suggest a method to find whether and where the defect happens in the building on real-time.

We use the Multi Class Deep SVDD method to find the location of the defect in this section.

## Dataset

3-story shake table test data with an undamaged case and several damaged cases

![image](https://github.com/happyleeyi/DeepSVDD/assets/173021832/f5b6d5f1-b504-4c4a-bdbe-be64fa2c6588)

(open-source by Engineering Institute at Los Alamos National Laboratory)

## Method

### 1. data processing
1. prepare dataset (we use 3F shake table test dataset, 8 accelerometers per floor, a total of 24 accelerometers)
2. process dataset (we downsample and cut the training dataset and concatenate 8 acc data per each floor -> training dataset : (8, 512) for one data)

### 2. Training
![Multi Deep svdd model pic](https://github.com/happyleeyi/DeepSVDD/assets/173021832/1c007e9a-019e-42bf-af1d-7b0caeb6db96)

1. training autoencoder (autoencoder extract the features from input data)
2. copy the encoder part of the autoencoder and use it as Multi Class Deep SVDD model
3. before training the Deep SVDD, obtain the center of the data points for each floor by calculating the average of the outputs from the Deep SVDD model (the Deep SVDD model is before training, so it is the same as the encoder part of AE yet)
4. train Multi Class Deep SVDD (Deep SVDD is trained to map the data points closer to each sphere of the corresponding floor)
5. obtain the radius of each sphere of the corresponding floor by calculating the 0.9 quantiles of the distances that are from the datas and to the corresponding center

### 3. Test
1. cut the test data for 3 floors (test dataset : (24, 512) -> (8,512) * 3 for one data)
2. put the data into the Multi Class Deep SVDD and map to hyperspace for every floor. (total 3 datas per one data in this dataset)
![image](https://github.com/happyleeyi/DeepSVDD/assets/173021832/167c7788-6fe7-4312-ab6f-ea98e0ff0dc6)

3. if every data is mapped into the sphere of the corresponding floor, the building is on normal state. (we can notice whether the datas are inside or outside of the corresponding spheres by using the centers and the radii of each spheres which are calculated on training section.)
4. however, if several datas are mapped outside of the sphere of the corresponding floor, the floor that the data mapped the furthest from the corresponding sphere is determined to be damaged.

## Result
![image](https://github.com/happyleeyi/DeepSVDD/assets/173021832/d16b1941-529c-4a9b-b14f-91e0d6a49e8f)

__Accuracy : 0.92__

If we use several One Class Deep SVDD per floor instead of one Multi Class Deep SVDD, the accuracy was 0.64.

If we use the traditional SVDD instead of Multi Class Deep SVDD, the top accuracy was 0.584

If we use the anomaly detection using autoencoder method, the accuracy was 0.582 when the dimension of latent vector space was 32

