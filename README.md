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

### Additional Method - KDE(Kernel Density Estimation) application
Instead of Determining hyperspheres for every floor by calculating radii (0.9 quantiles of distances) and centers(averages of data points) of spheres,
We can use KDE to determine whether the corresponding floor is normal state or damaged.

#### What is KDE?
Method to estimate the probability density function from point distribution

![KDE](https://github.com/happyleeyi/MCDSVDD-for-SHM/assets/173021832/2b25b0cc-cc22-42bb-948d-19ec51c0755b)

#### How to use in MCDSVDD?
![KDE - 복사본](https://github.com/happyleeyi/MCDSVDD-for-SHM/assets/173021832/8862adbe-bb1d-4cea-a29f-80e54d88cbf3)

1. Train KDEs with outputs of training normal data obtained with trained MCDSVDD, and then use them in the test section.
2. Put the outputs of test data obtained with MCDSVDD into the corresponding KDE for every floor.
3. If the output probability came from KDE is bigger then threshold probability, the corresponding floor is determined as normal. However, it is not, the corresponding floor is damaged.
4. If all floors are normal, the building is normal, but if there are abnormal floors, the floor where the output probability is lowest is determined to be damaged.



## Result

### 1. Accuracy depending on Representation Dimension of Hyperspace
![image](https://github.com/happyleeyi/MCDSVDD-for-SHM/assets/173021832/c9cf3f20-5cde-44b3-9cd4-6d972288868c)
![image](https://github.com/happyleeyi/MCDSVDD-for-SHM/assets/173021832/356fffb8-7c68-4be0-82eb-6b87591e8862)

Maximum accuracy was __95.58%__ on 4 representation dimensinon in 2nd test

Total average accuracy is __80.73%__

### 2. Accuracy depending on Radius Quantile (variable that used when we calculate the radius of sphere)
![image](https://github.com/happyleeyi/MCDSVDD-for-SHM/assets/173021832/340df077-5ce5-4d95-a735-310183cba243)

It seems that accuracy wasn't changed while the radius quantile value changes 0.7 to 0.9 

So It seems good to choose a radius quantile value __0.8 ~ 0.9__

### 3. Effect of application of KDE
![image](https://github.com/happyleeyi/MCDSVDD-for-SHM/assets/173021832/7729ccd8-5d5f-49ee-b50f-d5eaa0053eca)


![image](https://github.com/happyleeyi/MCDSVDD-for-SHM/assets/173021832/7a4d23a6-fa34-4b66-9aab-b09b95340772)

MCDSVDD with KDE shows better performance than original MCDSVDD method.

### 4. Accuracy of MCDSVDD with KDE depending on bandwidth (variable that determines how sharp the made distribution is)

![image](https://github.com/happyleeyi/MCDSVDD-for-SHM/assets/173021832/198bbc21-c199-482d-8f03-6f02c2fd141e)

It seems good to choose a bandwidth value __1 ~ 1.25__

### 5. Compare with other method

![image](https://github.com/happyleeyi/MCDSVDD-for-SHM/assets/173021832/6bc28aae-a228-4198-ab92-074a6bcd0184)

If we use several One Class Deep SVDD per floor instead of one Multi Class Deep SVDD, the accuracy was __0.64__

If we use the traditional SVDD instead of Multi Class Deep SVDD, the top accuracy was __0.584__

If we use the anomaly detection using autoencoder method, the accuracy was __0.582__ when the dimension of latent vector space was 32

