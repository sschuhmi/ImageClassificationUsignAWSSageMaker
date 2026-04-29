# Image Classification using AWS SageMaker

In this project, we use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. Therefor, we relied on the provided dog breed classication data set.

## Project Set Up and Installation
1. Enter AWS through the gateway in the course and open SageMaker Studio. 
2. Download the starter files.
3. Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you can easily switch it to a dataset of your choice.

### Access
The data was uploaded to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
For this experiment, I chose a pre-trained image classification model and fine-tuned it for the dog-breed classification task. A transfer learning approach was appropriate because the dataset is specialized, while the pre-trained model already contains useful visual feature representations learned from a large benchmark dataset. This reduces training time, improves convergence, and usually gives better performance than training a model from scratch.

For hyperparameter tuning, I focused on parameters that strongly affect optimization and generalization. For example, I varied the learning rate (LR) over a small logarithmic range such as 0.0001 to 0.01, and I tested different batch sizes such as 32 and 64. I could also include the number of training epochs or weight decay as additional parameters. These ranges were chosen to balance training stability, model performance, and computational cost while exploring both conservative and more aggressive training settings.

The best hyperparameters were the following: {'batch_size': 32, 'lr': '0.0007073488418203175'}

Screnshots on the completed training jobs as well as a detail screenshot of a specific training job are provided as 'TrainingJobs_Screenshot.jpg' and 'TrainingJob-Detail_Screenshot.jpg'.

## Debugging and Profiling
To perform model debugging and profiling in SageMaker, I configured the training job with SageMaker Debugger and Profiler in the training notebook. 
I enabled built-in rules to monitor training performance and resource utilization, including checks for dataloader issues, CPU and I/O bottlenecks, GPU utilization, batch size efficiency, and step outliers. 
After training, I generated the SageMaker Profiler Report and analyzed the collected metrics.

From the report, I observed that no built-in profiling rules were triggered, suggesting that the training job ran without major performance bottlenecks. 
The report also showed generally stable step durations, with only a small number of outlier steps. 
Overall, SageMaker Debugger and Profiler were useful for validating that the model training process was functioning correctly and making efficient use of the available resources.

### Results
Major results/insights by profiling/debugging our model:
- No anomalous behavior was observed in the debugging/profiler output. The training job completed normally, and the profiler metrics did not show obvious issues such as stalled steps, exploding resource usage, or repeated failures during forward/backward passes.
- If there had been an error, a likely example would be an out-of-memory issue. In the profiler output, this would typically appear as failing training steps, crashes during batch processing, or resource utilization spiking near the hardware limit. This issue could be fixed by reducing the batch size, using a smaller model, resizing input images, or switching to a larger instance type.
- Another possible anomaly could be low GPU utilization caused by a CPU or dataloader bottleneck. In the profiler report, this would appear as the LowGPUUtilization, CPUBottleneck, or Dataloader rules being triggered. These issues could by increasing the number of dataloader workers, improving data preprocessing/loading, increasing batch size, or using a more suitable instance type.

The profiler html/pdf is attached to the submission as 'profiler-report.html'.


## Model Deployment

The trained model was deployed to a SageMaker inference endpoint and queried using a sample dog image in JPEG format. The image was read as bytes and sent to the endpoint with the content type set to image/jpeg. The endpoint returned a probability vector across all dog-breed classes. I then selected the class with the highest probability using argmax and mapped the predicted class index to the corresponding class name.

Three different images of dogs from different classes were used to test the SageMaker inference endpoint - they are provided as 'DuckToller.jpg', 'Poodle.jpg', and 'Schaeferhund.jpg' within this project.
A screenshot of the deployed active endpoint in Sagemaker is also provided as 'Endpoint_Screenshot.jpg'.

## Standout Suggestions

- We could compare multiple pretrained backbones, like ResNet, EfficientNet, or MobileNet, and justify the final choice based on accuracy vs. training cost.
- We could run a broader hyperparameter search which - besides learning rate and batch size - additionally covers different optimizer, numbers of frozen/unfrozen layers, or numbers of epochs.
- We could monitor the endpoint after deployment with logging and latency/error metrics
- We could save and return the top-3 predictions with class names and confidence scores instead of only the top-1 class