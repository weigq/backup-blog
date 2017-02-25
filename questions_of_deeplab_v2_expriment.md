---
title: wiki of deeplab_v2 test
tags:
  - deeplab
  - deepl learning
date: 2017/01/13
---
this blog will be updated continually   
# 1. Install the VOC2012
&emsp;this project is trained on the VOC2012 dataset, so need to [install the dataset](http://host.robots.ox.ac.uk/pascal/VOC/).But you need to log up first and it may be slow to download the dataset in China, luckily, there is a [mirror site](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) which provide the mirrors of PASCAL VOC2007/2012.
&emsp;And here are some steps of installing the dataset:
1. download the 3 dataset compressed files including train/validation set, test set, development kit.
2. extract all the files and the tree is as following:
```
tensorflow-deeplab-resnet$  cd ./VOCdevkit/
tensorflow-deeplab-resnet/VOCdevkit$ ls

create_segmentations_from_detections.m
example_action.m       
example_classifier.m
example_layout.m     
local    
viewanno.m
VOC2012
devkit_doc.pdf                          
example_action_nobb.m  
example_detector.m   
example_segmenter.m  
results  
viewdet.m   
VOCcode
```
# 2. New error: lack of useful files
just run the instruction script:    

    python train.py --random-scale

but encounter the following error:
```
/tensorflow-deeplab-resnet$ python train.py --random-scale

I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties:
name: Tesla K40m
major: 3 minor: 5 memoryClockRate (GHz) 0.745
pciBusID 0000:06:00.0
Total memory: 11.17GiB
Free memory: 11.10GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K40m, pci bus id: 0000:06:00.0)


Traceback (most recent call last):
  File "train.py", line 189, in <module>
    main()
  File "train.py", line 168, in main
    load(loader, sess, args.restore_from)
  File "train.py", line 85, in load
    saver.restore(sess, ckpt_path)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/training/saver.py", line 1342, in restore
    "File path is: %r" % (save_path, file_path))


ValueError: Restore called with invalid save path: './deeplab_resnet.ckpt'. File path is: './deeplab_resnet.ckpt'
```

&emsp;this error is due to the lack of `./deeplab_resnet.ckpt`, which is the pre-trained Caffe-tensorflow model. The author gives the [link](https://drive.google.com/drive/folders/0B_rootXHuswsZ0E4Mjh1ZU5xZVU?usp=drive_web) to download deeplab_resnet.ckpt and deeplab_resnet_init.ckpt.

# 3. New error: wrong folders
just run the instruction script again:     

    python train.py --random-scale

but encounter the following error:
```
tensorflow-deeplab-resnet$ python train.py --random-scale


I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcudnn.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:111] successfully opened CUDA library libcurand.so locally
I tensorflow/core/common_runtime/gpu/gpu_device.cc:951] Found device 0 with properties:
name: Tesla K40m
major: 3 minor: 5 memoryClockRate (GHz) 0.745
pciBusID 0000:06:00.0
Total memory: 11.17GiB
Free memory: 11.10GiB
I tensorflow/core/common_runtime/gpu/gpu_device.cc:972] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1041] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K40m, pci bus id: 0000:06:00.0)
Restored model parameters from ./deeplab_resnet.ckpt

# XXX THE FOLLOWING IS THE ERROR! XXX

W tensorflow/core/framework/op_kernel.cc:968] Not found: ./VOCdevkit/JPEGImages/2009_004475.jpg
W tensorflow/core/framework/op_kernel.cc:968] Not found: ./VOCdevkit/SegmentationClassAug/2009_004475.png
W tensorflow/core/framework/op_kernel.cc:968] Out of range: FIFOQueue '_ 1_create_inputs/batch/fifo_queue' is closed and has insufficient elements (requested 4, current size 0)
.
.
.
.
.
.	 
.
Traceback (most recent call last):
  File "train.py", line 192, in <module>
    main()
  File "train.py", line 181, in main
    loss_value, images, labels, preds, summary, _ = sess.run([reduced_loss, image_batch, label_batch, pred, total_summary, optim])
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 717, in run
    run_metadata_ptr)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 915, in _ run
    feed_dict_string, options, run_metadata)
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.py", line 965, in _ do_run
    target_list, options, run_metadata)
  File "/usr/local/lib/python2.7/dist-message)
tensorflow.python.framework.errors.OutOfRangeError: FIFOQueue '_ 1_create_inputs/batch/fifo_queue' is closed and has insufficient elements (requested 4, current size 0)

	 [[Node: create_inputs/batch = QueueDequeueMany[_ class=["loc:@create_inputs/batch/fifo_queue"], component_types=[DT_FLOAT, DT_UINT8], timeout_ms=-1,  _ device="/job:localhost/replica:0/task:0/cpu:0"](create_inputs/batch/fifo_queue, create_inputs/batch/n)]]
```

I`ll update the error and its solution continually!  

I`ve solved the problem, note the hint message of the error:

```
Not found: ./VOCdevkit/JPEGImages/2009_004475.jpg
Not found: ./VOCdevkit/SegmentationClassAug/2009_004475.png
```
&emsp;it seems like that the archives are wrong, and the`JPEGImages` folder is in the VOC2012. So move all the folders and files of the VOC2012 and VOCcode to the `./VOCdevkit`.
&emsp;Another question is the SegmentationClassAug folder, which is not belong to the VOC2012, it is an another dataset designed by [Bharathh](http://home.bharathh.info/), and the author of the tensorflow-deeplab-resnet gives the [link](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0) to download the dataset, but it a link to dropbox, which is not accessible in China. I have downloaded it, and you can also download it from [百度网盘](https://pan.baidu.com/s/1pKQUXNP) or just from the dropbox if you can.
&emsp;Then, just extract the files to `./VOCdevkit`. And the error will be solved.
# 4. the version of cudnn is mismatch
&emsp;the program needs cudnn-5.1, but the cudnn on my machine is just 4.0, so need to update it.
&emsp;download the cudnn library from NVIDIA [website](https://developer.nvidia.com/cudnn).extract the folder and it will create a new folder _cuda_, which includes TWO folders: include and lib64. Next, need to set the path of cudnn:

    $ vim ~/.bashrc
insert the following to the file:
```
export CUDNN_PATH=/path_to_your_cuda/lib64/libcudnn.so.5.1.10
export LIBRARY_PATH=/path_to_your_cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/dpath_to_your_cuda/lib64:$LD_LIBRARY_PATH
export PATH=/path_to_your_cuda:$PATH
```
then enable the configuration, and it will works!

    $ source ~/.bashrc

# FINALLY
&emsp;Finally, I`ve solved all the problems, and get the program running! Excited! Thank the [author](https://github.com/DrSleep) of [tensorflow-deeplab-resnet](https://github.com/DrSleep/tensorflow-deeplab-resnet) very much for help solve the problem on github.    

&emsp;And I`ll do some augments or changes about this project next.

![alt text](https://raw.githubusercontent.com/weigq/image-raw/master/blog/running.png)
