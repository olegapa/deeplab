# DeeplabV3Plus segmentation

### **SORT Tracker**

First of all you need to build an image:
```
docker build -t deeplab-image .
```

## 1. Training or inference over images-masks dataset
raw_data_entrypoint.py entrypoint can be used to train DeepLabV3+ over some dataset (e.g. generated from fashionpedia) that consists of folder with images and folder with masks.
Example of executing an image with raw_data_entrypoint.py entrypoint
```
sudo docker run --gpus all --shm-size=16g -v ./output/images:/images -v ./output/weights:/weights -v ./output:/output -v /var/run/docker.sock:/var/run/docker.sock -v ./output/masks:/masks -it --rm deeplab_raw_data --work_format_training
```
#### Additional flags:

`--demo_mode` - Execute in demo mode. All temporal files like images will be saved in output directory. Colored masks are saved as well.

`--eval_mode` - Execute in metrics' evaluation mode. Metrics like Pixel Average, IoU, Loss function (cross entropy) sre printed in output log file.

#### Contents of `input_data` dictionary parameter:
`tr_h` and `tr_w` - height and weight of deeplab input and output layer. Default value: 224

`epoches` - maximum amount of training epoches. Default value: 50

## 2. Evaluate / training mode

You need to run the container and mount all the necessary directories. Optional deeplab execution parameters can be set through `input_data` parameter 
. Execution example:
```
sudo docker run --gpus all --shm-size=16g -v ./video/:/projects_data -v ./new_videos:/input_videos -v ./weights/trained/deeplab_weights.pt:/weights/deeplab_weights_new.pt -v ./new_dl_inf_output/custom_8cls_res50_4epo_224_224:/output -v /var/run/docker.sock:/var/run/docker.sock -v ./new_dl_inf_input:/input_data -it --rm --entrypoint "/bin/bash deeplab-final --input_data '{"weights": "deeplab_weights_new.pt"}' --host_web "http://127.0.0.1:5555"
```

#### Additional flags:

`--demo_mode` - Execute in demo mode. All temporal files like images will be saved in output directory. Colored masks are saved as well.

`min_height` and `min_width` - Restriction on bounder box minimum size. Only bounder boxes of bigger size are processed

#### Contents of `input_data` dictionary parameter:
`tr_h` and `tr_w` - Height and weight of deeplab input and output layer. Default value: 224

`epoches` - Maximum amount of training epoches. Default value: 50

`frame_frequency` - Filters frames to be processed by frame numbers. E.g. if `frame_frequency = 10` then only frames with numbers 1, 11, 21... are processed

`weights` - Deeplab weights file name

`pixel_hist_step` - Specifies data step for pixel-wise confidence score histogram. If the parameter is specified additional file is generated in inference mode. It can be used for pixelwise_hist.ipynb

`visualize` - If true then videos with processed masks are generated

`eval_mode` - Execute in metrics' evaluation mode. Metrics like Pixel Average, IoU, Loss function (cross entropy) sre printed in output log file.

`approx_eps` - Determines polygon approximation scale. The more value is the less polygons are in output. Default value 0.02

#### Format for input/output file:
```json
{
	files: [
		{
			file_id: <files.id>,		
			file_name: <files.name>,
			file_chains: [
				{
					chain_name: <chains.name>,
					chain_vector: <chains.vector>,
					chain_markups: [
						{
							markup_parent_id: <markups.parent_id **>,
							markup_frame: <пересчет в markups.mark_time ***>,
							markup_time: <markups.mark_time>,
							markup_path: <markups.mark_path>,
							markup_vector: <markups.vector>
						},
						... <список примитивов>
					]
				},
				... <список цепочек>
			]
		},
		... <список файлов>
	]
}
```
где markup_path:

```json
{
  x: <rect_x>,
  y: <rect_y>,
  width: <rect_width>,
  height: <rect_height>,
  polygons: <mask_polygons>,
  "class": <mask_class> //only for training/eval mode
}
```
List of classes presented in shorted_classes.txt file.

Pretrained weights can be downloaded: https://drive.google.com/file/d/18N3ZRyCcno1cLnV4GGHNOGfRMLYolTeI/view?usp=sharing
Fashionpedia pretrained weights can be downloaded: https://drive.google.com/file/d/1sTSvxiswkwZGGQIzd1Lf_aEAhlb1AGtU/view?usp=sharing
