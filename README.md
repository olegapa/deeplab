# DeeplabV3Plus segmentation

### **SORT Tracker**

First of all you need to build an image:
```
docker build -t deeplab-image .
```

#### 1. Evaluate / training mode

You need to run the container and mount all of the necessary directories. As an input_data parameter it is possible to pass parameters in json format, 
for now, only parameter "frame_frequency" is supported. Example how to launch inference:
```
sudo docker run --gpus all --shm-size=16g -v ./video/:/projects_data -v ./video_1:/input -v ./weights/deeplab_weights.pt:/models/deeplab_weights.pt -v ./output:/output -it --rm deeplab-image --input_data '26_ОП_юж_Выход_0_edited.json'
```
Example how to launch training:
```
sudo docker run --gpus all --shm-size=16g -v ./video/:/projects_data -v ./video_1:/input -v ./weights/deeplab_weights.pt:/models/deeplab_weights.pt -v ./output:/output -it --rm deeplab-image --input_data 'deeplab.json' --work_format_training
```

Apart from default keys: input_data and --work_format_training (flag that marks training mode)
there also --demo_mode flag for inference that allows to save output bounder boxes and masks images in output
directory (in order to visualize the results). 

Additionally added python file raw_data_entrypoint that can be used as entrypoint script in Dockerfile. In this case Deeplab takes image directory (and mask directory if
work_format_training) as input. Can be useful for testing and training on image datasets.

Format for inference output file:
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
  mask: <base_64_encoded_mask_image>,
  polygons: <classes_mask_polygons>
}
```
List of classes presented in classes_new.txt file.

Pretrained weights can be downloaded: https://drive.google.com/file/d/18N3ZRyCcno1cLnV4GGHNOGfRMLYolTeI/view?usp=sharing
Fashionpedia pretrained weights can be downloaded: https://drive.google.com/file/d/1sTSvxiswkwZGGQIzd1Lf_aEAhlb1AGtU/view?usp=sharing
