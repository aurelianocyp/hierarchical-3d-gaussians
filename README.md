

## Setup

Make sure to clone the repo using `--recursive`:
```
git clone https://github.com/graphdeco-inria/hierarchical-3d-gaussians.git --recursive
cd hierarchical-3d-gaussians
```
### Prerequisite

We tested on Ubuntu 22.04 and Windows 11 using the following: 

* CMake 3.22.1
* gcc/g++ 11.4.0 or Visual Studio 2019
* CUDA (11.8, 12.1 or 12.5)
* [COLMAP 3.9.1](https://github.com/colmap/colmap/releases/tag/3.9.1) (for preprocessing only). Linux: [build from source](https://colmap.github.io/install.html). Windows: add the path to the COLMAP.bat directory to the PATH environment variable.

### Python environment for optimization
```
conda create -n hierarchical_3d_gaussians python=3.12 -y
conda activate hierarchical_3d_gaussians
# 根据使用的cuda版本在下面命令中替换末尾三个数字
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 
pip install -r requirements.txt
```
### Weights for monocular depth estimation 
To enable depth loss, download the model weights of one of these methods:
* [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (suggested): download from [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) and place it under `submodules/Depth-Anything-V2/checkpoints/`.
* [DPT](https://github.com/isl-org/DPT) (used in the paper): download from [dpt_large-midas-2f21e586.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt) and place it under `submodules/DPT/weights/`.

### Compiling hierarchy generator and merger
```
cd submodules/gaussianhierarchy
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc
cmake --build build -j --config Release
cd ../..
```
这里的cmake好像是分两步，第一步配置，第二步构建。和之前的cmake ..不同。 CMAKE_CUDA_COMPILER视机器cuda版本而定。cuda版本可以通过nvcc -V查看。cuda位置可以通过whereis nvcc查看

原始的cmakelist可能会报错，需要字在project行前添加
```
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CUDA_ARCHITECTURES "native")
```
### Compiling the real-time viewer 
For Ubuntu 22.04, install dependencies:
```
sudo apt install cmake libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev  libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
```
自行源码安装opencv4.5，需要opencv contrib

sudo apt install git

在SIBR_viewers目录下 git clone https://github.com/graphdeco-inria/gaussian-hierarchy然后改目录名为GaussianHierarchy

克隆完下面的库后需要在SIBR_viewers/extlibs/CudaDiffRasterizer/CudaDiffRasterizer中的CMakeLists.txt的14行project CUDA前添加set(CMAKE_CUDA_ARCHITECTURES "native")

Clone the hierarchy viewer and build:
```
cd SIBR_viewers
git clone https://github.com/graphdeco-inria/hierarchy-viewer.git src/projects/hierarchyviewer
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_IBR_HIERARCHYVIEWER=ON -DBUILD_IBR_ULR=OFF -DBUILD_IBR_DATASET_TOOLS=OFF -DBUILD_IBR_GAUSSIANVIEWER=OFF  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc
cmake --build build -j --target install --config Release
```

如果遇到cannot find -lembree，可查看：https://elenacliu.github.io/post/gaussian_splatting/ ，或者将SIBR_viewers/src/core/raycaster/CMakeLists.txt的 embree 改成 embree3即可

## Running the method

Our method has two main stages: 
* Reconstruction, that takes a (usually large) set of images as input and outputs a "merged hierarchy"
* [Runtime](#3-real-time-viewer), that displays the full hierarchy in real-time. 

Reconstruction has two main steps: 1) **[Preprocessing](#1-preprocessing)** the input images and 2) **[Optimization](#2-optimization)**. We present these in detail next. For each step we have automatic scripts that perform all the required steps, and we also provide details about the individual components.

#### Dataset 
To get started, prepare a dataset or download and extract the [toy example](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/example_dataset.zip). 
The dataset should have sorted images in a folder per camera in `${DATASET_DIR}/inputs/images/` and optional masks (with `.png` extension) in `${DATASET_DIR}/inputs/masks/`. Masks will be multiplied to the input images and renderings before computing loss. 

toy example结构：
```
example_dataset
    └─inputs
        ├─images
        │  ├─cam1（含很多jpg文件夹）
        │  ├─n个cam文件夹
        └─masks（optional）
            ├─cam1
            ├─n个cam文件夹
```

You can also work from our full scenes. As we provide them calibrated and subdivided, you may skip to [Generate monocular depth maps](#13-generate-monocular-depth-maps). The datasets:
* [SmallCity](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/full_scenes/small_city.zip)


In the following, replace `${DATASET_DIR}` with the path to your dataset or set DATASET_DIR:
```
# Bash:
DATASET_DIR=<Path to your dataset>
# 最好是在hierarchical目录下，路径就写相对路径就行，比如DATASET DIR=data/data2/zch_ISPRS/

# PowerShell:
${DATASET_DIR} = "<Path to your dataset>"
```

>*To skip the reconstruction and only display scenes, download pretrained hierarchies and scaffolds, place them under `${DATASET_DIR}/output/` and follow the [viewer instructions](#3-real-time-viewer). The pretrained hierarchies:* 
>* [*SmallCity*](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/results/small_city.zip) 

## 1. Preprocessing
As in [3dgs](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) we need calibrated cameras and a point cloud to train our hierarchies on.

### 1.1 Calibrating the cameras
数据集放置：只需要像上面列的toy example（example_dataset）那样放置即可，也就是把jpg图片放在cam1中就可以开始直接运行程序了，但是最好这样放：hierarchical-3d-guassians/data/data2/toy_example/inputs/images/cam1

23735 is not a valid TiffByteOrder：https://github.com/graphdeco-inria/hierarchical-3d-gaussians/issues/2 。这应该是通过强行改后缀名png为jpg造成的问题。数据格式并没有发生改变

报错重试的时候可以把camera_calibration文件夹删掉

图片格式使用jpg格式，不要通过改后缀名的方式，而是通过专业的修改数据来修改，我已经保存了修改的代码

它给的原始数据集里面有多个cam，实际上把nerfstudio里的workspace里的下载下来都行。然后就当做cam1，无需多cam

The first step is to generate a "global colmap". The following command uses COLMAP's hierarchical mapper, rectify images and masks, and align and scale the sparse reconstruction to facilitate subdivision.
```
python preprocess/generate_colmap.py --project_dir ${DATASET_DIR} 
```

<details>
<summary><span style="font-weight: bold;">Using calibrated images</span></summary>

If your dataset already has COLMAP (with 2D and 3D SfM points) and rectified images, they should be placed under `${DATASET_DIR}/camera_calibration/rectified`. As they still need alignment, run: 
```
python preprocess/auto_reorient.py --input_path ${DATASET_DIR}/camera_calibration/rectified/sparse --output_path ${DATASET_DIR}/camera_calibration/aligned/sparse/0
```

</details>
<br>

*This step takes ~ 47 minutes on our example dataset using a RTX A6000, more details on each steps of the script [here](#generating-colmap).*

### 1.2 Generate chunks
Once the "global colmap" generated, it should be split into chunks. We also run a per-chunk bundle adjustment as COLMAP's hierarchical mapper is faster but less accurate (if your global colmap is accurate, you can skip this time consuming step with `--skip_bundle_adjustment`). 

```
python preprocess/generate_chunks.py --project_dir ${DATASET_DIR}
```
*This step takes ~ 95 minutes on our example dataset using a RTX A6000, more details on each steps of the script [here](#generating-chunks).*
> note that by using `--use_slurm` you can refine the chunks in parallel, remember to set your [slurm parameters](#slurm-parameters) in `preprocess/prepare_chunks.slurm` (gpu, account, etc ...).

### 1.3 Generate monocular depth maps
这一步会试图在hierarchical目录下的 ../../example_data创建文件，所以建议把hierarchical放在更深两层文件夹去防止权限不允许。
In order to use depth regularization when training each chunks, depth maps must be generated for each rectified image. Then, depth scaling parameters needs to be computed as well, these two steps can be done using:
```
python preprocess/generate_depth.py --project_dir ${DATASET_DIR}
```
### Project structure
Now you should have the following file structure, it is required for the training part: 
```
# 在上层目录（与hierarchical同层次的）会出现data文件夹（虽然该文件夹内全是文件夹没有任何文件）
# 这个project目录指的是hierarchical/data/data2/数据集文件夹
# 没有ply文件是正常的，不要觉得第二步full train报错时这里没有ply文件就有问题
project
└── camera_calibration
    ├── aligned
    │   └── sparse/0
    │       ├── images.bin
    │       ├── cameras.bin
    │       └── points3D.bin
    ├── chunks
    │   ├── 0_0
    │   └── 0_1
    │   .
    │   .
    │   .
    │   └── m_n
    │       ├── center.txt
    │       ├── extent.txt
    │       └── sparse/0
    │           ├── cameras.bin
    │           ├── images.bin
    │           ├── points3d.bin
    │           └── depth_params.json
    └── rectified
        ├── images
        ├── depths
        └── masks
```
## 2. Optimization

如果后面还需要跑评价指标不要使用这一步里面的full train，使用后面评价指标那里的full train

*The scene training process is divided into five steps; 1) we first train a global, coarse 3D Gaussian splatting scene ("the scaffold"), then 2) train each chunk independently in parallel, 3) build the hierarchy, 4) optimize the hierarchy in each chunk and finally 5) consolidate the chunks to create the final hierarchy*.

Make sure that you correctly [set up your environment](#setup) and [built the hierarchy merger/creator](#compiling-hierarchy-generator-and-merger)

The `full_train.py` script performs all these steps to train a hierarchy from a preprocessed scene. While training, the progress can be visualized with the original 3DGS remote viewer ([build instructions](#compiling-the-real-time-viewer)).

可能报Could not recognize scene type!。应该是报错缺少文件。但是应该不是缺少文件，只是程序的路径写的有点问题：
* a6000/h3dgs/h3dgs2/hierarchical-3d-gaussians/scene/__init__.py里的:
* ```python
  #if os.path.exists(os.path.join(args.source_path, "camera_calibration/aligned/sparse")):改为
  if os.path.exists(os.path.join(args.source_path, "sparse")):
  ```
* a6000/h3dgs/h3dgs2/hierarchical-3d-gaussians/scene/dataset_readers.py里：
* ```python
          #cameras_extrinsic_file = os.path.join(path, "camera_calibration/aligned/sparse/0", "images.bin")改为
        #cameras_intrinsic_file = os.path.join(path, "camera_calibration/aligned/sparse/0", "cameras.bin")改为
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
```
python scripts/full_train.py --project_dir ${DATASET_DIR} 
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments</span></summary>

### --colmap_dir
Input aligned colmap.

### --images_dir
Path to rectified images.

### --depths_dir
Path to rectified depths.

### --masks_dir
Path to rectified masks.

### --chunks_dir
Path to input chunks folder.

### --env_name
Name the conda env you created earlier.

### --output_dir
Path to output dir.

### --use_slurm
Flag to enable parallel training using slurm (`False` by default).

</details>
<br>

> note that by using `--use_slurm`, chunks will be trained in parallel, to exploit e.g. multi-GPU setups. To control the process, remember to set your [slurm parameters](#slurm-parameters) in `coarse_train.slurm`, `consolidate.slurm` and `train_chunk.slurm` (gpu, account, etc ...)

*This step takes ~ 171 minutes on our example dataset using a RTX A6000, more details on each steps of the script [here](#training-steps).*

# 3. Real-time viewer

The real-time viewer is based on SIBR, similar to original 3DGS. For setup, please see [here](#compiling-the-real-time-viewer)

## Running the viewer on a merged hierarchy
The hierarchical real-time viewer is used to vizualize our trained hierarchies. It has a `top view` that displays the structure from motion point could as well as the input calibrated cameras in green. The hierarchy chunks are also displayed in a wireframe mode.

![alt text](assets/hierarchy_viewer_0.gif "hierarchy viewer")

After [installing the viewers](#compiling-the-real-time-viewer), you may run the compiled SIBR_gaussianHierarchyViewer_app in `<SIBR install dir>/bin/`. Controls are described [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#navigation-in-sibr-viewers).

If not a lot of VRAM is available, add `--budget <Budget for the parameters in MB>` (by default set to 16000, assuming at least 16 GB of VRAM). Note that this only defines the budget for the SCENE representation. Rendering will require some additional VRAM (up to 1.5 GB) for framebuffer structs. Note that the real-time renderer assumes that CUDA/OpenGL Interop is available on your system (see the original 3DGS documentation for more details). 

The interface includes a field for ```tau (size limit)``` which defines the desired granularity setting. Note that ```tau = 0``` will try to render the complete dataset (all leaf nodes). If the granularity setting exceeds the available VRAM budget, instead of running out of memory, the viewer will auto-regulate and raise the granularity until the scene can fit inside the defined VRAM budget.
```
SIBR_viewers/install/bin/SIBR_gaussianHierarchyViewer_app --path ${DATASET_DIR}/camera_calibration/aligned --scaffold ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000 --model-path ${DATASET_DIR}/output/merged.hier --images-path ${DATASET_DIR}/camera_calibration/rectified/images
```
<details>
<summary><span style="font-weight: bold;">Command Line Arguments for Real-Time Viewer</span></summary>

  #### --model-path / -m
  Path to a trained hierarchy.
  #### --iteration
  Specifies which of state to load if multiple are available. Defaults to latest available iteration.
  #### --path / -s
  Argument to override model's path to source dataset.
  #### --rendering-size 
  Takes two space separated numbers to define the resolution at which real-time rendering occurs, ```1200``` width by default. Note that to enforce an aspect that differs from the input images, you need ```--force-aspect-ratio``` too.
  #### --images-path
  Path to rectified input images to be viewed in the top view.
  #### --device
  Index of CUDA device to use for rasterization if multiple are available, ```0``` by default.
  #### --budget
  Amount of VRAM memory that may be used for the hierarchical 3DGS scene representation.
</details>
<br>

---


# Details on the different steps 上面三个步骤的具体细节，下面的命令是上面命令的分解

## Generating colmap
> note that in our experiments we used [colmap 3.9.1 with cuda support](https://github.com/colmap/colmap/releases/tag/3.9.1)<br>
> the parameters of each colmap commands as well as our scripts are the ones we used in the example dataset.<br>
> More details on these parameters can be found [here](https://colmap.github.io/cli)

- Create a `project` folder and create the required folders to have the following file structure:
    ```
    project
    ├── camera_calibration
    │   ├── aligned
    │   ├── rectified
    │   └── unrectified
    └── output
    ```

- Generate a `database.db` in the `unrectified` subfolder by extracting features from images:<br>
    *Input image folder should be organised by subfolders per camera.*

    ```
    cd project/unrectified
    colmap feature_extractor --database_path database.db --image_path <path to images> --ImageReader.single_camera_per_folder 1 --ImageReader.default_focal_length_factor 0.5 --ImageReader.camera_model OPENCV
    ```
    [**Command Line Arguments**](https://colmap.github.io/cli)

- Create a custom `matching.txt` file using:
    ```
    cd hierarchical_3d_gaussians
    python preprocess/make_colmap_custom_matcher.py --image_path <path to images> --output_path <matching.txt file path> 
    ```
    `<matching.txt file path>` will contain pairs of camera indices that are close using the image order and the gps data when it is available.
    <details>
    <summary><span style="font-weight: bold;">Command Line Arguments</span></summary>

    ### --n_gps_neighbours
    Number of closest neighbors to add if gps data is available (`25` by default).

    ### --n_loop_closure_match_per_view
    Number of matches to add to each camera provided in [loop_matches], it is not used when no custom [loop_matches] is provided (`5` by default).

    ### --image_path
    Path to input `images` folder.

    ### --output_path
    Path to output file with the image pairs to match.

    ### --n_seq_matches_per_view
    For each image, match with the next [n_seq_matches_per_view] images of the other cameras (`0` by default).
    ### --n_quad_matches_per_view
    For each image, add [n_quad_matches_per_view] matches that will range from $image$ to $image+2^{\text{n\_quad\_matches\_per\_view}}$ images of the other cameras (`10` by default).
    ### --loop_matches
    Custom matches that can be added by hand (`None` by default).
    </details>
    <br>

- The previously created `matching.txt` file will be used with the feature matching:
    ```
    cd ${DATASET_DIR}/unrectified
    colmap matches_importer --database_path <database.db> --match_list_path <matching.txt file path>
    ``` 
    [**Command Line Arguments**](https://colmap.github.io/cli)

- Launch the `hierarchical mapper` to create the scene colmap:
    > note that this step will take a lot more time than the previous steps, it took ~39 minutes on the example dataset.
    ```
    colmap hierarchical_mapper --database_path <database.db> --image_path <path to images> --output_path <sparse> --Mapper.ba_global_function_tolerance=0.000001 
    ```
    [**Command Line Arguments**](https://colmap.github.io/cli)

- Remove floating cameras and feature points that don't have sfm points, to make the colmap lighter:
    ```
    cd hierarchical_3d_gaussians
    python preprocess/simplify_images.py --base_dir ${DATASET_DIR}/unrectified/sparse/0
    ```
    <details>
    <summary><span style="font-weight: bold;">Command Line Arguments</span></summary>

    ### --base_dir
    Path to a colmap folder having `images`, `cameras` and `points3D` files.

    ### --mult_min_dist
    Points at distance > [mult_min_dist]*median_dist_neighbors are removed (`10` by default).

    ### --model_type
    Colmap model type, can be either `bin` or `txt` (`bin` by default).

    </details>
    <br>

- Undistort calibrated cameras, resulting images will be used during training:
    ```
    cd ${DATASET_DIR}
    colmap image_undistorter --image_path <path to images> --input_path <unrectified/sparse/0> --output_path <rectified> --output_type COLMAP --max_image_size 2048
    ```
    *If alpha masks are used, they should be undistorted the same way as images. Please find instructions on how to do it in the `generate_colmap.py` script.*

    [**Command Line Arguments**](https://colmap.github.io/cli)



- Align and scale the rectified colmap:<br>
    The rectified colmap is aligned and scaled to be metric so that it can be easily cut into chunks later.
    ```
    cd hierarchical_3d_gaussians
    python preprocess/auto_reorient.py --input_path <project_dir/rectified/sparse> --output_path <project_dir/aligned>
    ```

    <details>
    <summary><span style="font-weight: bold;">Command Line Arguments</span></summary>

    ### --input_path
    Path to input colmap dir.

    ### --output_path
    Path to output colmap dir.
    
    ### --upscale
    Custom upscaling factor, is automatically computed if not set (`0` by default).

    ### --target_med_dist
    Median distance of all calibrated cameras to their 3D points for the scene to be roughly metric, experimentally found. Ignored if `upscale` is set (`20` by default)
    
    ### --model_type
    Colmap model type, can be either `bin` or `txt` (`bin` by default).

    </details>
    <br>

## Generating chunks

The last preprocessing step is to divide the colmap into chunks, each chunk will have its own colmap that will be refined with two rounds of bundle adjustment and triangulation:

- Cut the calibration under `project/camera_calibration/aligned` into chunks, each chunk has its own colmap:
    ```
    python preprocess/make_chunk.py --base_dir <project/aligned/sparse/0> --images_dir <project/rectified/images> --output_path <project/raw_chunks>
    ```
    <details>
    <summary><span style="font-weight: bold;">Command Line Arguments</span></summary>

    ### --base_dir
    Path to input colmap dir.

    ### --images_dir
    Path to rectified images.
    
    ### --chunk_size
    Chunks are cubes of extents [chunk_size] (`100` by default).

    ### --min_padd
    Padding for the global bounding box (`0.2` by default).

    ### --lapla_thresh
    Threshold to discard blurry images: if their laplacians are < mean - lapla_thresh * std (`1` by default).

    ### --min_n_cams
    Min nb of cameras in each chunk (`100` by default).

    ### --max_n_cams
    Max nb of cameras in each chunk (`2000` by default).
    
    ### --output_path
    Path to output chunks folder.

    ### --add_far_cams
    Add cameras that are far from the chunk (`False` by default).

    ### --model_type
    Colmap model type, can be either `bin` or `txt` (`bin` by default).

    </details>
    <br>

- Refine each chunk by applying two rounds of `triangulation` and `bundle adjustment`:
    ```
    ## do this for each chunk
    python preprocess/prepare_chunk.py --raw_chunk <path to raw chunk> --out_chunk <path to output chunk> --images_dir <project/rectified/images> --depths_dir <project/rectified/depths> --preprocess_dir <path to hierarchical_gaussians/preprocess_dir>
    ```
    <details>
    <summary><span style="font-weight: bold;">Command Line Arguments</span></summary>

    ### --preprocess_dir
    Path to repo/preprocess.

    ### --images_dir
    Path to rectified images.

    ### --depths_dir
    Path to depths.
    
    ### --raw_chunk
    Path to the unrefined chunk.
    
    ### --out_chunk
    Path to the output refined chunk.

    </details>
    <br>


## Monocular depth maps
Make sure to have the [depth estimator weights](#weights-for-monocular-depth-estimation).
1. #### Generate depth maps (should run for each subfolder in `images/`)
    * Using Depth Anything V2 (prefered):
        ```
        cd submodules/Depth-Anything-V2
        python run.py --encoder vitl --pred-only --grayscale --img-path [path_to_input_images_dir] --outdir [path_to_output_depth_dir]
        ```
    * Using DPT: 
        ```
        cd submodules/DPT
        python run_monodepth.py -t dpt_large -i [path_to_input_images_dir] -o [path_to_output_depth_dir]
        ```

2. #### Generate `depth_params.json` file from the depth maps created on step *1.* 
      *this file will be used for the depth regularization for single chunk training. It needs to be generated for each chunk.* 
      ```bash
      cd ../../
      python preprocess/make_depth_scale.py --base_dir [path to colmap] --depths_dir [path to output depth dir]
      ```

## Training steps

Make sure that you correctly [set up repositories and environments](#setup)

- **Coarse optimization**<br>
   To allow consistent training of all chunks, we create a basic scaffold and skybox for all ensuing steps:
    ```
    python train_coarse.py -s <path to project/aligned> -i <../rectified/images> --skybox_num 100000 --position_lr_init 0.00016 --position_lr_final 0.0000016 --model_path <path to output scaffold>
    ```
    
- **Single chunk training**<br>
It is recommended to train using depth regularization to have better results, especially if your scene contains textureless surfaces such as roads. Make sure you [generated depth maps](#monocular-depth-maps)
    ```
    python -u train_single.py -s [project/chunks/chunk_name] --model_path [output/chunks/chunk_name] -i [project/rectified/images] -d [project/rectified/depths] --alpha_masks [project/rectified/masks] --scaffold_file [output/scaffold/point_cloud/iteration_30000] --skybox_locked --bounds_file [project/chunks/chunk_name]    
    ```

- **Per chunk hierarchy building**<br>
*Make sure you followed the [steps to generate the hierarchy creator executable file](#compiling-hierarchy-generator-and-merger)*.
Now we will generate a hierarchy in each chunk:
    ```
    # Linux: 
    submodules/gaussianhierarchy/build/GaussianHierarchyCreator [path to output chunk point_cloud.ply] [path to chunk colmap] [path to output chunk] [path to scaffold]

    # Windows:
    submodules/gaussianhierarchy/build/Release/GaussianHierarchyCreator.exe [path to output chunk point_cloud.ply] [path to chunk colmap] [path to output chunk] [path to scaffold]
    ```

- **Single chunk post-optimization**<br>
    ```
    python -u train_post.py -s [project/chunks/chunk_name] --model_path [output/chunks/chunk_name] --hierarchy [output/chunks/chunk_name/hierarchy_name.hier]  --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --save_iterations -1 -i [project/rectified/images] --alpha_masks [project/rectified/masks] --scaffold_file [output/scaffold/point_cloud/iteration_30000] --skybox_locked --bounds_file [project/chunks/chunk_name]    
    ```

- **Consolidation**
*Make sure you followed the [steps to generate the hierarchy merger executable file](#compiling-hierarchy-generator-and-merger)*.
Now we will consolidate and merge all the chunk hierarchies:
    ```
    # Linux:
    submodules/gaussianhierarchy/build/GaussianHierarchyMerger [path to output/trained_chunks] "0" [path to chunk colmap] [path to output merged.hier] [list of all the chunk names] 
    
    # Windows:
    submodules/gaussianhierarchy/build/Release/GaussianHierarchyMerger.exe [path to output/trained_chunks] "0" [path to chunk colmap] [path to output merged.hier] [list of all the chunk names] 
    ```

## Slurm parameters
The beginning of each `.slurm` script must have the following parameters:
```bash
#!/bin/bash

#SBATCH --account=xyz@v100      # your slurm account (ex: xyz@v100)
#SBATCH --constraint=v100-32g   # the gpu you require (ex: v100-32g)
#SBATCH --ntasks=1              # number of process you require
#SBATCH --nodes=1               # number of nodes you require 
#SBATCH --gres=gpu:1            # number of gpus you require
#SBATCH --cpus-per-task=10      # number of cpus per task you require
#SBATCH --time=01:00:00         # maximal allocation time
``` 
Note that the slurm scripts have not been thouroughly tested.

# Evaluations 评价指标
We use a test.txt file that is read by the dataloader and splits into train/test sets when `--eval` is passed to the training scripts. This file should be present in `sprase/0/` for each chunk and for the aligned "global colmap" (if applicable).

他的test.txt在他的处理好的small city里面有。大概就是30行的数据。每一行都类似于：
```
pass1_0213.png
pass1_0743.png
pass2_0087.png
...
```
很随机的选取的，上至一千四百多，下至0036。他的images文件夹有1500个。并没有分cam。

300张图片的话创建个10个的test就差不多了？放在east_gate/camera_calibration/aligned/sparse/0/test.txt。我也不知道为什么small city是png明明png会TiffByteOrder。如果自己的图片是放在cam1中那么格式最好是cam1/frame_00030.jpg

cnm训了一小时发现east_gate/camera_calibration/chunks/0_1/sparse/0/test.txt也需要

east_gate/camera_calibration/chunks/1_0/sparse/0/test.txt也需要

应该就chunks和aligned里面的sparse 0需要了。这个需要重新训练，所以很费时间，需要上午九点到下午一两点。而且不知道为什么，好像放了300张图片只能训练209张。正常情况下不写eval参数的full train也是只读取209张。

### Single chunk  不用使用single chunk即使只有一个cam使用large scenes
The single chunks we used for evaluation: 
* [SmallCity](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/standalone_chunks/small_city.zip) 

To run the evaluations on a chunk:
```
python train_single.py -s ${CHUNK_DIR} --model_path ${OUTPUT_DIR} -d depths --exposure_lr_init 0.0 --eval --skip_scale_big_gauss

# Windows: build/Release/GaussianHierarchyCreator 
submodules/gaussianhierarchy/build/GaussianHierarchyCreator ${OUTPUT_DIR}/point_cloud/iteration_30000/point_cloud.ply ${CHUNK_DIR}  ${OUTPUT_DIR} 

python train_post.py -s ${CHUNK_DIR} --model_path ${OUTPUT_DIR} --hierarchy ${OUTPUT_DIR}/hierarchy.hier --iterations 15000 --feature_lr 0.0005 --opacity_lr 0.01 --scaling_lr 0.001 --eval

python render_hierarchy.py -s ${CHUNK_DIR} --model_path ${OUTPUT_DIR} --hierarchy ${OUTPUT_DIR}/hierarchy.hier_opt --out_dir ${OUTPUT_DIR} --eval
```

### Large scenes 运行full train然后render hierarchy就完事了
Ensure that the test.txt is present in all `sparse/0/` folders. `preprocess/copy_file_to_chunks.py` can help copying it to each chunk.
Then, the scene can be optimized with `eval`:
```
DATASET_DIR=<Path to your dataset>
python scripts/full_train.py --project_dir ${DATASET_DIR} --extra_training_args '--exposure_lr_init 0.0 --eval'
```

如果报 assert False, "Could not recognize scene type!"，可以去相关程序上看看是找不到哪个文件，大概率是目录设置问题

render_hierarchy有些许问题。会导致test.txt即使写的没问题（cam1/xxx.jpg）也会报0test images，而同样的txt在full train时可以6test images。可以关注下scene/dataset_readers.py里的readColmapCameras

把scene/init里的if os.path.exists(os.path.join(args.source_path, "sparse"))改为if os.path.exists(os.path.join(args.source_path, "camera_calibration/aligned/sparse"))
scene/dataset——readers里的cameras_extrinsic_file与cameras_intrinsic_file的路径都由 "sparse/0"改为 "camera_calibration/aligned/sparse/0"
以上该路径的方式与下面方式二选一执行（选择改路径的话还会有一堆需要改可能）：
将camera calibbration里的aligned文件夹复制到east gate文件夹下
把images文件夹复制到east gate中去


The following renders the test set from the optimized hierarchy. Note that the current implementation loads the full hierarchy in GPU memory.
```
DATASET_DIR=data/data2/east_gate/
 model_path=data/data2/east_gate/output/scaffold/

python render_hierarchy.py -s ${DATASET_DIR}/camera_calibration/aligned --model_path ${model_path} --hierarchy ${DATASET_DIR}/output/merged.hier --out_dir ${DATASET_DIR}/output/renders --eval --scaffold_file ${DATASET_DIR}/output/scaffold/point_cloud/iteration_30000
```

### Exposure optimization
We generally disable exposure optimization for evaluations. If you want to use it, you can optimize exposure on the left half of the test image and evaluate on their right half. To achieve this, remove `--exposure_lr_init 0.0` from the commands above and add `--train_test_exp` to all training scripts.
