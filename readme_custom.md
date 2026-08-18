## Installation

### 1. Env setup option 1: docker

```bash
# nvidia-container-toolkit 설치가 전제되어 있어야 함

# 이미지 다운로드
docker pull shingarey/foundationpose_custom_cuda121:latest

# run_container.sh가 참조하는 이름(foundationpose:latest)으로 태깅
docker tag shingarey/foundationpose_custom_cuda121:latest foundationpose:latest

# 컨테이너 생성 + 실행 (최초 1회)
bash docker/run_container.sh
```

```bash
# 컨테이너 재접속 (이미 생성된 경우)
docker start foundationpose && docker exec -it foundationpose bash
```

### 2. Env setup option 2: conda

```bash
conda env create -f environment.yml
conda activate foundationpose

# PyTorch 설치 (CUDA 12.4)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 툴킷 설치. label 채널로 버전을 고정해야 하위 패키지도 고정됨
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit

# gcc/g++ 13.4.0으로 다운그레이드 (CUDA 12.4 nvcc는 gcc 13까지만 지원)
conda install -y -c conda-forge "gcc=13.4.0" "gxx=13.4.0"

# PyTorch3D / NVDiffRast 소스 빌드
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"

# 나머지 의존성 설치 + mycpp 빌드
python -m pip install -r requirements.txt
bash build_all_conda.sh
```

```bash
# 환경 활성화
conda activate foundationpose
```

---

## Run

단일 객체 실행기 : 첫 프레임에서 Pose Estimation -> 이후 프레임에서 Tracking (실증에서 활용)  
멀티 객체 실행기 : 전체 프레임·객체에 대해 Pose Estimation (데이터 구축에서 활용) 

### 1. 가상환경

단일 객체 실행기
```bash
python run_custom_demo.py \
  --cad_name paper_cup \
  --camera_name top_view_camera \
  --debug_dir debug_cough \
  --no_gui
```

멀티 객체 실행기 (전체 프레임 실행)
```bash
python run_multi_object_demo.py \
  --camera_name top_view_camera \
  --debug_dir debug_cough \
  --no_gui
```

멀티 객체 실행기 (특정 프레임 실행)
```bash
python run_multi_object_demo.py \
  --camera_name top_view_camera \
  --frame_id 0002 \
  --debug_dir debug_cough \
  --no_gui
```

### 2. 실환경

멀티 객체 실행기 (전체 프레임 실행)
```bash
# 호스트와 컨테이너 마운트 경로는 다름
# 호스트 /media/uon/data1, 컨테이너 /media/uon/data
python run_multi_object_demo.py \
  --scene_dir /media/uon/data1/gemini/real_v1/Home/LivingRoom_Kitchen/dining_table \
  --camera_name top_view_camera \
  --cad_root /media/uon/data1/3d_model/peel3_scan_data_2026 \
  --objects_metadata /media/uon/data1/gemini/objects_metadata.csv \
  --debug_dir debug_real \
  --no_gui
```

---

## Reference

train_data:
https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-

model_free_ref_views:
https://drive.google.com/drive/folders/1PXXCOJqHXwQTbwPwPbGDN9_vLVe0XpFS

BOP Benchmark for 6D Object Pose Estimation:
https://bop.felk.cvut.cz/leaderboards/pose-estimation-unseen-bop23/core-datasets/
