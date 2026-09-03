# FoundationPose Custom

이 fork는 upstream [FoundationPose](https://github.com/NVlabs/FoundationPose)에 실행기
3개를 추가합니다: 추적 실행기(첫 프레임 Pose Estimation → 이후 Tracking, **실증용**),
프레임당 물체 1개인 단일 객체 실행기와 여러 개인 멀티 객체 실행기(**데이터셋
구축용** — 가상/실환경 자동 판별, 추적 없이 매 프레임 독립적으로 Pose Estimation).
기본 설치는 upstream [`readme.md`](readme.md)를 따르고, 이 문서는 추가된 실행기만
다룹니다.

## Installation

[`readme.md`](readme.md)의 "Data prepare" 항목을 확인하여 모델 체크포인트(`weights/`)를 다운로드하세요.

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

# PyTorch 설치 (CUDA 12.4, 버전 고정)
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# CUDA 툴킷 설치. label 채널로 버전을 고정해야 하위 패키지도 고정됨
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit

# gcc/g++ 13.4.0으로 다운그레이드 (CUDA 12.4 nvcc는 gcc 13까지만 지원)
conda install -y -c conda-forge "gcc=13.4.0" "gxx=13.4.0"

# PyTorch3D / NVDiffRast 소스 빌드 (커밋 고정)
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@2bce7110d5621ef2a349a72ac3aacbfba8bfb461"
python -m pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae"

# 나머지 의존성 설치(이미 버전 고정된 requirements.txt) + mycpp 빌드
python -m pip install -r requirements.txt
bash build_all_conda.sh
```

```bash
# 환경 활성화
conda activate foundationpose
```

---

## 세 실행기 비교

| | 용도 | 처리 방식 | 프레임당 객체 수 | 가상/실환경 | 상태 |
|---|---|---|---|---|---|
| `run_custom_demo.py` | 실증 | 첫 프레임만 Pose Estimation, 이후는 Tracking | 1개 | 계획: 실증(실시간 연속 프레임)용으로 개편 예정 — 아직 미구현, 지금은 가상환경 전용 코드 그대로 | 검증용으로 만든 원본 코드 |
| `run_one_object_demo.py` | 데이터셋 구축 | 매 프레임 독립적으로 Pose Estimation (Tracking 없음) | 1개 | 둘 다 지원 (자동 판별) | 사용 가능 |
| `run_multi_object_demo.py` | 데이터셋 구축 | 매 프레임 독립적으로 Pose Estimation (Tracking 없음) | 여러 개 | 둘 다 지원 (자동 판별) | 사용 가능 |

`run_custom_demo.py`는 실시간으로 연속된 프레임이 들어온다는 전제(실제 로봇 조작 중
추적) 하에 쓸 실행기로 개편할 계획입니다. 지금 real 배치 데이터셋(`real_v1/...`)은
frame마다 다른 물체가 섞여 있어서 이 전제가 안 맞기 때문에, 이 데이터셋에는
`run_one_object_demo.py`/`run_multi_object_demo.py`를 씁니다.

가상/실환경 판별은 CLI 인자가 아니라 각 프레임의 `conf/<frame_id>.json`에 있는
`"domain"` 값(`"real"`이면 real, 그 외는 virtual)을 스크립트가 프레임마다 자동으로
읽어서 정합니다.

---

## Run

아래 명령어의 경로는 호스트 기준입니다. Docker로 설치했다면 컨테이너 안에서는
마운트 경로가 다릅니다 — 호스트 `/media/uon/data1` → 컨테이너 `/media/uon/data`.

### 1. 추적 실행기 — `run_custom_demo.py`

가상환경에서, 첫 프레임만 Pose Estimation하고 이후 프레임은 Tracking으로
이어갑니다.

`sample/cough/mesh/`는 CAD 파일이 커서(100MB 넘음) git에 없습니다. 아래 명령어를
쓰려면 `3d_model/peel3_scan_data_2025/paper_cup/`에서
`paper_cup.obj`, `paper_cup.mtl`, `paper_cup_edited.bmp` 세 파일을 직접
`sample/cough/mesh/`에 복사해 넣어주세요.

```bash
python run_custom_demo.py \
  --mesh_file sample/cough/mesh/paper_cup.obj \
  --camera_name top_view_camera \
  --debug_dir outputs \
  --no_gui
```

결과는 `--debug_dir`로 지정한 폴더 안에 저장됩니다.

### 2. 단일 객체 실행기 — `run_one_object_demo.py`

프레임마다 물체가 하나뿐인 경우에 씁니다 (real 배치 데이터셋은 항상 이 경우). 카메라 한 대만 처리합니다(`--camera_name`, 기본값 `top_view_camera`).

```bash
python run_one_object_demo.py \
  --dataset-root /media/uon/data1/gemini \
  --scene real_v1/home/LivingRoom_Kitchen/dining_table
```

결과는 scene 폴더 안에 카메라 구분 없이 저장됩니다.

- `6d_pose/`, `6d_pose_json/` — 계산된 pose (같은 값을 txt/json 두 형식으로)
- `diagnostics/foundationpose/track_vis/` — 바운딩박스+좌표축을 그려 넣은 확인용 이미지
- `diagnostics/foundationpose/overlay/` — CAD mesh를 그 pose로 실제 렌더링해서 원본 위에
  합성한 이미지 (CAD가 실물과 실제로 맞는지 확인용)
- `diagnostics/foundationpose/stitched/` — 원본과 overlay를 위아래로 이어붙인 비교 이미지
- `6d_pose_debug/` — 문제 생겼을 때 보는 상세 기록
- `inference_meta/foundationpose/cad_asset_issues.jsonl` — 실패한 항목 목록

### 3. 멀티 객체 실행기 — `run_multi_object_demo.py`

프레임 하나에 물체가 여러 개 있을 수 있는 경우에 씁니다. 인자 구성은 단일 객체
실행기와 동일합니다.

```bash
python run_multi_object_demo.py \
  --dataset-root /media/uon/data1/gemini \
  --scene real_v1/home/LivingRoom_Kitchen/dining_table
```

특정 프레임만 처리하려면 `--frame_id`를 추가합니다.

```bash
python run_multi_object_demo.py \
  --dataset-root /media/uon/data1/gemini \
  --scene real_v1/home/LivingRoom_Kitchen/dining_table \
  --frame_id 0002
```

결과는 scene 폴더 안에, 단일 객체 실행기 결과와 절대 안 섞이도록 전부
`_multi`(또는 `multi/`) 표시를 붙여서 저장됩니다. 한 프레임에 물체가 여러 개면 한
파일 안에 class_id별로 나열합니다.

- `6d_pose_multi/<frame_id>.txt`, `6d_pose_multi_json/<frame_id>.json` — 그 프레임의
  모든 객체 pose (json은 class_id를 키로 하는 딕셔너리, txt는 `# <class_id>` 줄로
  구분된 블록)
- `diagnostics/foundationpose/multi/combined/` — 한 프레임의 모든 객체를 한 장에 합쳐
  그린 이미지
- `diagnostics/foundationpose/multi/track_vis/<frame_id>/` — 객체별 개별 바운딩박스 오버레이 이미지
- `diagnostics/foundationpose/multi/overlay/<frame_id>/` — 객체별 CAD mesh 렌더 합성 이미지
- `diagnostics/foundationpose/multi/stitched/<frame_id>/` — 객체별 원본+overlay 비교 이미지
- `6d_pose_multi_debug/` — 문제 생겼을 때 보는 상세 기록 (객체별 폴더 분리)
- `inference_meta/foundationpose/cad_asset_issues_multi.jsonl` — 실패한 항목 목록

(SAM3 결과는 `diagnostics/sam3/...`, `inference_meta/sam3/...`로 따로 저장돼서 안 섞입니다.)

---

## Reference

train_data:
https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-

model_free_ref_views:
https://drive.google.com/drive/folders/1PXXCOJqHXwQTbwPwPbGDN9_vLVe0XpFS

BOP Benchmark for 6D Object Pose Estimation:
https://bop.felk.cvut.cz/leaderboards/pose-estimation-unseen-bop23/core-datasets/
