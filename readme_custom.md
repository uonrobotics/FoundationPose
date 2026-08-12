## 가장 먼저 읽기

호스트 환경에서 Run 부분을 진행한다.
코드 수정은 VsCode 내에서 진행하고 호스트 환경에서 파일을 실행한다.
Codex 내용 참고하면, A 모드까지만 진행되어있음. (B/C/D 필요시 추가 구현)
Codex 내용 참고하면, instance segmentation 색상에 관한 내용이 있음.
1차 목적 (어댑터 및 실행기)에 의해서만 개발이 진행되었음.
2차는 실환경 데이터셋 (Scene Gen) 구축 이후 다시 진행.
1차에서 산출물(6D Pose)에 대해 실제 사용할만한 알고리즘인지 데이터 분석 작업을 거치지 않았음. 2차에서 진행 필요.


## 주의

데이터셋 용량이 매우 크다.


## 공통 기능

1. Old_name → Object_name 순으로 CAD 탐색
2. 자동 segmentation 색상 매칭
3. 표준 map_Kd 검사 (경로와 일치하는 경우만 6D Pose 계산)
4. 비표준 map_Kd 객체 건너뛰기
5. 존재하지 않는 map_Kd 객체 건너뛰기
6. 지원하지않는 2024 CAD 건너뛰기
7. 동일한 cad_asset_issues.json 로그 형식 (4, 5, 6에 한해서 로그 기록)


## 기능 차이

1. 단일 객체 실행기 : 첫 프레임에서 Pose Estimation -> 이후 프레임에서 Tracking (실증에서 활용) (run_custom_demo.py)
2. 멀티 객체 실행기 : 전체 프레임·객체에 대해 Pose Estimation (데이터 구축에서 활용) (run_multi_object_demo.py)


## 향후 작업

1. 2024 CAD DB는 모두 Old_name 규칙이다.
2. 2024는 texture 파일 규칙이 2025와 다르다.
-> 2024에는 <Old_name>_edited.bmp 파일이 edited 폴더 내에 있다.


## install

docker install

nvidia-container-toolkit install

```
docker pull shingarey/foundationpose_custom_cuda121:latest
```

```
docker tag shingarey/foundationpose_custom_cuda121:latest foundationpose:latest
```

```
bash docker/run_container.sh
```


## Check
```
docker ps
```

```
watch -n 1 nvidia-smi
```

호스트 마운트
```
ls /media/uon/data1/3d_model/peel3_scan_data_2025/paper_cup
```

컨테이너 마운트
```
ls /media/uon/data/3d_model/peel3_scan_data_2025/paper_cup
```

컨테이너 내부 파일 삭제
```
docker exec foundationpose \
  rm -rf /home/uon/workspace/FoundationPose/debug_cough
```


## Run

```
docker start foundationpose && docker exec -it foundationpose bash
```

```
cd /home/uon/workspace/FoundationPose
```

### 1. 가상환경

단일 객체 실행기
```
python run_custom_demo.py \
  --cad_name paper_cup \
  --camera_name top_view_camera \
  --debug_dir debug_cough \
  --no_gui
```

멀티 객체 실행기 (전체 프레임 실행)
```
python run_multi_object_demo.py \
  --camera_name top_view_camera \
  --debug_dir debug_cough \
  --no_gui
```

멀티 객체 실행기 (특정 프레임 실행)
```
python run_multi_object_demo.py \
  --camera_name top_view_camera \
  --frame_id 0002 \
  --debug_dir debug_cough \
  --no_gui
```

### 2. 실환경

멀티 객체 실행기 (전체 프레임 실행)
```
python run_multi_object_demo.py \
  --scene_dir /media/uon/data/gemini/real_v1/Home/LivingRoom_Kitchen/dining_table \
  --camera_name top_view_camera \
  --cad_root /media/uon/data/3d_model/peel3_scan_data_2026 \
  --objects_metadata /media/uon/data/gemini/objects_metadata.csv \
  --debug_dir debug_real \
  --no_gui
```

## Reference

train_data:
https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-

model_free_ref_views:
https://drive.google.com/drive/folders/1PXXCOJqHXwQTbwPwPbGDN9_vLVe0XpFS

BOP Benchmark for 6D Object Pose Estimation:
https://bop.felk.cvut.cz/leaderboards/pose-estimation-unseen-bop23/core-datasets/
