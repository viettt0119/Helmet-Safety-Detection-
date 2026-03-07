# Helmet Safety Detection (Python 3.14 - Local)

Project da duoc toi gian de chay on dinh voi Python 3.14 trong PyCharm.

## Cau truc hien tai

```text
JupyterProject/
|-- .venv/
|-- data/
|   |-- train/images, train/labels
|   |-- valid/images, valid/labels
|   |-- test/images, test/labels
|   `-- data.yaml
|-- models/
|   `-- yolov10n.pt
|-- train.py
|-- val.py
|-- inference.py
|-- requirements.txt
`-- README.md
```

## Tao moi truong (Python 3.14)

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Chay train

```powershell
python train.py --epochs 50 --imgsz 640 --batch 8 --device 0
```

Neu may yeu/GPU VRAM it, giam `--batch` xuong `4`.

## Chay evaluate (test split)

```powershell
python val.py --weights .\runs\detect\helmet_yolov10n\weights\best.pt --split test --imgsz 640 --device 0
```

## Chay inference

### Anh
```powershell
python inference.py --weights .\runs\detect\helmet_yolov10n\weights\best.pt --source .\data\test\images\helmet-10-_jpg.rf.eee985865a1a52407316f0c9ea7a9b3e.jpg --conf 0.3
```

### Video
```powershell
python inference.py --weights .\runs\detect\helmet_yolov10n\weights\best.pt --source .\sample.mp4 --conf 0.3
```

### Webcam
```powershell
python inference.py --weights .\runs\detect\helmet_yolov10n\weights\best.pt --source 0 --conf 0.3
```

Nhan `q` hoac `ESC` de thoat khoi video/webcam.

## Ghi chu quan trong

- Script da co `if __name__ == '__main__':` de tranh loi multiprocessing tren Windows.
- Trong so `yolov10n.pt` duoc tu dong tai ve thu muc `models/` neu chua co.
