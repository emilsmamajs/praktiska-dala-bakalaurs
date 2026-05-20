# Bakalaura darba praktiskā daļa
Aizseguma identificēšanas modeļu apmācība (YOLo26 un RF-DETR)
### BAKALAURS_DATASET/
Izveidotā datu kopa
- train/occluded/ - apmācības attēli ar aizsegtiem objektiem (465 attēli un JSON anotācijas)
- train/not_occluded/ - apmācības attēli ar neaizsegtiem objektiem (465 attēli un JSON anotācijas)
- val/occluded/ - vaalidācijas attēli ar aizsegtiem objektiem (116 attēli un JSON anotācijas)
- val/not_occluded/ - vaalidācijas attēli ar neaizsegtiem objektiem (116 attēli un JSON anotācijas)
### RF-DETR/
Viss kas saistīts ar RF-DETR
- compose_coco_dataset.py - python kods, kas parveido datu kopu nepieciešamajā COCO formātā mapē coco_dataset/
- coco_dataset/ - izveidotā datu kopa nepieciešamajā formātā un satur
  - train/
  - valid/ 
- iteracija1/, iteracija2/, iteracija3/, iteracija4/ - četras apmācības iterācijas ar dažādiem hiperparametriem
  - train,py - apmācības kods
  - Dockerfile un docker-compose.yml - konteinerizētai apmācībai
  - runs/train/ - apmācības rezultāti (checkpoint_best_ema.pth, metrics.csv)
- ### YOLO26/
Viss kas saistīts ar YOLO26
- compose_yolo_dataset.py - python kods, kas parveido datu kopu nepieciešamajā YOLO formātā mapē yolo_dataset/
- yolo_dataset/ - izveidotā datu kopa nepieciešamajā formātā un satur
  - images/train/
  - images/val/
  - labels/train/
  - labels/val/
  - data.yaml - datu kopas konfigurācija
- iteracija1/, iteracija2/, iteracija3/, iteracija4/ - četras apmācības iterācijas ar dažādiem hiperparametriem
  - train,py - apmācības kods
  - Dockerfile un docker-compose.yml - konteinerizētai apmācībai
  - runs/train/ - apmācības rezultāti (best.pt, results.csv, informatīvas ilustrācijas par treniņu norisi, u.t.t.)
