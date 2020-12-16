Notes DEEP_JPEG:

- comparaison de deux modèles en vitesse d'inférence

ssd_rgb
ssd_dct (full)

- Générateurs de données

    - pour dct, image chargée via Pip.Image.Open, convert RGB, np.array(image), apply transforms (resize (rgb)), Image.fromarray save, puis jpeg2dct.numpy.loads() (retourne band1, band2, band3)


- Pour VGG, générateur de DCT 

- time_prediction : pourquoi seulement une taille de batch de 1 ? (gros bottleneck)

- Export sous TensorRT ?


Etapes : 

	- jouer inférence (dataset a intégrer)
	- tester temps d'inférence
	- trouver bottlenecks
	- utiliser jpeg2dct TF, comparer
	- exporter TensorRT


- Tests

Besoin d'installer lxml pour faire fonctionner beautifulSoup

16/12 - Comparaison RGB / DCT -> goulot en prédiciton pour DCT (5it/s vs 30 it/s pour RGB)
Problème de resize systematique des images ?

### Ligne de commande pour lancement étude comparative : 
/opt/nvidia/nsight-systems/2019.6.1/bin$ ./nsight-sys 

/home/ysc/projects/APT/AIXOM/jpeg_deep/venv2/bin/python3.7 /home/ysc/projects/APT/AIXOM/jpeg_deep/scripts/prediction.py  /mnt/dcaf7e38-46ed-4126-a9a1-7d80df393639/Sandbox /mnt/dcaf7e38-46ed-4126-a9a1-7d80df393639/weights/jpeg_deep/detection_dct/vgg/07/ssd_jpeg_deep_5gElSUOnUPiFuifcgcOOxLhBNC3TZRak/checkpoints/epoch-47_loss-4.5323_val_loss-6.5883.h5 /mnt/dcaf7e38-46ed-4126-a9a1-7d80df393639/weights/jpeg_deep/detection_rgb/vgg/own_training/07/ssd_jpeg_deep_3pQjrAJq2vapGVtHzUzhATLoSvVwWAYq/checkpoints/epoch-37_loss-3.7112_val_loss-6.2008.h5

### Tests de throughput
(1x GTX 1080)

Batch size = 8
  Througput DCT w NMS : 101.88435107309694 imgs/s
  Througput RGB w NMS : 69.84226038189922 imgs/s
  Througput DCT w/o NMS : 146.93217555676733 imgs/s
  Througput RGB w/o NMS : 91.93245584570333 imgs/s


Batch size = 32
  Througput DCT w NMS : 152.15985686322264 imgs/s
  Througput RGB w NMS : 90.90917871909308 imgs/s
  
  Througput DCT w/o NMS: 227.01979690667082 imgs/s
  Througput RGB w/o NMS : 108.72085665239989 imgs/s

Batch size = 128
  Througput DCT w NMS : 166.3404487886101 imgs/s
  Througput RGB w NMS : 93.92361192642025 imgs/s
  Througput DCT w/o NMS : 242.3443284639605 imgs/s
  Througput RGB w/o NMS : 107.56113791885736 imgs/s
