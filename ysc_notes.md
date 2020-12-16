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
/home/ysc/projects/APT/AIXOM/jpeg_deep/venv2/bin/python3.7 /home/ysc/projects/APT/AIXOM/jpeg_deep/scripts/prediction.py  /mnt/dcaf7e38-46ed-4126-a9a1-7d80df393639/Sandbox /mnt/dcaf7e38-46ed-4126-a9a1-7d80df393639/weights/jpeg_deep/detection_dct/vgg/07/ssd_jpeg_deep_5gElSUOnUPiFuifcgcOOxLhBNC3TZRak/checkpoints/epoch-47_loss-4.5323_val_loss-6.5883.h5 /mnt/dcaf7e38-46ed-4126-a9a1-7d80df393639/weights/jpeg_deep/detection_rgb/vgg/own_training/07/ssd_jpeg_deep_3pQjrAJq2vapGVtHzUzhATLoSvVwWAYq/checkpoints/epoch-37_loss-3.7112_val_loss-6.2008.h5