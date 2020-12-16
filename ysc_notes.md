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