# **EAC : samshap**

[![Licence EUPL 1.2](https://img.shields.io/badge/licence-EUPL_1.2-blue)](https://interoperable-europe.ec.europa.eu/collection/eupl/eupl-text-eupl-12)

Implémentation du papier EAC : https://proceedings.neurips.cc/paper_files/paper/2023/file/44cdeb5ab7da31d9b5cd88fd44e3da84-Paper-Conference.pdf

## **Description :**

L'ensemble de ce dépôt dépeint l'implémentation de EAC (Explain Any Concept).

## **Usages :**

### **Installation :**

```bash
git clone https://github.com/merpo686/samshap_msia.git
cd samshap_msia
git clone https://github.com/facebookresearch/segment-anything.git SAM
sed -i \"s/torch.load(f)/torch.load(f, weights_only=True)/g\" SAM/segment_anything/build_sam.py
```

### **Exécution :**

#### **Usage en ligne de commande :**
```bash
python xai_samshap.py -h
usage: EAC [-h] [--task {run,test}] [--logfile [LOGFILE]] [--savefile [SAVEFILE]] [--nolog] [--checkpoint CHECKPOINT]
           [--device DEVICE] [--sam_type {vit_h,vit_l,vit_b}] [--model MODEL] [--output [OUTPUT]] [--input INPUT]
           [--shapley_mc SHAPLEY_MC] [--pie_mc PIE_MC] [--pie_epoch PIE_EPOCH]

Lance la XAI avec le EAC.

options:
  -h, --help            show this help message and exit
  --task {run,test}     [défaut=run] tâche à accomplir
  --logfile [LOGFILE]   sortie du programme
  --savefile [SAVEFILE]
                        spécifie s'il faut sauvegarder.
  --nolog               désactive les log
  --checkpoint CHECKPOINT
                        [défaut=checkpoints] dossier où sont les poids des modèles.
  --device DEVICE       [défaut=cpu] device où charger le modèle [auto, cpu, cuda, torch_directml]
  --sam_type {vit_h,vit_l,vit_b}
                        [défaut=vit_h] modèle VIT de SAM [vit_h, vit_l, vit_b]
  --model MODEL         [Défaut=resnet18] modèle à tester (de torchvision)
  --output [OUTPUT]     [défaut=results] chemin du dossier de sortie
  --input INPUT         chemin de l'image d'entrée (ou dossier de test en mode test)
  --shapley_mc SHAPLEY_MC
                        [défaut=10000] échantillonnage Monté-Carlo pour les valeurs de Shapley.
  --pie_mc PIE_MC       [défaut=2500] échantillonnage Monté-Carlo pour l'entraînement PIE.
  --pie_epoch PIE_EPOCH
                        [défaut=50] epoch pour l'entraînement PIE.

Exemples : python xai_samshap.py --input=dog.jpeg --sam_type vit_b --model=resnet18 --device=cuda
```
- **Exemple d'exécution EAC sur une image :**

`python xai_samshap.py --sam_type vit_b --model=resnet18 -input=dog.jpeg`

- **Exemple de test sur plusieurs images et sauvegarde des résultats**

`python xai_samshap.py --task=test --sam_type vit_b --model=resnet18 --input=data\imagenet_subsample --device=cuda --logfile --savefile`

  - les logs seront sauvegardés dans _'logs\<timestamp>\pipeline.log'_
  - les résultats seront sauvegardés dans _'results\<timestamp>\'_
    - _<input_test_folder>/<img_name>\_mask.jpg_ pour les masques d'explication
    - _EAC_values.csv_ pour les valeurs des AUC


#### **Usage Python :**

- ***CAS : Mêmes arguments qu'en ligne de commande :***
```python
import xai_samshap

args = vars(xai_samshap.parse_args())  # args par défaut
args["input"] = ...
model_eac = xai_samshap.run_process(args)
model_eac.results
  # model_eac.results["sam"]
  # model_eac.results["mask"]
```

## **Étudiants :**

| Name               | GitHub Profile                              |
|--------------------|---------------------------------------------|
| **Nicolas ALLÈGRE**| [nicolas-allegre](https://github.com/nicolas-allegre) |
| **Louis Borreill**   | [LouisBorr](https://github.com/LouisBorr) |
| **Merlin Poitou**   | [merpo686](https://github.com/merpo686) |
