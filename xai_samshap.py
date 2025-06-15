"""BGDIA7065 Projet XAI, samshap (EAC).

Sujet : https://proceedings.neurips.cc/paper_files/paper/2023/file/44cdeb5ab7da31d9b5cd88fd44e3da84-Paper-Conference.pdf

Implémentation du modèle EAC (Explain Any Concept).


MEMO vérif code : python -m pylama -l all --pydocstyle-convention pep257 *.py
    # python -m pylama -l eradicate,mccabe,pycodestyle,pydocstyle,pyflakes,pylint,radon,vulture,isort --pydocstyle-convention pep257
    # python -m mypy
    # python -m autopep8 -a -p2 -r -v [-i | --diff]
    # python -m isort --diff
# MEMO console :
import importlib
importlib.reload(p)
"""
__author__ = ['Nicolas Allègre', 'Louis Borreill', 'Merlin Poitou']
__date__ = '11/06/2025'
__version__ = '0.2'

###############################################################################
# IMPORTS :
# /* Modules standards */
import argparse
import glob
import logging
import os
import shlex
import sys
import time
from collections.abc import Iterable
from typing import Any, Final, Literal

# /* Modules externes */
import numpy as np
import numpy.typing as npt
from sklearn.metrics import auc
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from PIL import Image
from pprint import pprint
from tqdm import tqdm

# /* Module interne */
IA_AGENT_SAM_FOLDER = "SAM"
sys.path.append(IA_AGENT_SAM_FOLDER)
from SAM.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
###############################################################################
# CONSTANTES :
EXIT_OK: Final[int] = 0

DEFAULT_SAVE_MODEL_FILENAME: Final[str] = "pie.pth"
DEFAULT_SAVE_FOLDER: Final[str] = "results"
DEFAULT_LOG_FOLDER: Final[str] = "logs"
DEFAULT_LOG_FILENAME: Final[str] = "pipeline.log"
DEFAULT_MODEL_FOLDER: Final[str] = "checkpoints"
DEFAULT_SAM_MODEL: Final[str] = "vit_b"
DEFAULT_MODEL_TEST: Final[str] = "resnet50"
DEFAULT_SHAPLEY_MC_SAMPLING: Final[int] = 50_000
DEFAULT_PIE_MC_SAMPLING: Final[int] = 2_500
DEFAULT_PIE_EPOCH: Final[int] = 50

DEVICE_GLOBAL: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_RESIZE = [transforms.Resize(256), transforms.CenterCrop(224)]
IMG_CONVERT_TENSOR = [transforms.ToImage(
), transforms.ToDtype(torch.float32, scale=True)]
IMG_NORMALIZE = [transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

DEFAULT_TRANSFORM_BEGIN = transforms.Compose(IMG_RESIZE)
DEFAULT_TRANSFORM_NORM = transforms.Compose(IMG_CONVERT_TENSOR + IMG_NORMALIZE)
DEFAULT_TRANSFORM_IMAGENET = transforms.Compose(
    IMG_RESIZE + IMG_CONVERT_TENSOR + IMG_NORMALIZE)


###############################################################################
# CLASSES PROJET :
class Model_PIE(torch.nn.Module):
    """Le modèle PIE d'EAC.
    """

    def __init__(self, pie_args: dict | None = None, model_fc=None, logger_master=None):
        """Initialise le PIE.

        :param (dict) pie_args:     argument de configuration du PIE
        :param model_fc:            la dernière couche du modèle à expliquer.
        """
        super().__init__()

        self.logger = logger_master
        if pie_args is None:
            pie_args = {}
        dim_in = pie_args.get("dim_in")
        dim_fc = pie_args.get("dim_fc")
        if dim_in is None or dim_fc is None:
            return None

        self.linear = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_fc))
        self.model_fc = model_fc

    def forward(self, x, with_fc: bool = True):
        """Effectue l'inférence du modèle.

        :param x:   l'entrée
        :param (bool) with_fc:  détermine s'il faut prédire avec fc comme dernière couche
        :return:    la prédiction
        """
        out = self.linear(x)
        if self.model_fc is not None and with_fc:
            out = self.model_fc(out)
        return out

    def predict_prob(self, x):
        """Prédiction et donne les probas.

        :param x:   l'entrée
        :return:    les probabilités de prédiction (de chaque classe)
        """
        out = self(x, with_fc=True)
        prob = torch.nn.functional.softmax(out, dim=1)
        return prob

    def logging(self, msg: str, level: int | str = logging.INFO, **kwargs):
        """Permet de logger.

        :param (str) msg:   message à logger
        :param (int | str) level: [défaut INFO] niveau de log (cf. logging)
        """
        if self.logger is not None:
            self.logger.logging(
                msg, level=level, caller_name=self.__class__.__name__, **kwargs)
        else:
            print(msg)

    def training_pie(self, f_model, image, list_of_mask, transform_img=None,
                     n_samples: int = 10, num_epochs: int = 10, device=DEVICE_GLOBAL, label=None):
        """Entraîne ce PIE pour une image et model donnée.

        Génère un jeu d'image d'entraînement (n image composé de sous-set de mask),
        et récupère leurs probabilitiés de prédiction, puis entraine le PIE sur
        ce trainset.

        :param f_model:         le modèle cible
        :param image:           l'image pré-tranformée
        :param list_of_mask:    les masques pour l'image
        :param transform_img:   la torchvision transforms à utiliser avec le réseau cible
        :param n_samples:       nombre d'échantillon de sous-image masqué pour l'entraînement
        :param num_epochs:      quantité d'epoch
        :param device:          le device d'exécution
        """
        self.logging(f"PIE : Début entraînement {n_samples=}, {num_epochs=}.")
        start_time = time.time()
        if transform_img is None:
            transform_img = DEFAULT_TRANSFORM_IMAGENET

        f_model = f_model.to(device).eval()
        model_pie = self.to(device).train()

        n_concept = len(list_of_mask)
        fc_layer = f_model.fc
        for param in model_pie.model_fc.parameters():
            param.requires_grad = False

        # 1-Création des images masqués par sous-ensemble pour l'entraînement du PIE
        # 1.1 création liste de tous les sous-ensemble
        # idx_mask_combinations = [list(combinations(range(n_concept), r)) for r in range(1, n_concept+1)]
        # idx_mask_combinations = [list(sublist) for g in idx_mask_combinations for sublist in g]
        # Changement, nombre de sous-ensemble est l'arrangement donc n! et donc au-delà de 20 Python ne gère pas et MemoryError
        self.logging(
            f"PIE : Création liste de sous-ensemble de masque via MC ({n_samples=})")
        list_masks = self._image_to_masks_mc(
            image, list_of_mask, n_samples)

        # 2-Récupération des probabilités de prédiction du modèle
        self.logging("PIE : Prédiction du modèle f_model sur cette liste")
        target_probabilities = []
        list_images_masked = []
        for mask_vec in list_masks:
            indices = np.where(mask_vec == 1)[0]
            if len(indices) == 0:
                fused_mask = np.zeros_like(list_of_mask[0], dtype=bool)
            else:
                fused_mask = np.logical_or.reduce([list_of_mask[i] for i in indices])
            list_images_masked.append(image * fused_mask[:, :, np.newaxis])
        
        for masked_image in list_images_masked:
            input_tensor = transform_img(masked_image).to(device)
            with torch.no_grad():
                output = f_model(input_tensor.unsqueeze(0))
                probabilities = torch.softmax(output, dim=1)[:, label]
                target_probabilities.append(probabilities.squeeze().cpu().numpy())

        # 3-Préparation des données d'entraînement
        input_data = list_masks

        input_data = torch.tensor(np.array(input_data), dtype=torch.float32).to(device)
        target_probabilities = torch.tensor(
            np.array(target_probabilities), dtype=torch.float32).to(device)

        # 4-Entraînement du PIE
        self.logging("PIE : Entraînement sur cette liste")
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model_pie.parameters(), lr=0.008, momentum=0.9)
        for epoch in range(num_epochs):
            outputs = model_pie(input_data)[:, label]  # Prédiction PIE pour la classe cible
            loss = criterion(outputs, target_probabilities)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        timing = f"--- {time.time() - start_time:.2f} seconds ---"
        self.logging(f"PIE : Fin entraînement {timing}.")

    @staticmethod
    def _image_to_masks_mc(image: npt.NDArray, list_of_mask: list[npt.NDArray], n_mc_sample: int = 50):
        """Crée un échantillonnage Monté-Carlo d'image d'un sous-ensemble de masque.

        :param (np.array) image:    l'image à masquer
        :param list(np.array(bool)) list_of_mask:    liste de masques booléens
        :param (int) n_mc_sample:   nombre d’échantillons Monte-Carlo
        :return:                 tuple (list_images_masked, list_idx_masks, list_fused_masks)
        """
        num_masks = len(list_of_mask)

        return np.random.binomial(1,0.5,size=(n_mc_sample,num_masks))


class Model_EAC:
    """Le modèle EAC.

    ============
    Attributs :
        logger (logging.Logger)  le logger
        device (Torch.Device)    le device utilisé
        model (Torch.nn.Module)  le modèle à expliquer
        model_fc                 pointeur vers la dernière couche du modèle à expliquer
        image                    l'image dont l'explication de la prédiction du modèle est demandé
        sam             le modèle de segmentation sémantique utilisé par EAC
        pie             le modèle de modélisation du modèle à expliquer
        results         les différent résultats

    ============
    Méthodes :
        ...

    ============
    Exemples :
        ...
    """

    SAM_URL: dict[str, str] = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",  # 2.39G
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",  # 1.2G
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",  # 358M
    }
    name = "EAC"

    def __init__(self, model: torch.nn.Module | None = None, image: Any | None = None,
                 sam_args: dict | None = None, pie_args: dict | None = None,
                 log_filepath: str | None = None,
                 device=DEVICE_GLOBAL):
        """Initialise la classe.
        """
        self.results: dict = {}
        self.image: npt.NDArray = None
        self.image_filename: str = self.name
        self.logger: logging.Logger = self._create_logger(log_filepath)
        self.device: torch.device = device
        self.model_fc = None
        self.model: torch.nn.Module = self.load_model_to_explain(model)
        self.class_names: list[str] = torchvision.models.ResNet50_Weights.DEFAULT.meta["categories"]
        self.label_predicted: str = None
        
        # Modèle EAC :
        self.sam: torch.nn.Module = None
        self.pie: torch.nn.Module = None
        if sam_args is not None and sam_args.get("other_model") is not None:
            self.sam = sam_args["other_model"]
        else:
            self.sam = self.load_sam(sam_args)

        if pie_args is not None and pie_args.get("other_model") is not None:
            self.pie = pie_args["other_model"]
        elif pie_args:
            self.pie = self.load_pie(pie_args)

        if self.sam is not None:
            self.sam = self.sam.to(self.device)
        if self.pie is not None:
            self.pie = self.pie.to(self.device)

        if image is not None:
            self.image = self.load_img(image)

    def _create_logger(self, filepath: str | None = None) -> logging.Logger:
        """Création et configuration d'un loggeur.

        :param filepath: chemin de sauvegarde du fichier des logs.
        :return:    le logger ainsi créé
        """
        tmp_txt = ""
        self.file = sys.stdout
        if filepath is not None:
            self.file = filepath

        logger = logging.getLogger(__class__.__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        # console_handler.setLevel("DEBUG")
        logger.addHandler(console_handler)
        if isinstance(self.file, str):
            os.makedirs(os.path.dirname(self.file), exist_ok=True)
            file_handler = logging.FileHandler(self.file, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            tmp_txt = f" et dans {self.file}"

        logger.info(f"Loggeur à stdout{tmp_txt}.")
        return logger

    @staticmethod
    def log_str_format(obj) -> str:
        """Retire l'affichage des tableaux Numpy dans les logs."""
        tmp_obj = obj
        if isinstance(obj, dict):
            tmp_obj = obj.copy()
            for item in tmp_obj:
                if isinstance(tmp_obj[item], np.ndarray):
                    tmp_obj[item] = f"np.ndarray={tmp_obj[item].shape}"
        elif isinstance(obj, list):
            tmp_obj = obj.copy()
            for i, item in enumerate(tmp_obj):
                if isinstance(tmp_obj[i], np.ndarray):
                    tmp_obj[i] = f"np.ndarray={tmp_obj[i].shape}"

        return tmp_obj

    def logging(self, msg: str, level: int | str = logging.INFO,
                caller_name: str = "", **kwargs):
        """Permet l'appel et wrapper sur la classe `logging.Logger`.

        :param (str) msg:   message à logger
        :param (int | str) level: [défaut INFO] niveau de log (cf. logging)
        :param (str) caller_name:   nom du logger à afficher dans les logs
        """
        log_fct = {
            logging.DEBUG: self.logger.debug,
            "debug": self.logger.debug,
            logging.INFO: self.logger.info,
            "info": self.logger.info,
            logging.WARNING: self.logger.warning,
            "warning": self.logger.warning,
            logging.ERROR: self.logger.error,
            "error": self.logger.error,
            logging.CRITICAL: self.logger.critical,
            "critical": self.logger.critical,
        }
        if isinstance(level, str):
            level = level.lower()
        if caller_name == "":
            caller_name = self.__class__.__name__
        msg_text = f"{caller_name} - {self.log_str_format(msg)}"
        log_fct.get(level, logging.INFO)(msg_text, **kwargs)

    def load_model_to_explain(self, model: torch.nn.Module) -> None:
        """Charge le modèle à expliquer avec le device de la cette classe.

        Met le modèle `model` dans self.model et dans le même device de la classe.
        Et met à jour self.model_fc.

        :param (torch.nn.Module):   modèle à expliquer
        :return None:   le modèle sera chargé dans self.model
        """
        self.model = None
        self.model_fc = None
        if model is not None:
            self.logging(f"Chargement du modèle à expliquer : {type(model)} ...")
            self.model = model.to(self.device)
            self.model.eval()
            self.model_fc = self.model.fc

    def load_sam(self, sam_args: dict | None = None) -> torch.nn.Module:
        """Charge SAM soit à partir du disque soit en le téléchargeant.

        Utilise un hack en utilisant une fonction de Torch (`torch.hub.load_state_dict_from_url`)
        pour déléguer la bonne présence des poids sinon de les télécharger. Mais
        comme cela s'occupe uniquement des poids, le modèle n'est pas construit.
        D'où la suppression en mémoire des poids avant de charger SAM correctement,
        afin d'éviter d'avoir les poids 2 fois en mémoire.

        :param (dict) sam_args:     paramètres de chargement de SAM
            sam_args["sam_type"]    le type de model SAM
            sam_args["model_dir"]     le dossier où sont les poids
              si le dossier ou les poids n'existent pas, ils sont téléchargés dedans
        :return (segment_anything.modeling.sam.Sam):    le modèle SAM demandé
        """
        if sam_args is None:
            sam_args = {}
        sam_args["sam_type"] = sam_args.get("sam_type", DEFAULT_SAM_MODEL)
        sam_args["model_dir"] = sam_args.get("model_dir", DEFAULT_MODEL_FOLDER)
        model_type = sam_args["sam_type"]
        model_dir = sam_args["model_dir"]

        # 1-Vérification présence des poids sinon téléchargement
        url = self.SAM_URL.get(model_type, self.SAM_URL[DEFAULT_SAM_MODEL])
        # model = torch.utils.model_zoo.load_url(url, model_dir=model_dir, weights_only=True)
        try:
            model = torch.hub.load_state_dict_from_url(
                url, model_dir=model_dir, weights_only=True)
        except:  # ancienne version de Torch
            model = torch.hub.load_state_dict_from_url(url, model_dir=model_dir)
        del model

        # 2-Création et chargement de SAM en mémoire demandée
        checkpoint_path = os.path.join(model_dir, os.path.basename(url))
        self.logging(f"Chargement SAM de {checkpoint_path}")
        model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        return model

    def load_pie(self, pie_args: dict | None = None) -> Model_PIE:
        """Charge le modèle PIE utilisé dans EAC.

        :param (dict) pie_args:     paramètres de chargement de SAM
            pie_args["model_type"]    le type de model SAM
            pie_args["model_dir"]     le dossier où sont les poids
        :return (Model_PIE):    le modèle PIE demandé
        """
        model = None
        if pie_args is not None:
            self.logging(f"Chargement du PIE avec {pie_args=}")
            model = Model_PIE(pie_args, self.model_fc, logger_master=self)
        return model

    def load_img(self, args: dict | str,
                 transform_PIL=DEFAULT_TRANSFORM_BEGIN) -> npt.NDArray | None:
        """Charge l'image à partir d'un fichier et la prépare.

        :param (dict | str) args: les arguments contenant le chemin ou directement le chemin
            args["input"]   arg ligne de commande pour le chemin de l'image
        :param transform_PIL:    la transformation PIL initiale (Par défaut resize)
        :return (np.array):    l'image préparé
        """
        if args is None or (isinstance(args, dict) and args.get("input") is None):
            self.logging("Image non donnée !", level=logging.WARNING)
            return None
        if isinstance(args, np.ndarray):
            return args

        image_filename = args
        if isinstance(image_filename, dict):
            image_filename = args["input"]

        if not os.path.exists(image_filename):
            self.logging(
                f"Chemin image non trouvé {image_filename}!", level=logging.WARNING)
            return None

        self.logging(f"Chargement image de {image_filename}")
        image_PIL = Image.open(image_filename)
        image_PIL = image_PIL.convert("RGB")
        self.image_filename = image_filename

        if transform_PIL is not None:
            image_PIL = transform_PIL(image_PIL)

        return np.array(image_PIL)

    def run_sam(self, args: dict | None = None, mode="mask"):
        """Exécute SAM.

        :param (dict) args:    paramètres pour exécution de SAM
            args["sam_img_in"]  (np.array | Path) chemin ou image précharger
            args["SamAutomaticMaskGenerator"]   paramètre à passer à SamAutomaticMaskGenerator
                model: Sam,
                points_per_side: Optional[int] = 32,
                points_per_batch: int = 64,
                pred_iou_thresh: float = 0.88,
                stability_score_thresh: float = 0.95,
                stability_score_offset: float = 1.0,
                box_nms_thresh: float = 0.7,
                crop_n_layers: int = 0,
                crop_nms_thresh: float = 0.7,
                crop_overlap_ratio: float = 512 / 1500,
                crop_n_points_downscale_factor: int = 1,
                point_grids: Optional[List[np.ndarray]] = None,
                min_mask_region_area: int = 0,
                output_mode: str = "binary_mask",
        :return None:   Enregistre le résultat de SAM dans self.results["sam"]:
            self.results["sam"] : list(dict(str, any)) A list over records for masks.
             Each record is a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. shape HW
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """
        results: dict[str, Any] = {}

        if args is None:
            args = {}

        # 1- Récupération image
        image_rgb = self.load_img(args.get("sam_img_in"), transform_PIL=None)
        # 2. Prédire avec SAM
        sam_result = None
        if mode == "mask":
            self.logging(
                f"SAM : exécution SAM en mode {mode} pour image={image_rgb.shape} ({image_rgb.dtype}):")
            start_time = time.time()
            sam_args = args.get("SamAutomaticMaskGenerator", {})
            sam_args["model"] = self.sam
            # mask_generator = SamAutomaticMaskGenerator(self.sam)
            mask_generator = SamAutomaticMaskGenerator(**sam_args)
            sam_result = mask_generator.generate(image_rgb)
            timing = f"--- {time.time() - start_time:.2f} seconds ---"
            self.logging(f"SAM : {timing} nb_mask={len(sam_result)}")
        else:
            self.logging(
                f"SAM : Exécution non implémenté {mode}", level=logging.WARNING)

        # Rappel : nous on va vouloir avoir besoin de self.results["sam"][i]["segmentation"]
        self.results["sam"] = sam_result

    def run(self, args: dict | None = None):
        """Exécute la tâche de l'EAC

        :param (dict) args: les arguments de la fonction
        :return: les résultats sont enregistrés dans l'attributs self.results :
            self.results["image_class_name"]
            self.results["shapley_values"]
            self.results["image_masked_xai"]
            self.results["sorted_masks_xai"]
        """
        results: dict[str, Any] = {}
        self.logging(f"EAC : Début exécution.")
        if args is None:
            args = {}

        args["shapley_mc"] = args.get("shapley_mc", DEFAULT_SHAPLEY_MC_SAMPLING)
        args["pie_mc"] = args.get("pie_mc", DEFAULT_PIE_MC_SAMPLING)
        args["pie_epoch"] = args.get("pie_epoch", DEFAULT_PIE_EPOCH)
        args["transform_img"] = args.get("transform_img", DEFAULT_TRANSFORM_IMAGENET)

        # 1- Récupération image
        image_rgb = self.load_img(args.get("sam_img_in"), transform_PIL=None)
        if image_rgb is None:
            return

        # 2-Exécution de SAM
        self.logging(f"EAC : Exécution de SAM sur {image_rgb.shape} ...")
        self.run_sam(args)
        list_of_mask = np.array([i['segmentation'].tolist()
                                for i in self.results["sam"]])

        # 3-Création PIE
        pie_args = {}
        pie_args["dim_in"] = len(list_of_mask)
        pie_args["dim_fc"] = self.model_fc.in_features
        self.pie = self.load_pie(pie_args)

        # 3.5-Prédiction de la classe de l'image
        self.logging("EAC : Prédiction de la classe de l'image ...")
        model = self.model.eval().to(self.device)
        input_tensor = args["transform_img"](image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = model(input_tensor)
        image_class = int(torch.argmax(torch.nn.functional.softmax(pred, dim=1)))
        self.label_predicted = image_class
        self.results["image_class_name"] = self.class_names[image_class]
        self.logging(f"EAC : => classe de l'image={image_class} ({self.class_names[image_class]})")
        
        # 4-Entraînement PIE
        self.logging("EAC : Entraînement du PIE ...")
        self.pie.training_pie(self.model, image_rgb, list_of_mask,
                              transform_img=args["transform_img"],
                              n_samples=args["pie_mc"], num_epochs=args["pie_epoch"], label= image_class)

        # 5-Calcul de la Shapley-values
        self.logging("EAC : Calcul des valeurs de Shapley ...")
        start_time = time.time()
        shapley_values = self.calc_shapley(image_rgb, list_of_mask,
                                           shapley_mc=args["shapley_mc"])

        # 6-Récupérer des concepts explicatifs.
        idx_max = np.argmax(shapley_values)
        mask_max = list_of_mask[idx_max]  
        image_masked_max = image_rgb * mask_max[:, :, np.newaxis]
        sorted_masks = list_of_mask[np.argsort(-shapley_values)]
        timing = f"--- {time.time() - start_time:.2f} seconds ---"
        self.logging(
            f"EAC : => {timing} mask={idx_max}, Shapley={shapley_values[idx_max]}")

        # 7-Calcul AUC
        self.logging("EAC : Calcul des AUC ...")
        auc_deletion = self.calc_auc(
            image_rgb, sorted_masks, transform_img=args["transform_img"], is_deletion=True)
        auc_augmentation = self.calc_auc(
            image_rgb, sorted_masks, transform_img=args["transform_img"], is_deletion=False)
        self.logging(f"EAC : => {auc_deletion=} , {auc_augmentation=}")

        self.results["shapley_values"] = shapley_values
        self.results["image_masked_xai"] = image_masked_max
        self.results["sorted_masks_xai"] = sorted_masks
        self.results["auc_deletion"] = auc_deletion
        self.results["auc_augmentation"] = auc_augmentation

        self.logging(f"EAC : Fin exécution.")

    def calc_shapley(self, image: npt.NDArray, list_of_mask: npt.NDArray[npt.NDArray[bool]],
                     shapley_mc: int = DEFAULT_SHAPLEY_MC_SAMPLING) -> npt.NDArray[float]:
        """Calcule la valeur de Shapley pour tous les concepts.

        :param image:   l'image à expliquer (H, W, C)
        :param list_of_mask:    liste des masques (K, H, W)
        :param shapley_mc:      valeur de Monte-Carlo sampling
        :param label:    la classe de l'image à expliquer (si None, on utilise la prédiction du modèle)
        :return:                les valeurs de Shapley pour chaque masque
        """
        nb_mask = len(list_of_mask)
        shapley_values = np.zeros(nb_mask, dtype=float)

        # 2-Calcul des valeurs de Shapley pour chaque masque
        for i in tqdm(range(nb_mask), desc="Calcul Shapley"):
            # 2.1-échantillonnage par Monte-Carlo de masques
            batch_mask = self.pie._image_to_masks_mc(
                image, list_of_mask, n_mc_sample=shapley_mc)
            batch_mask_false = batch_mask.copy()
            # 2.2-calcul de la proba du PIE avec ou sans le concept sur la liste des sous-masques
            model_pie = self.pie.eval().to(self.device)
            with torch.no_grad():
                batch_mask_false[:,i] = 0
                batch_mask[:,i] =1 
                batch_mask = torch.tensor(np.array(batch_mask), dtype=torch.float32).to(self.device)
                batch_mask_false = torch.tensor(
                    np.array(batch_mask_false), dtype=torch.float32).to(self.device)
                outs = model_pie(batch_mask)
                outs_false = model_pie(batch_mask_false)
                probas = torch.nn.functional.softmax(outs, dim=1)[:,self.label_predicted] 
                probas_false = torch.nn.functional.softmax(outs_false, dim=1)[:,self.label_predicted] 

            # 2.3-Calcul de Shapley moyenne
            shapley_values[i] = (probas -probas_false).mean().item()

        return shapley_values

    def calc_auc(self, image: npt.NDArray, sorted_masks: npt.NDArray[npt.NDArray[bool]],
                 transform_img=DEFAULT_TRANSFORM_IMAGENET, is_deletion: bool = True) -> float:
        """Calcul les AUC montant et descendant pour le modèle EAC.

        :param image:           l'image à expliquer
        :param sorted_masks:    l'ensemble des masques issue de SAM ordonnée par Shapley
        :param transform_img:   la transformation à appliquer avant la prédiction par le modèle
        :param is_deletion:     pour calculer l'AUC en mode délition ou augmmetation.
        :return:    l'auc du mode souhaité
        """
        model = self.model.eval().to(self.device)
        # 1- Initialisation avec image entière si mode délition, image noire sinon
        current_mask = np.zeros(image.shape[:2], dtype=bool) if is_deletion else np.ones(image.shape[:2], dtype=bool)
        x, imgs_masked = [], []
        next_img = image * (~current_mask)[:, :, np.newaxis]
        imgs_masked.append(next_img)
        x.append(100 * np.sum(current_mask) / current_mask.size)
        for i in range(len(sorted_masks)):
            if is_deletion:
                current_mask = np.logical_or(current_mask, sorted_masks[i])
            else:
                current_mask = np.logical_and(current_mask, np.logical_not(sorted_masks[i]))
            next_img = image * (~current_mask)[:, :, np.newaxis]
            imgs_masked.append(next_img)
            x.append(100 * np.sum(current_mask) / current_mask.size)
        imgs_masked = np.stack(imgs_masked)
        with torch.no_grad():
            # 3-Prédiction de la classe de l'image
            input_tensor = transform_img(image).unsqueeze(0).to(self.device)
            pred = model(input_tensor)
            proba = torch.nn.functional.softmax(pred, dim=1)
            image_class = int(torch.argmax(proba))
            image_class_proba = float(torch.max(proba))
            image_class_name = self.class_names[image_class]  # str nom de la classe

            # 4-Prédiction de chaque images masquées
            input_tensor = torch.stack([transform_img(img) for img in imgs_masked]).to(self.device)
            outs = torch.nn.functional.softmax(model(input_tensor), dim=1)
            pred_masks = outs[:, image_class].cpu().numpy()

        # 5-Préparation pour l'auc
        pred_masks[pred_masks >= image_class_proba] = image_class_proba
        y = pred_masks / image_class_proba

        # 6-Calcul de l'AUC
        auc_value = auc(x, y)

        return auc_value

    def save(self, mode: Literal["all", "model", "results"] = "all",
             args: dict | None = None):
        """Sauvegarde le modèle entraîné ou les résultats.

        :param (str) mode:  spécifie quoi enregistrer ["all", "model", "results"]
        :param (dict) args: les arguments d'enregistrement
            args["save_filepath"]
            args["save_is_print"]
        """
        if mode in ("all", "model"):
            self.save_model(args)
        if mode in ("all", "results"):
            self.save_results(args)
        return

    def save_model(self, args: dict | None = None):
        """Sauvegarde le modèle.

        :param (dict) args: les arguments pour la sauvegarde du modèle
            args["model_save_filename"]
            args["model_save_folder"]
        """
        if args is None:
            args = {}
        filename = args.get("model_save_filename", DEFAULT_SAVE_MODEL_FILENAME)
        foldername = args.get("model_save_folder", DEFAULT_SAVE_FOLDER)
        return

    def save_results(self, args: dict | None = None):
        """Sauvegarde les résultats.

        :param (dict) args: les arguments pour la sauvegarde du modèle
            args["results_save_filename"]   préfixe du nom de fichier "SAM" par défaut
            args["results_save_folder"]
        """
        if args is None:
            args = {}

        filename_img = args.get("results_save_filename", self.image_filename)
        foldername = args.get("results_save_folder", DEFAULT_SAVE_FOLDER)
        # args["results_save_filename"] = filename_img
        args["results_save_folder"] = foldername
        os.makedirs(foldername, exist_ok=True)
        img_mask_filename = f"{filename_img}_mask.jpg"
        img_bg_filename = "EAC_values.csv"

        img_to_save = []
        if self.results.get("image_masked_xai") is not None:
            img_mask_PIL = Image.fromarray(self.results["image_masked_xai"])
            img_to_save.append((img_mask_PIL, img_mask_filename))

        # sauvegarde des images
        for img, filename in img_to_save:
            filepath = os.path.join(foldername, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.logging(f"Sauvegarde des résultats : {filepath}")
            img.save(filepath)

        # sauvegarde de l'auc
        if self.results.get("auc_deletion") is not None and self.results.get("auc_augmentation") is not None:
            filepath = os.path.join(foldername, img_bg_filename)
            first = not os.path.exists(filepath)
            with open(filepath, "a") as f:
                self.logging(f"Sauvegarde des AUC dans le : {filepath}")
                if first:
                    f.write("img;auc_deletion;auc_augmentation\n")
                auc_deletion = self.results.get("auc_deletion")
                auc_augmentation = self.results.get("auc_augmentation")
                f.write(f"{filename_img};{auc_deletion};{auc_augmentation}\n")


###############################################################################
# FONCTIONS PROJET :
def run_process(args: dict | None = None) -> Model_EAC:
    """Exécute le déroulé du programme.

    :param (dict) args:    les paramètres fournis
    """
    ###
    # Gestion des arguments par défaut - compatible mode console ou mode Python
    ###
    if args is None:
        args = {}

    # 3.1 args génériques
    # task [--task TASK] = (str) "run", "train", ...
    args["task"] = args.get("task", "run")
    # nolog [--nolog] = (bool)
    args["nolog"] = args.get("nolog", False)
    # logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
    args["logfile"] = args.get("logfile", sys.stdout)
    # savefile [--savefile [FILENAME]] = défaut stdout, None si pas FILENAME
    args["savefile"] = args.get("savefile", sys.stdout)
    # log_filepath = (str) None si pas de log en fichier
    filename = args["logfile"] if args["logfile"] is not None else DEFAULT_LOG_FILENAME
    timestamp = str(int(time.time()))
    foldername = os.path.join(DEFAULT_LOG_FOLDER, timestamp)
    if args["nolog"] or not isinstance(filename, str):
        args["log_filepath"] = None
    else:
        folder = args.get("log_folder", foldername)
        args["log_filepath"] = os.path.join(folder, filename)
    # log_is_print = (bool) local pour afficher les log
    args["log_is_print"] = args.get("log_is_print", not args["nolog"])
    # device [--device DEVICE] = (str) 'auto, cpu, cuda, torch_directml'
    # args["device"] = args.get("device", utils.torch_pick_device())
    args["device"] = args.get("device", DEVICE_GLOBAL)
    # output [--output [FOLDER]] = (str) défaut 'results'
    args["output"] = args.get("output", DEFAULT_SAVE_FOLDER)
    # checkpoint [--checkpoint FOLDER] = (str)
    args["checkpoint"] = args.get("checkpoint", DEFAULT_MODEL_FOLDER)

    is_saving = False
    foldername = os.path.join(DEFAULT_SAVE_FOLDER, timestamp)
    args["results_save_folder"] = args.get("results_save_folder", foldername)
    args["model_save_folder"] = args.get("model_save_folder", DEFAULT_SAVE_FOLDER)
    args["save_filepath"] = None
    if args["savefile"] is None or isinstance(args["savefile"], str):  # SAVE :
        is_saving = True
        if isinstance(args["savefile"], str) and args.get("results_save_filename") is None:
            args["results_save_filename"] = args["savefile"]
            args["model_save_filename"] = args["savefile"]

    # Gestion de l'image :
    args["sam_img_in"] = args.get("sam_img_in")
    if args.get("input") is None and args.get("sam_img_in") is None:
        print("SAM a besoin d'image or aucune image n'a été fournie !")
        return

    # sam_type [--model_type VIT_TYPE] = (str) "vit_h", "vit_l", "vit_b"
    args["sam_type"] = args.get("sam_type", DEFAULT_SAM_MODEL)
    # model [--model VIT_TYPE] = (str)
    args["model"] = args.get("model", DEFAULT_MODEL_TEST)
    # shapley_mc [--shapley_mc] SHAPLEY_MC = (int)
    args["shapley_mc"] = args.get("shapley_mc", DEFAULT_SHAPLEY_MC_SAMPLING)
    # pie_mc [--pie_mc] PIE_MC = (int)
    args["pie_mc"] = args.get("pie_mc", DEFAULT_PIE_MC_SAMPLING)
    # pie_epoch [--pie_epoch] PIE_EPOCH = (int)
    args["pie_epoch"] = args.get("pie_epoch", DEFAULT_PIE_EPOCH)

    ###
    # Gestion du flux d'exécution
    ###
    pprint(Model_EAC.log_str_format(args))

    # 1- Chargement du modèle EAC
    sam_args = {}
    sam_args["sam_type"] = args["sam_type"]
    sam_args["model_dir"] = args["checkpoint"]
    model_xai = Model_EAC(sam_args=sam_args, log_filepath=args["log_filepath"])

    # 2- Chargement des images à tester
    list_img_test = []
    if (args["task"] == "run" and args["sam_img_in"] is None) or (args["task"] == "test" and os.path.isfile(args.get("input",""))):
        list_img_test.append(args.get("input"))

    if args["task"] == "test" and os.path.isdir(args.get("input","")):
        types = ('*.png', '*.jpg', '*.jpeg',  '*.JPEG')
        for files in types:
            list_img_test.extend(glob.glob(os.path.join(args.get("input"), files)))

    model_xai.logging(f"Chargement de {len(list_img_test)} images.", caller_name="run_process")

    # 3- Suivant la tâche exécution de celle-ci
    modes: dict[str, str] = {"run": "results", "train": "model", "test": "results"}
    mode: str = modes.get(args["task"], "results")

    f_model = torchvision.models.get_model(args["model"])
    model_xai.load_model_to_explain(f_model)
    for img in list_img_test:
        args["sam_img_in"] = model_xai.load_img(
            img, transform_PIL=DEFAULT_TRANSFORM_BEGIN)
        model_xai.run(args)

        # 3-Sauvegarde
        if is_saving:
            model_xai.logging("Action : sauvegarde ...", caller_name="run_process")
            model_xai.save(mode, args)

    return model_xai


###############################################################################
# FONCTIONS MODE CONSOLE :
def parse_args(args_str: str | None = None) -> argparse.Namespace:
    """Gestion des arguments à modifier en fonction.

    ====
    Arguments :
        logfile [--logfile [FILENAME]] = défaut stdout, None si pas FILENAME
        nolog [--nolog] = (bool)

    :param (str) args_str: pour simuler les arguments données au programme
    :return (argparse.Namespace):   les arguments parsés
    """
    if args_str is not None:
        args_str = shlex.split(args_str)

    # 1 - Définition des listes de choix :
    list_task_agentIA = ["run", "test"]
    list_model_SAM = ["vit_h", "vit_l", "vit_b"]

    # 2 - Création du parseur à arguments:
    parser = argparse.ArgumentParser(prog="EAC",
                                     description="Lance la XAI avec le EAC.",
                                     epilog="Exemples : python xai_samshap.py --input=dog.jpeg --sam_type vit_b --model=resnet50 --device=cuda ")

    # 3 - Définition des arguments :
    parser.add_argument("--task", type=str, choices=list_task_agentIA, default="run",
                        help="[défaut=run] tâche à accomplir")
    parser.add_argument("--logfile", nargs='?', type=argparse.FileType("w"),
                        default=sys.stdout, help="sortie du programme")
    parser.add_argument("--savefile", nargs='?', type=argparse.FileType("w"),
                        default=sys.stdout,
                        help="spécifie s'il faut sauvegarder.")
    parser.add_argument("--nolog", action='store_true',
                        help="désactive les log")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_MODEL_FOLDER,
                        help=f"[défaut={DEFAULT_MODEL_FOLDER}] dossier où sont les poids des modèles.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="[défaut=cpu] device où charger le modèle [auto, cpu, cuda, torch_directml]")
    parser.add_argument("--sam_type", type=str, choices=list_model_SAM, default=DEFAULT_SAM_MODEL,
                        help=f"[défaut={DEFAULT_SAM_MODEL}] modèle VIT de SAM [vit_h, vit_l, vit_b]")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_TEST,
                        help=f"[Défaut={DEFAULT_MODEL_TEST}] modèle à tester (de torchvision)")
    parser.add_argument("--output", type=str, nargs='?', default=DEFAULT_SAVE_FOLDER,
                        help=f"[défaut={DEFAULT_SAVE_FOLDER}] chemin du dossier de sortie")
    parser.add_argument("--input", type=str,
                        help="chemin de l'image d'entrée (ou dossier de test en mode test)")

    parser.add_argument("--shapley_mc", type=int, default=DEFAULT_SHAPLEY_MC_SAMPLING,
                        help=f"[défaut={DEFAULT_SHAPLEY_MC_SAMPLING}] échantillonnage Monté-Carlo pour les valeurs de Shapley.")
    parser.add_argument("--pie_mc", type=int, default=DEFAULT_PIE_MC_SAMPLING,
                        help=f"[défaut={DEFAULT_PIE_MC_SAMPLING}] échantillonnage Monté-Carlo pour l'entraînement PIE.")
    parser.add_argument("--pie_epoch", type=int, default=DEFAULT_PIE_EPOCH,
                        help=f"[défaut={DEFAULT_PIE_EPOCH}] epoch pour l'entraînement PIE.")

    # 4 - Parser les arguments
    args = parser.parse_args(args_str)

    return args


def main(args: argparse.Namespace) -> int:
    """Exécute le flux d'exécution des tâches pour l'agent.

    :param args:    les paramètres fournis en ligne de commande
    :return (int):  le code retour du programme
    """
    exit_value: int = EXIT_OK
    # Gestion particulière des args externes si besoin
    print(args)

    # Exécution
    start_time = time.time()
    eac = run_process(vars(args))
    print(f'\n--- {time.time() - start_time} seconds ---')

    return exit_value


###############################################################################
if __name__ == "__main__":
    print(sys.argv)
    sys.exit(main(parse_args()))
# end if
