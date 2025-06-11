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
__author__ = ['Nicolas Allègre', 'Louis Borreill', 'Merlin Poitou', 'Julian Sliva']
__date__ = '17/05/2025'
__version__ = '0.1'

###############################################################################
# IMPORTS :
# /* Modules standards */
import argparse
import logging
import os
import shlex
import sys
import time
from typing import Any, Final, Literal

# /* Modules externes */
import cv2
import numpy as np
import torch
from PIL import Image
from pprint import pprint

# /* Module interne */
IA_AGENT_SAM_FOLDER = "SAM"
sys.path.append(IA_AGENT_SAM_FOLDER)
from SAM.segment_anything import sam_model_registry, SamAutomaticMaskGenerator

###############################################################################
# CONSTANTES :
EXIT_OK: Final[int] = 0

DEFAULT_SAVE_FOLDER: Final[str] = "results"
DEFAULT_LOG_FOLDER: Final[str] = "logs"
DEFAULT_LOG_FILENAME: Final[str] = "pipeline.log"
DEFAULT_MODEL_FOLDER: Final[str] = "checkpoints"
DEFAULT_SAM_MODEL: Final[str] = "vit_h"

DEVICE_GLOBAL: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###############################################################################
# CLASSES PROJET :
class Model_PIE(torch.nn.Module):
    """Le modèle PIE d'EAC.

    ============
    Attributs :
        

    ============
    Méthodes :
        ...
    """

    def __init__(self, pie_args: dict | None = None, model_fc=None):
        """Initialise le PIE.

        :param (dict) pie_args:     argument de configuration du PIE
        :param model_fc:            la dernière couche du modèle à expliquer.
        """
        super().__init__()
        if pie_args is None:
            pie_args = {}
        dim_in = pie_args.get("dim_in") 
        dim_fc = pie_args.get("dim_fc")
        if dim_in is None or dim_fc is None:
            return None

        self.linear = torch.nn.Sequential(nn.Linear(dim_in, dim_fc))
        self.model_fc = model_fc
    
    def forward(self, x, with_fc: bool = True):
        out = self.linear(x)
        if self.model_fc != None and with_fc:
            out = self.model_fc(out)
        return out

    def train(self, x, y, optimizer_fct, optimizer_opt, loss_fct, loss_opt, device):
        model = self.train()
        optimizer = optimizer_fct(**optimizer_opt)
        loss = loss_fct(**loss_opt)
        
        optimizer.zero_grad()
        output = loss(x, y)
        output.backward()
        optimizer.step()
        return output.detach().cpu().numpy()

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

    def __init__(self, model: torch.nn.Module | None = None,
                 sam_args: dict | None = None, pie_args: dict | None = None,
                 device=DEVICE_GLOBAL):
        """Initialise la classe.
        """
        self.results: dict = {}
        self.image = None
        self.logger: logging.Logger = self._create_logger()
        self.device: torch.device = device
        self.model_fc = None
        self.model: torch.nn.Module = self.load_model_to_explain(model)

        # Modèle EAC :
        self.sam: torch.nn.Module = None
        self.pie: torch.nn.Module = None
        if sam_args is not None and sam_args.get("other_model") is not None:
            self.sam = sam_args["other_model"]
        else:
            self.sam = self.load_sam(sam_args).to(self.device)

        if pie_args is not None and pie_args.get("other_model") is not None:
            self.pie = pie_args["other_model"]
        else:
            self.pie = self.load_pie(pie_args).to(self.device)

    def _create_logger(self, filepath: str | None = None) -> logging.Logger:
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

    def logging(self, msg: str, level: int | str = logging.INFO, caller_name: str = "", **kwargs):
        """Permet l'appel et wrapper sur la classe `logging.Logger`.

        :param (str) msg:   message à logger
        :param (int | str): [défaut INFO] niveau de log (cf. logging)
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
            sam_args["model_type"]    le type de model SAM
            sam_args["model_dir"]     le dossier où sont les poids
              si le dossier ou les poids n'existent pas, ils sont téléchargés dedans
        :return (segment_anything.modeling.sam.Sam):    le modèle SAM demandé
        """
        if sam_args is None:
            sam_args = {}
        sam_args["model_type"] = sam_args.get("model_type", DEFAULT_SAM_MODEL)
        sam_args["model_dir"] = sam_args.get("model_dir", DEFAULT_MODEL_FOLDER)
        model_type = sam_args["model_type"]
        model_dir = sam_args["model_dir"]

        # 1-Vérification présence des poids sinon téléchargement
        url = self.SAM_URL.get(model_type, self.SAM_URL[DEFAULT_SAM_MODEL])
        # model = torch.utils.model_zoo.load_url(url, model_dir=model_dir, weights_only=True)
        try:
            model = torch.hub.load_state_dict_from_url(url, model_dir=model_dir, weights_only=True)
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
        self.logging(f"Chargement du PIE avec {pie_args=}")
        model = Model_PIE(pie_args, self.model_fc)

        return model

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
        image_rgb = args.get("sam_img_in")
        if image_rgb is None:
            self.logging("image non donnée !", level=logging.WARNING)
            return
        if isinstance(image_rgb, str):
            image_filename = image_rgb
            if not os.path.exists(image_filename):
                self.logging(f"chemin image non trouvée {image_filename}!", level=logging.WARNING)
                return
            self.logging(f"Chargement img {image_filename}")
            image_PIL = Image.open(image_rgb)
            image_PIL = image_PIL.convert("RGB")
            image_rgb = np.array(image_PIL)

        # 2. Prédire avec SAM
        sam_result = None
        if mode == "mask":
            sam_args = args.get("SamAutomaticMaskGenerator", {})
            sam_args["model"] = self.sam
            # mask_generator = SamAutomaticMaskGenerator(self.sam)
            mask_generator = SamAutomaticMaskGenerator(**sam_args)
            sam_result = mask_generator.generate(image_rgb)
        else:
            self.logging(f"Exécution non implémenté {mode}", level=logging.WARNING)

        # Rappel : nous on va vouloir avoir besoin de self.results["sam"][i]["segmentation"]
        self.results["sam"] = sam_result

    def run(self, args: dict | None = None):
        """Exécute la tâche de l'EAC

        :param (dict) args: les arguments de la fonction
        """
        results: dict[str, Any] = {}

        if args is None:
            args = {}
        is_print = args.get("print", False)
        if is_print:
            print("Exécution de ", self.name)


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
    filename = args["logfile"] if args["logfile"] is not None else f"{DEFAULT_LOG_FILENAME}.log"
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
    args["results_save_folder"] = args.get("results_save_folder", DEFAULT_SAVE_FOLDER)
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
    if args["sam_img_in"] is None:
        image_filename = args["input"]
        if not os.path.exists(image_filename):
            print(f"Chemin image non trouvée {image_filename}!")
            return
        print(f"Chargement img {image_filename}")
        image_PIL = Image.open(image_filename)
        image_PIL = image_PIL.convert("RGB")
        args["sam_img_in"] = np.array(image_PIL)

    # model_type [--model-type VIT_TYPE] = (str) "vit_h", "vit_l", "vit_b"
    args["model_type"] = args.get("model_type", DEFAULT_SAM_MODEL)

    ###
    # Gestion du flux d'exécution
    ###
    pprint(Model_EAC.log_str_format(args))

    # 1- Chargement du modèle EAC
    sam_args = {}
    sam_args["model_type"] = args["model_type"]
    sam_args["model_dir"] = args["checkpoint"]
    pie_args = {}
    model_xai = Model_EAC(sam_args=sam_args, pie_args=pie_args)

    # 2- Suivant la tâche exécution de celle-ci

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
    list_task_agentIA = ["run"]
    list_model_SAM = ["vit_h", "vit_l", "vit_b"]

    # 2 - Création du parseur à arguments:
    parser = argparse.ArgumentParser(prog="EAC",
                                     description="Lance la XAI avec le EAC.",
                                     epilog="Exemples : ... ")

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
                        help="[défaut=checkpoints] dossier où sont les poids des modèles.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="[défaut=cpu] device où charger le modèle [auto, cpu, cuda, torch_directml]")
    parser.add_argument("--model_type", type=str, choices=list_model_SAM, default=DEFAULT_SAM_MODEL,
                        help="[défaut=vit_h] modèle VIT de SAM [vit_h, vit_l, vit_b]")
    parser.add_argument("--output", type=str, nargs='?', default=DEFAULT_SAVE_FOLDER,
                        help="[défaut=results] chemin du dossier de sortie")
    parser.add_argument("--input", type=str,
                        help="chemin de l'image d'entrée")

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
