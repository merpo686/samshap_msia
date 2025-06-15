"""BGDIA7065 Projet XAI, samshap (EAC).

Sujet : https://proceedings.neurips.cc/paper_files/paper/2023/file/44cdeb5ab7da31d9b5cd88fd44e3da84-Paper-Conference.pdf

Implémentation du modèle EAC (Explain Any Concept).
Partie pour effectuer les tests humain.
"""
__author__ = ['Nicolas Allègre', 'Louis Borreill', 'Merlin Poitou']
__date__ = '11/06/2025'
__version__ = '0.2'

###############################################################################
# IMPORTS :
# /* Modules standards */
import os
import random

# /* Modules externes */
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from PIL import Image, ImageTk

###############################################################################
# ———————— CONFIG ————————
DIR_ORIGINAL = 'data/imagenet_subsample/'
DIR_EAC      = 'data/imagenet-explain-eac/'
# DIR_EAC      = 'data/imagenet-explain-ours/'
DIR_LIME     = 'data/imagenet-explain-lime/'
CSV          = 'data/auc_scores.csv'
IMG_COUNT    = 100  # nombre max d’itérations


###############################################################################
# ———————— FONCTIONS ————————
def load_images_from(folder):
    # charge et trie la liste des fichiers image d’un dossier
    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
    ])


def show_image(label, folder, filename, image_size):
    path = os.path.join(folder, filename)
    img = Image.open(path)
    orig_w, orig_h = img.size
    max_w, max_h = image_size
    ratio = min(max_w/orig_w, max_h/orig_h)
    new_size = (int(orig_w*ratio), int(orig_h*ratio))
    resized = img.resize(new_size, resample=Image.LANCZOS)
    tkimg = ImageTk.PhotoImage(resized)
    label.image = tkimg
    label.config(image=tkimg)


def update_score_table():
    lines = [f"{names[i]} : {votes[i]} votes" for i in range(len(names))]
    lines.append(f"None of them : {none_votes} votes")
    score_label.config(text="\n".join(lines))


def next_image():
    global current_index
    # si on n'a pas atteint la fin
    if current_index + 1 < IMG_COUNT:
        current_index += 1
        update_images()
    else:
        # désactiver tous les boutons
        for btn in buttons:
            btn.config(state="disabled")
        btn_none.config(state="disabled")
        messagebox.showinfo("Terminé", "Toutes les images ont été votées.")
    update_images()


def update_images():
    global current_perm
    # afficher image originale en haut
    key = os.path.splitext(orig_files[current_index])[0]
    # show_image(label_top, DIR_ORIGINAL, orig_map[key], (top_max_w, top_max_h))
    show_image(label_top, DIR_ORIGINAL, orig_map[key], image_size)
    # texte de la classe
    label_phrase.config(text=f"Which image best describes: {df.loc[key, 'classe']}")
    # permutation aléatoire
    current_perm = random.sample(range(len(image_sets)), k=len(image_sets))
    # afficher chaque méthode sur chaque label
    for pos, method_idx in enumerate(current_perm):
        folder   = folders[method_idx]
        filename = image_sets[method_idx][current_index]
        # show_image(labels[pos], folder, filename, (bot_max_w, bot_max_h))
        show_image(labels[pos], folder, filename, image_size)
    update_score_table()


def vote(pos):
    method_idx = current_perm[pos]
    votes[method_idx] += 1
    update_score_table()
    next_image()

def vote_none():
    global none_votes
    none_votes += 1
    update_score_table()
    next_image()


###############################################################################
# listes d'images tronquées
orig_files = load_images_from(DIR_ORIGINAL)[:IMG_COUNT]
eac_files  = load_images_from(DIR_EAC)[:IMG_COUNT]
lime_files = load_images_from(DIR_LIME)[:IMG_COUNT]

# mapper nom sans extension -> nom complet
orig_map = {os.path.splitext(f)[0]: f for f in orig_files}

df = pd.read_csv(CSV)
# on suppose df['image'] contient les noms sans extension
# on filtre sur les clés
df = df[df['image'].isin(orig_map.keys())]
# index sur le nom sans extension
df = df.set_index('image')

# regrouper méthodes pour échelle M
folders    = [DIR_EAC, DIR_LIME]
image_sets = [eac_files, lime_files]
names      = ['EAC', 'LIME']

# compteurs de votes pour chaque méthode
votes = [0] * len(image_sets)
none_votes = 0

# drapeau de permutation courant
current_perm = []
current_index = 0

# ———————— INIT TK ————————
root = tk.Tk()
root.title("Comparateur Méthodes")
root.state('zoomed')  # plein écran

# récupérer taille écran
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.geometry(f"{screen_w}x{screen_h}")

WIDTH = screen_w // 4
HEIGHT = screen_h // 4
image_size = (HEIGHT, WIDTH)

# calculer dimensions dynamiques des images
top_max_w = int(screen_w * 0.8)
top_max_h = int(screen_h * 0.4)
bot_max_w = int(screen_w * 0.45)
bot_max_h = int(screen_h * 0.3)

# ———————— WIDGETS & LAYOUT ————————
# widgets
label_top    = tk.Label(root)
label_phrase = tk.Label(root, text="", font=("Courier", 24))
frame_bottom = tk.Frame(root)
labels       = []
buttons      = []
score_label  = tk.Label(root, font=("Courier", 24), justify='left')

# layout pack pour le haut
label_top.pack(pady=10)
label_phrase.pack(pady=5)

# layout bottom
frame_bottom.pack(pady=5, fill='x')
for pos, name in enumerate(names):
    frame = tk.Frame(frame_bottom)
    frame.pack(side='left', expand=True, padx=20)
    lbl = tk.Label(frame)
    lbl.pack()
    btn = tk.Button(frame, text="Vote", font=("Courier", 20), command=lambda p=pos: vote(p))
    btn.pack(pady=5)
    labels.append(lbl)
    buttons.append(btn)

# bouton None of them centré en dessous des votes
frame_none = tk.Frame(root)
frame_none.pack(pady=10)
btn_none = tk.Button(frame_none, text="None of them", font=("Courier", 20), command=vote_none)
btn_none.pack()

# score label en bas
score_label.pack(pady=10)

# lancement
update_images()
root.mainloop()
