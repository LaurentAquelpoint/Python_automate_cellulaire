import tkinter as tk
import numpy as np
import pandas as pd

couleur_carres = '#ee502d'

fen1 = tk.Tk()

fr1 = tk.Frame(fen1, height = 1000, width = 1000)
fr1.grid(row = 0, column = 0)

fr2 = tk.Frame(fen1, height = 1000, width = 1000)
fr2.grid(row = 0, column = 1)

# probabilité de quitter un emplacement : poids = poids matrice depart + nbr voisins * -0.125 + 2 ou nbr voisins * 0.125 + 1

class cellule():
    def __init__(self, nom, position, liste_voisins, pos_rel_vois, rectangle):
        self.nom = nom
        self.position = position # position dans la matrice n + 2
        self.liste_voisins = liste_voisins
        self.pos_rel_vois = pos_rel_vois
        self.vois_connexe = True
        self.nbr_vois = 0
        self.rectangle = rectangle
        
##########

class class_glob():
    def __init__(self):
        
        self.n = 25 # nombre de cellules
        self.taille = 10  # taille d'un petit carré
        self.nb_cel = 5 # nombre de cellules vivantes
        
        self.limite_cel = 3
        self.can = tk.Canvas(fr1, height = self.taille*self.n, width = self.taille * self.n, bg = '#f2d38a')
        self.can.grid(row = 1, column = 1)
        
        self.abscisse = 0
        self.labx = tk.Label(fr1, textvariable = "testx")
        self.labx.grid(row = 1, column = 0)
        self.ordonnee = 0

        
        self.MA = 0
        self.type_con = 8 # 4 ou 8
        self.p = 10
        self.k = 0
                 
        self.temps = 20
        self.liste_culture = []
        self.nourriture = np.zeros((self.n + 2, self.n + 2))
#        for i in range(self.n):
#            for j in range(self.n):
#                if((i/self.n > 0.5) & (i/self.n < 0.9) & (j/self.n > 0.5) & (i/self.n < 0.9)):
#                    self.nourriture[i, j] = 2
        
tot = class_glob()                

##########
               
class culture_cel():
    def __init__(self, name, col, x, y, nb_cel):
        self.nb_cel = nb_cel
        self.col = col
        self.dico_cellules = {}
        
        self.matrice_cellules = np.zeros(shape = (tot.n + 2, tot.n + 2)) # matrice donnant la position de toutes les cellules, de taille n + 2
        self.matrice_depart = np.ones(shape = (tot.n + 2, tot.n + 2)) # matrice des poids de départ
        self.matrice_arrivee = np.ones(shape = (tot.n + 2, tot.n + 2)) # matrice des poids d'arrivée
        self.x = x # position des cellules de départ
        self.y = y
        self.name = name
        self.liste_place_libre = []

        for i in range(self.nb_cel):
             for j in range(self.nb_cel):
        
                self.matrice_cellules[int(self.x * tot.n - self.nb_cel/2 + i + 1), int(self.y * tot.n - self.nb_cel/2 + j + 1)] = 1
                nom = self.name + str(i) + str(j)
                self.dico_cellules[nom] = cellule(nom,
                                  (int(self.x * tot.n - self.nb_cel/2 + i) + 1, int(self.y * tot.n - self.nb_cel/2 + j) + 1),
                                  [],
                                  [],
                                  rectangle = tot.can.create_rectangle(int(self.y * tot.n - self.nb_cel/2 + j) * tot.taille + 2,
                                             int(self.x * tot.n - self.nb_cel/2 + i) * tot.taille + 2,
                                             int(self.y * tot.n - self.nb_cel/2 + j + 1) * tot.taille + 2,
                                             int(self.x * tot.n - self.nb_cel/2 + i + 1) * tot.taille + 2,
                                             fill = self.col,
                                             width = 0)) # attention x et y sont inversés
         
#                print(type(self.dico_cellules[nom].rectangle))
   
tot.liste_culture =  {'c' : culture_cel('c', '#ee502d', 0.5, 0.5, tot.nb_cel)}
                                        
#                                        'd' : culture_cel('d', 'red', 0.6, 0.6, 4),
#                                        'e' : culture_cel('e', 'blue', 0.4, 0.6, 4),
#                                        'f' : culture_cel('f', 'green', 0.6, 0.4, 4)
#                                        }

#print(tot.liste_culture["c"].matrice_cellules)
                                        
##########

#print(len(cult1.dico_cellules))
#cult1.dico_cellules = {}
#cult1.dico_cellules['c00'] = cellule('c00',
#                                  (10, 10),
#                                  ['c01'],
#                                  [],
#                                  0)
#
#cult1.dico_cellules['c01'] = cellule('c01',
#                                  (11, 10),
#                                  ['c00', 'c02'],
#                                  [],
#                                  0)
#
#cult1.dico_cellules['c02'] = cellule('c02',
#                                  (12, 10),
#                                  ['c01', 'c03'],
#                                  [],
#                                  0)
#
#cult1.dico_cellules['c03'] = cellule('c03',
#                                  (13, 10),
#                                  ['c02', 'c04'],
#                                  [],
#                                  0)
#
#cult1.dico_cellules['c04'] = cellule('c04',
#                                  (14, 10),
#                                  ['c03'],
#                                  [],
#                                  0)

                                        
##########

def ajoute_pos(t1, t2):
    x1 = t1[0]
    x2 = t2[0]
    y1 = t1[1]
    y2 = t2[1]
    return (int(x1 - x2), int(y1 - y2))

##########

def translate_pos(pos1, pos2):
    return (pos1[0] + pos2[0], pos1[1] + pos2[1])

##########
    
def dist_l1(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

##########

def ajoute_proba(mat, pos, p):
    x = pos[0]
    y = pos[1]

    for i in range(-1, 2):
        for j in range(-1, 2):
            if((x + i <= tot.n + 1) & (y + j <= tot.n + 1)):
                mat[int(x + i), int(y + j)] *= p

##########
            
def dist8(tupl1, tupl2):
    return max(abs(tupl1[0] - tupl2[0]), abs(tupl1[1] - tupl2[1]))
    
##########

def creer_liste_voisins():
    for name in tot.liste_culture:
        culture = tot.liste_culture[name]
        
        for key1 in culture.dico_cellules:
            for key2 in culture.dico_cellules:
                
                diff_x = culture.dico_cellules[key1].position[0] - culture.dico_cellules[key2].position[0]
                diff_y = culture.dico_cellules[key1].position[1] - culture.dico_cellules[key2].position[1]
                
    
                
                if(max(abs(diff_x), abs(diff_y)) == 1):                
                    
                    culture.dico_cellules[key1].liste_voisins.append(key2)
                    culture.dico_cellules[key1].pos_rel_vois.append((diff_x, diff_y))

                culture.dico_cellules[key1].nbr_vois = len(culture.dico_cellules[key1].liste_voisins)

creer_liste_voisins()
        
##########

def test_connexite_vois(nom): # test de connexité par voisins
                            # prend une cellule et renvoie True si ses voisins sont connexes, False sinon                  
    pl = nom[0]
    cellule = tot.liste_culture[pl].dico_cellules[nom]
    
    if(len(cellule.pos_rel_vois) in [1, 7, 8]):
        return True
    elif(tot.type_con == 8):
        
        liste_dist = []
        nbr_vois = len(cellule.pos_rel_vois)
        # on calcule toutes les distances possibles entre deux points de la cellule
        # les voisins sont connexes ssi le nombre de fois où la distance est égale à 1 est 2n - 2 où n est le nbr de voisins
        
        for num1 in cellule.pos_rel_vois:
            for num2 in cellule.pos_rel_vois :
                if(dist8(num1, num2) == 1):
                    liste_dist.append(1)
        
        compt = len(liste_dist)

        if(all([elem in cellule.pos_rel_vois for elem in [(0, -1), (-1, -1), (-1, 0)]])):
           compt = compt - 2
        if(all([elem in cellule.pos_rel_vois for elem in [(-1, 0), (-1, 1), (0, 1)]])):
           compt = compt - 2
        if(all([elem in cellule.pos_rel_vois for elem in [(0, 1), (1, 1), (1, 0)]])):
           compt = compt - 2
        if(all([elem in cellule.pos_rel_vois for elem in [(1, 0), (1, -1), (0, -1)]])):
           compt = compt - 2
 
        return (compt >= 2*nbr_vois - 2)

    elif(tot.type_con == 4):
        
        liste_dist = []
        nbr_vois = len(cellule.pos_rel_vois)
        # on calcule toutes les distances possibles entre deux points de la cellule
        # les voisins sont connexes ssi le nombre de fois où la distance est égale à 1 est 2n - 2 où n est le nbr de voisins
        
        for num1 in cellule.pos_rel_vois:
            for num2 in cellule.pos_rel_vois :
                if(dist_l1(num1, num2) == 1):
                    liste_dist.append(1)
        
        compt = len(liste_dist)

          
        return (compt == 2*nbr_vois - 2)

##########

def test_conn_tot(nom): # test de connexité total
                        # vérifie si en retirant la cellule nom, la culture est toujours connexe
    premiere_lettre = nom[0]
    if(nom == (premiere_lettre + '00')):
        nom_dep = premiere_lettre + '01'
    else:
        nom_dep = premiere_lettre + '00'
    
    culture = tot.liste_culture[premiere_lettre]
    
    liste_cel = [nom_dep]
    liste_temp = [nom_dep]
    
    
    while(liste_temp != []):

        liste_temp2 = list(liste_temp)
        liste_temp = []

        for name in liste_temp2:
            cel = culture.dico_cellules[name]
 
            for vois in cel.liste_voisins:

                if(tot.type_con == 4):
                    if((vois not in liste_cel) & (vois != nom) & (dist_l1(cel.position, culture.dico_cellules[vois].position) == 1)):
                        liste_cel.append(vois)
                        liste_temp.append(vois)
                        
                elif(tot.type_con == 8):
                    if((vois not in liste_cel) & (vois != nom)):
                        liste_cel.append(vois)
                        liste_temp.append(vois)
                        
#    print(len(liste_cel))              
    return(len(liste_cel) == len(culture.dico_cellules) - 1)
    
##########

def select_cellule(culture): # plus une cellule a de voisins, plus elle bougera 
    
    liste_cellules_bougeables = []
    liste_nbr_vois = []
    liste_poids_sup = []
    
    for name in culture.dico_cellules:
        cel = culture.dico_cellules[name]
        
        if(test_connexite_vois(cel.nom)):                                                                         # creer une liste avec les cellules ayant des voisins connexes
            liste_cellules_bougeables.append(name)
            liste_nbr_vois.append(len(cel.liste_voisins))
            liste_poids_sup.append(culture.matrice_depart[int(cel.position[0]), int(cel.position[1])])
    
    if(len(liste_cellules_bougeables) < tot.limite_cel):
        for name in culture.dico_cellules:
            cel = culture.dico_cellules[name]
            
            if(test_conn_tot(name)):
                liste_cellules_bougeables.append(name)
                liste_nbr_vois.append(len(cel.liste_voisins))
                liste_poids_sup.append(culture.matrice_depart[int(cel.position[0]), int(cel.position[1])])
            
    liste_poids = list(np.array([65 - w**2 for w in liste_nbr_vois]) * np.array(liste_nbr_vois)) # le poids d'une cellule qui decroit
                                                                                # avec le nbr de voisins + le poids de la matrice_depart
#nbr voisins * -0.125 + 1

    df = pd.DataFrame({'cellule': liste_cellules_bougeables,
                   'poids': liste_poids})
    

    nom_cel = df.sample(n = 1, weights = 'poids')
    nom = nom_cel.iloc[0, 0]

    return nom

########## Non utilisé
    
def creer_liste_cel_libre(culture):
    for i in range(0, tot.n + 2):
        for j in range(0, tot.n + 2):
            if(test_libre((i, j), culture)):
                culture.liste_place_libre.append((i, j))

#for cult in tot.liste_culture:
#    culture = tot.liste_culture[cult]
#    creer_liste_cel_libre(culture)
    
########## Non utilisé
    
def test_libre(pos, culture): # teste si un emplacement est vide et a au moins un voisin
    
    x = pos[0]
    y = pos[1]
    if((x == 0) or (x == tot.n + 1) or (y == 0) or (y == tot.n + 1)):
        return True
    if(tot.type_con == 8):
        liste_tot = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    elif(tot.type_con == 4):
        liste_tot = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    n_vois = 0
    
    for position in liste_tot:
        n_vois += (culture.matrice_cellules[x - position[0], y - position[1]])
    
    return((culture.matrice_cellules[x, y] == 0) & (n_vois > 0))

##########  

def suppr_cellule(cellule):
    
    pl = cellule.nom[0]
    culture = tot.liste_culture[pl]
    
    for vois in cellule.liste_voisins: # maj les infos des voisins
        voisin = culture.dico_cellules[vois]
        
        voisin.liste_voisins.remove(cellule.nom)
        voisin.pos_rel_vois.remove(ajoute_pos(voisin.position, cellule.position))
    
    
    cellule.liste_voisins = []
    cellule.pos_rel_vois = []
    
    x = int(cellule.position[0])
    y = int(cellule.position[1])
    
    culture.matrice_cellules[x, y] = 0
    tot.can.delete(cellule.rectangle)
    
    pos = (cellule.position[0], cellule.position[1])

    ajoute_proba(culture.matrice_depart, pos, tot.p)
    ajoute_proba(culture.matrice_arrivee, pos, 1/tot.p)

##########

def ajoute_cellule(pos, cellule):
    cellule.position = pos
    pl = cellule.nom[0]
    culture = tot.liste_culture[pl]
    
    x = pos[0]
    y = pos[1]
    
    culture.matrice_cellules[int(x), int(y)] = 1
    
    for key in culture.dico_cellules: # maj les infos des voisins
        
        cel = culture.dico_cellules[key]
        pos2 = cel.position
        
        if(dist8(pos, pos2) == 1):
            cel.liste_voisins.append(cellule.nom)
            
            x2 = pos2[0]
            y2 = pos2[1]
            
            cel.pos_rel_vois.append((x2 - x, y2 - y))
            
            cellule.liste_voisins.append(key)
            cellule.pos_rel_vois.append((x - x2, y - y2))

            
    cellule.rectangle = tot.can.create_rectangle((y-1)*tot.taille + 2,
                                                 (x-1)*tot.taille + 2,
                                                 (y)*tot.taille + 2,
                                                 (x)*tot.taille + 2,
                                                 fill = culture.col, width = 0)     # attention x et y sont inversés pour être cohérent avec les abscisses et ordonnées
          

    ajoute_proba(culture.matrice_depart, pos, 1/tot.p)
    ajoute_proba(culture.matrice_arrivee, pos, tot.p)

########## 
    
def supprime_doublons(liste): # une fonction qui supprime tous les doublons d'une liste
    nouvelle_liste = []
    for i in liste:
        if(i not in nouvelle_liste):
            nouvelle_liste.append(i)
            
    return nouvelle_liste
    
##########
    
def cellules_libres(nom): # calcule toutes les cellules libres adjacentes à une cellule
    
    if(tot.type_con == 8):
        liste_tot = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    elif(tot.type_con == 4):
        liste_tot = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
    pl = nom[0]
    culture = tot.liste_culture[pl]
    culture.liste_place_libre = []
    
    for key in culture.dico_cellules:
        
        if key != nom:
       
            cellule = culture.dico_cellules[key]
            pos_temp = []
            for pos in liste_tot:
                
                if(pos not in cellule.pos_rel_vois):
                    aj = ajoute_pos(cellule.position, pos)
                    
                    if((aj[0] != 0) &
                      (aj[1] != 0) &
                      (aj[0] != tot.n - 1) &
                      (aj[1] != tot.n - 1)):
                       pos_temp.append(aj)
                    
                    pos_temp.append(aj)
                
            culture.liste_place_libre = culture.liste_place_libre + pos_temp
    culture.liste_place_libre = [pos for pos in culture.liste_place_libre if pos not in [(0, 0), (0, tot.n + 1), ( tot.n + 1, 0), ( tot.n + 1,  tot.n + 1)]] # empêche les coins d'être sélectionnés comme libres
    culture.liste_place_libre = supprime_doublons(culture.liste_place_libre) # supprime les doublons

##########

def compte_vois(pos, cult):
    
    x = pos[0]
    y = pos[1]
    
    n_vois = 0
    
    for i in range(-1, 2):
        for j in range(-1, 2):
            if(((i != 0) or (j != 0)) & ((int(x + i) <= tot.n + 1) & (int(y + j) <= tot.n + 1))):
                if(cult.matrice_cellules[int(x + i), int(y + j)] == 1):
                    n_vois += 1
    return n_vois

##########
    
def fonc_mat(x):
    
    p = (tot.p)**0.25
    
    if(x > 1):
        x =  x / p
    elif(x < 0):
        x = x * p
    return x

##########

def maj_matrices():
    for cult in tot.liste_culture:
        culture = tot.liste_culture[cult]
        for i in range(tot.n + 2):
            for j in range(tot.n + 2):
                culture.matrice_depart[i, j] = fonc_mat(culture.matrice_depart[i, j])
                culture.matrice_arrivee[i, j] = fonc_mat(culture.matrice_arrivee[i, j])
                
    culture.matrice_depart = np.maximum(culture.matrice_depart, tot.nourriture)
    culture.matrice_arrivee = np.maximum(culture.matrice_arrivee, tot.nourriture)
    
##########

def maj_cellules():
    
    maj_matrices()

    for key in tot.liste_culture:
        
        culture = tot.liste_culture[key]
        
        nom = select_cellule(culture)           # choisi une cellule déplaçable au hasard
        cellule = culture.dico_cellules[nom]
        position_aj = cellule.position
        
        cellules_libres(nom)
        
        suppr_cellule(cellule)

        liste_poids = []
        liste_poids_sup = []

        for place in culture.liste_place_libre:

            liste_poids.append(compte_vois(place, culture))                                         # pour chaque emplacement libre, ajoute le poids correspondant
            liste_poids_sup.append(culture.matrice_arrivee[int(place[0]), int(place[1])])
                
        liste_poids = list(np.array([w**2 for w in liste_poids]) * np.array(liste_poids_sup)) # poids pour les cellules arrivantes
        # plus une place a de voisins plus elle est susceptible de se faire prendre
        
        df = pd.DataFrame({'place_libre': culture.liste_place_libre,
                       'poids': liste_poids})
        
        nouv_pos = df.sample(n = 1, weights = 'poids')
        
        pos = nouv_pos.iloc[0, 0]
        
        y = pos[0]                              # abscisse de la place libre sélectionnée
        x = pos[1]                              # ordonnée de la place libre sélectionnée
        
        culture.liste_place_libre.append(position_aj)
        ajoute_cellule(pos, cellule)
        
        if(x <= 0):
            "gauche"
            reinitialise_rect(1, 0)
            tot.abscisse = tot.abscisse - 1
            text_x.set(str(tot.abscisse))

        elif(x >= tot.n + 1):
            "droite"
            reinitialise_rect(-1, 0)
            tot.abscisse = tot.abscisse + 1
            text_x.set(str(tot.abscisse))

        elif(y <= 0):
            "haut"
            reinitialise_rect(0, 1)
            tot.ordonnee = tot.ordonnee + 1
            text_y.set(str(tot.ordonnee))

        elif(y >= tot.n + 1):
            "bas"
            reinitialise_rect(0, -1)
            tot.ordonnee = tot.ordonnee - 1
            text_y.set(str(tot.ordonnee))
    
    tot.k = tot.k + 1
    texte_compte.set(str(tot.k))
#    print(culture.matrice_cellules)
    tot.can.after(tot.temps, marche_auto)
    
##########    

def reinitialise_rect(x, y): # x et y coordonnées de la translation à opérer
        
    for cult in tot.liste_culture:
        cult2 = tot.liste_culture[cult]
        new_liste_place_libre = []
        
        for place in cult2.liste_place_libre:
            new_liste_place_libre .append(translate_pos(place, (y, x)))
        cult2.liste_place_libre = new_liste_place_libre
        
        for cell in cult2.dico_cellules:
            cell2 = cult2.dico_cellules[cell]
            tot.can.delete(cell2.rectangle)
            pos = cell2.position
            cell2.position = (pos[0] + y, pos[1] + x)                           # change les positions des rectangles
            
            pos2 = cell2.position
            cell2.rectangle = tot.can.create_rectangle((pos2[1] - 1)*tot.taille + 2,
                                                 (pos2[0] - 1)*tot.taille + 2,
                                                 pos2[1]*tot.taille + 2,
                                                 pos2[0]*tot.taille + 2,
                                                 fill = cult2.col, width = 0)
            
        cult2.matrice_depart = translate_coef_mat(1, cult2.matrice_depart, x, y)
        cult2.matrice_arrivee = translate_coef_mat(1, cult2.matrice_arrivee, x, y)
        cult2.matrice_cellules = translate_coef_mat(0, cult2.matrice_cellules, x, y)

##########

def translate_coef_mat(val, matrice, x, y):

    dim = matrice.shape[0]
    
    if x > 0:
        "gauche"
        matrice = matrice[:, 0:(dim - 1)] 
        matrice = np.column_stack((val * np.ones(shape = (dim, 1)), matrice))
        
    elif x < 0:
        "droite"
        matrice = matrice[:, 1:dim]
        matrice = np.column_stack((matrice, val * np.ones(shape = (dim, 1))))

    elif y > 0:
        "haut"
        matrice = matrice[0:(dim - 1), :]
        matrice = np.row_stack((val * np.ones(shape = (1, dim)), matrice))
        
    elif y < 0:
        "bas"
        matrice = matrice[1:dim, :]
        matrice = np.row_stack((matrice, val * np.ones(shape = (1, dim))))
        
    return(matrice)
                     
a = np.arange(16).reshape((4, 4))

#print(a)
#print(translate_coef_mat(0, a, -1, 0))

##########

def marche_arret():
    if tot.MA == 1:
        tot.MA = 0
        
    else :
        tot.MA = 1
        marche_auto()

##########      

def marche_auto():
    
    if(tot.MA == 1):
        tot.can.after(tot.temps, maj_cellules)
  
##########

boutonfont = ('times', '20')

Bnext = tk.Button(fr2, text = 'Next', width = 10, height = 4, command = maj_cellules)
Bnext.grid(row = 0, column = 1, sticky = 'nw')
Bnext.config(font = boutonfont)

Bauto = tk.Button(fr2, text = 'Marche auto', width = 10, height = 4, command = marche_arret)
Bauto.grid(row = 1, column = 1, sticky = 'nw')
Bauto.config(font = boutonfont)

texte_compte = tk.StringVar()
texte_compte.set(str(tot.k))

Label_compt = tk.Label(fr2, textvariable = texte_compte, width = 10, height = 4)
Label_compt.grid(row = 2, column = 1, sticky = 'nw', columnspan = 2)
Label_compt.config(font = boutonfont)

text_x = tk.StringVar()
text_x.set(str(tot.abscisse))

labx = tk.Label(fr1, textvariable = text_x)
labx.grid(row = 0, column = 1)
labx.config(font = boutonfont)

text_y = tk.StringVar()
text_y.set(str(tot.ordonnee))

laby = tk.Label(fr1, textvariable = text_y)
laby.grid(row = 1, column = 0)
laby.config(font = boutonfont)

fen1.mainloop()

#A = np.matrix(([1, 2], [4, 5]))
#print(A)
#
#B = A
#
#B = B + 1
#print(B)

#
#
#a = tk.Tk()
#
#canvas = tk.Canvas(a, width = 500, height = 500, bg = 'blue')
#canvas.pack()
#
#myrect = canvas.create_rectangle(0,0,100,100)
##canvas.delete(myrect) #Deletes the rectangle
#
#
#a.mainloop()

#for key in cult1.dico_cellules:
#    print(cult1.dico_cellules[key].position)

u = np.array([-1, 5, -4, 5, 7, 6])
np.maximum(u, 1)

#for i in tot.liste_culture['c'].dico_cellules:
#    print(tot.liste_culture['c'].dico_cellules[i].vois_connexe)






















