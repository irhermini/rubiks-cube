from AEtoileNoeud import *
from Cube import Cube3
import torch
from Model import CubeNet
import time
from affichage_cube import *
import streamlit as st
import streamlit.components.v1 as components

#composition du cube


st.set_page_config(page_title = "Rubiks Cube", page_icon = ":tada", layout = "wide")
fonctionh = CubeNet(4)
cube = Cube3()
path_network = "save_models/numEtats 250 numEpochs 7000 lr 0.00100 batch 100 time 16092023-005352.pt"
fonctionh.load_state_dict(torch.load(path_network))
fonctionh.eval()
app = torch.device("cuda" if torch.cuda.is_available() else "cpu")





if 'generate' not in st.session_state:
    st.session_state.generate = False 
    st.session_state.etatinitial = None


if st.button("Réinitialiser"):
    st.session_state.generate = False
    st.session_state.etatinitial = None

    
if st.button("Générer un Rubiks Cube"):
    etat_initial = cube.generateState(100)
    figure_etat = affichage_cube(cube, etat_initial)
    st.pyplot(figure_etat)
    st.session_state.generate = True
    st.session_state.etatinitial = etat_initial
if st.button("Résoudre"):
    try: 
        etat_initial = st.session_state.etatinitial
        print(etat_initial[0])
        
    except AttributeError and TypeError:
        st.write("Générez un cube d'abord")
    else:
        etat_initial = st.session_state.etatinitial
        figure_etat = affichage_cube(cube, etat_initial)
        st.pyplot(figure_etat)
        st.write('Début de la résolution ...')
        time.sleep(5)
        st.write('Résolution en cours, veuillez patienter')
        mvts , numNoeuds, Itr, Resolu, DureeRecherche = AEtoileRech(etat_initial, cube, 0.02, fonctionh, app, 4000, 32)

        if Resolu:
            st.write("Résolu ! ")
            st.write("Durée de résolution : %i minutes %i secondes"%(DureeRecherche // 60, DureeRecherche % 60))
            st.write("Nombre d'itérations : %i"%(Itr))
            st.write("Nombre de mouvements : %i"%(len(mvts)))

            ani = affichage_resolution(cube, etat_initial, mvts)
#         ani.save("animation.mp4")
            st.title("Animation de la résolution : ")
            components.html(ani.to_jshtml(), height=1000)
            st.write("Résolution :")
            for mvt in mvts:
                st.write(cube.Actions[mvt][1])


        else: 
            st.write("Désolé, nous n'avons pas pu résoudre votre énigme ! ")
        
