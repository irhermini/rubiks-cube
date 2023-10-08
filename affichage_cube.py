import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.animation as animation



def init_faces():
    mat_squares = [None] * 54
    square0 = np.array([[0, 3, 3], [0, 2, 3], [1, 2, 3], [1, 3, 3]])
    square9 = np.array([[0, 0, 3], [1, 0, 3], [1, 0, 2], [0, 0, 2]])
    square18 = np.array([[0, 3, 3], [0, 2, 3], [0, 2, 2], [0, 3, 2]])
    square27 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    square36 = np.array([[3, 3, 3], [2, 3, 3], [2, 3, 2], [3, 3, 2]])
    square45 = np.array([[3, 0, 3], [3, 1, 3], [3, 1, 2], [3, 0, 2]])
    
    for j in range(0, 3):
        for k in range(0, 3):
            mat_squares[3 * j + k] = square0 + k * np.array([[1, 0, 0]] * 4) - j * np.array([[0, 1, 0]] * 4)
            mat_squares[9 + 3 * j + k] = square9 + k * np.array([[1, 0, 0]] * 4) - j * np.array([[0, 0, 1]] * 4)
            mat_squares[18 + 3 * j + k] = square18 - k * np.array([[0, 1, 0]] * 4) - j * np.array([[0, 0, 1]] * 4)
            mat_squares[27 + 3 * j + k] = square27 + k * np.array([[1, 0, 0]] * 4) + j * np.array([[0, 1, 0]] * 4)
            mat_squares[36 + 3 * j + k] = square36 - k * np.array([[1, 0, 0]] * 4) - j * np.array([[0, 0, 1]] * 4)
            mat_squares[45 + 3 * j + k] = square45 + k * np.array([[0, 1, 0]] * 4) - j * np.array([[0, 0, 1]] * 4)
    squares = [Poly3DCollection([square]) for square in mat_squares]
    
    return squares

def affichage_cube(cube, etat):
    fig = plt.figure()
    
    ax = a3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_zlim(-1, 4)
    squares = init_faces()
    for j in range(len(squares)):
            #print(squares[i])
        squares[j].set_facecolor(cube.Couleurs[int(etat[j])])
        squares[j].set_edgecolor('Black')
        ax.add_collection3d(squares[j])
    
    return fig
   

def affichage_resolution(cube, etatinitial, actions):

    fig = plt.figure()


    etats = [etatinitial]
    etat = etatinitial
    
    for action in actions:
        etat = cube.doAction(action, etat)
        etats.append(etat)
    
    ax = a3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.set_zlim(-1, 4)
    
 #   squares = init_faces()
 #   for j in range(len(squares)):
            #print(squares[i])
 #       squares[j].set_facecolor(cube.Couleurs[int(etatinitial[j])])
 #       squares[j].set_edgecolor('Black')
 #       ax.add_collection3d(squares[j])
        
    def animate(state):
        ax.clear()
   #     global ax
   #     ax = a3.Axes3D(fig, auto_add_to_figure=False)
   #     fig.add_axes(ax)
   #     ax.set_xlabel('X')
   #     ax.set_ylabel('Y')
   #     ax.set_zlabel('Z')
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.set_zlim(-1, 4)
        squares = init_faces()
        for j in range(len(squares)):
            #print(squares[i])
            squares[j].set_facecolor(cube.Couleurs[int(state[j])])
            squares[j].set_edgecolor('Black')
            ax.add_collection3d(squares[j])
        return ax,

        
    ani = animation.FuncAnimation(fig, animate, frames = etats, interval = 400, blit = False, repeat=False)
  #  state = etatinitial
  #  ax = a3.Axes3D(fig, auto_add_to_figure=False)
  #  fig.add_axes(ax)
 #   ax.set_xlabel('X')
 #   ax.set_ylabel('Y')
 #   ax.set_zlabel('Z')
 #   ax.set_xlim(-1, 4)
 #   ax.set_ylim(-1, 4)
 #   ax.set_zlim(-1, 4)
#    for i in range(len(squares)):
            #print(squares[i])
 #       squares[i].set_facecolor(cube.Couleurs[int(state[i])])
 #       squares[i].set_edgecolor('Black')
 #       ax.add_collection3d(squares[i])
#    ani.save('animation.mp4', writer='ffmpeg')
    return ani
    plt.show()
    

