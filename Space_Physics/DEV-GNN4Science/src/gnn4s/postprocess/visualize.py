import os
import h5py
import math
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib import animation
from matplotlib import tri as mtri
from scipy.interpolate import griddata
from PIL import Image

import gnn4s

def plot_history(history, label, name, dirt):
    """
    Plot a graph along with the corresponding stl (if available)

    Arguments:
        history_train: list of train metrics. First value: epochs. Second value:
                        metric
        history_test: list of test metrics. First value: epochs. Second value:
                        metric
        label (string): name of the metric to use for y-axis and plot title
        folder: output folder. Default -> None

    """
    fig = plt.figure(figsize=(8,4))
    ax = plt.gca()
    ax.set_aspect('auto')
    ax.plot(history[0], history[1], linewidth=3, label=label)
    ax.legend()
    ax.set_xlim((history[0][0],history[0][-1]))

    ax.set_xlabel('epoch')
    ax.set_ylabel(label)
    plt.tight_layout()
    plt.legend(frameon=False)

    if dirt!=None:
        os.makedirs(dirt, exist_ok=True)
        plt.savefig(f'{dirt}/{name}.eps')
    else:
        plt.show()

def save_figure(save_dir: str, title: str='figure', dpi: int=300) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Adjust the position and size of the sub-graph to display as much content as possible in the graph
    plt.tight_layout()
    # Adjust the position of the subgraph so that the distance between the top and the top of the canvas is 0.85
    plt.subplots_adjust(top=0.85)
    # Cut out the blank area around the graphic, set the graphic to be opaque
    plt.savefig(os.path.join(save_dir, title), bbox_inches='tight',
                dpi=dpi, transparent=False)
    plt.close()

def plot_graph_list(graphs, title='title', max_num=16, save_dir=None):
    max_num = min(len(graphs), max_num)
    col_num = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure()

    options = {'node_size':2, 'edge_color':'black', 'linewidths':1, 'width':0.5}

    for i in range(max_num):
        # If the graph is not a NetworkX graph, then copy its underlying graph structure.
        if not isinstance(graphs[i], nx.Graph):
            g = graphs[i].g.copy()
        else:
            g = graphs[i].copy()
        assert isinstance(g, nx.Graph)

        # Delete the isolated nodes in the graph
        g.remove_nodes_from(list(nx.isolates(g)))

        # Calculate the number of edges, nodes, and self-loops of the graph
        e = g.number_of_edges()
        v = g.number_of_nodes()
        l = nx.number_of_selfloops(g)

        # Use the network layout algorithm (spring_layout) to determine the location of nodes
        pos = nx.spring_layout(g)
        
        ax = plt.subplot(col_num, col_num, i + 1)
        nx.draw(g, pos, with_labels=False, **options)
        title_str = f'e={e-l}, n={v}'
        ax.title.set_text(title_str)
    figure.suptitle(title)

    save_figure(save_dir=save_dir, title=title)

def plot_smiles_graph(smiles, save_dir, title):
    graphs = []
    for k in range(len(smiles)):
        # Create an RDKit molecule object from the Smiles string
        mol = Chem.MolFromSmiles(smiles[k])

        # Create a new NetworkX graph object
        graph = nx.Graph()

        # Add a node for each atom in the graph, with the node index being the atom index 
        # and the node attribute being the atom symbol
        for atom in mol.GetAtoms():
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol()
            graph.add_node(atom_idx, symbol=atom_symbol)

        # Add a edge from the starting atom to the ending atom in the graph, 
        # with the weight of the edge being the key type
        for bond in mol.GetBonds():
            start_atom_idx = bond.GetBeginAtom().GetIdx()
            end_atom_idx = bond.GetEndAtom().GetIdx()
            bond_type = bond.GetBondTypeAsDouble()
            graph.add_edge(start_atom_idx, end_atom_idx, weight=bond_type)

        graphs.append(graph)

    plot_graph_list(graphs, save_dir=save_dir, title=title)

def plot_smiles_rdkit(smiles, save_path=None, max_num=16):
    max_num = min(len(smiles), max_num)
    col_num = int(math.ceil(np.sqrt(max_num)))

    # Convert smiles string to molecular structure
    mols = []
    for i in range(max_num):
        mols.append(Chem.MolFromSmiles(smiles[i]))

    # Convert molecular to figure
    image = Draw.MolsToGridImage(mols, molsPerRow=col_num, subImgSize=(200, 150), legends=smiles)

    # save figure
    image.save(save_path)

def draw_graph_list(graph_list, save_path, row: int=4, col: int=4,
                    iterations: int=100, layout: str='spring', is_single: bool=False, k=1,
                    nodes_num: int=55, alpha: float=1.0, width: float=1.3, remove_inodes: bool=True):

    # 
    graph_list = [nx.to_networkx_graph(graph_list[i]) for i in range(len(graph_list))]

    # remove_inodes isolate nodes in graphs
    if remove_inodes:
        for gg in graph_list:
            gg.remove_nodes_from(list(nx.isolates(gg)))
    
    plt.switch_backend('agg')
    for i, G in enumerate(graph_list):
        plt.subplot(row, col, i+1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        # plt.axis("off")

        # turn off axis label
        plt.xticks([])
        plt.yticks([])

        if layout == 'spring':
            pos = nx.spring_layout(G, k=k / np.sqrt(G.number_of_nodes()), iterations=iterations)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            raise ValueError(f'{layout} not recognized.')

        if is_single:
            nx.draw_networkx_nodes(G, pos, node_size=nodes_num, node_color='#336699', alpha=1, linewidths=0)
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=1.5, node_color='#336699', alpha=1, linewidths=0.2)
            nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def visualize_graphs(graph_list, dir_path, row: int=4, col: int=4, remove_inodes: bool=True):
    fig_graph_num = row * col

    fig_num = int(np.ceil(len(graph_list) / fig_graph_num))

    def init():
        save_path = os.path.join(dir_path, "sample"+str(0)+".png")
        draw_graph_list(graph_list[0*fig_graph_num:1*fig_graph_num], save_path=save_path, row=row, col=col,
                        remove_inodes=remove_inodes)
    
    def draw(i):
        save_path = os.path.join(dir_path, "sample"+str(i)+".png")
        draw_graph_list(graph_list[i*fig_graph_num:(i+1)*fig_graph_num], save_path=save_path, row=row, col=col,
                        remove_inodes=remove_inodes)
    
    fig = plt.figure(figsize=(8, 6))
    ani = animation.FuncAnimation(fig=fig, frames=fig_num, func=draw, init_func=init, interval=500, blit=False)
    ani.save('animation.gif', writer='pillow')
    '''
    init()
    for i in range(1, fig_num):
        draw(i)
    '''
def plot_anime_3d(data, pred, name: str, dirt: str,
                  skip: int=10, intervel: int=100, ratio: float=1.0, dpi: int=100) -> None:
    fig = plt.figure(figsize=(3, 6))
    
    def animate(num):
        x, y = data, pred[:,:,num]
        ax = fig.add_subplot(111, projection='3d')
        ax._axis3don = False
        ax.scatter(x[:,0], x[:,1], x[:,2], c=y[:,1], cmap='coolwarm', depthshade=0, s=5)
        ax.set_box_aspect((np.ptp(x[:,0]), np.ptp(x[:,1]), np.ptp(x[:,2])))
        #if stl_mesh != None:
        #    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors, alpha=0.08))
        ax.set_title(name)

        plt.subplots_adjust(left=-0.4, right=1.4)
        return fig
    
    frame_num = 50 # pred.shape[-1]
    ani = animation.FuncAnimation(fig, animate, frames=frame_num, interval=intervel)
    
    os.makedirs(dirt, exist_ok=True)
    ani.save(f'{dirt}/{name}.gif', writer='pillow', dpi=dpi)

def plot_anime_trimesh2d(data, pred, name: str, dirt: str,
                         skip: int=10, intervel: int=100, dpi: int=100) -> None:
    """ plot predict value on trimesh and create animation.
    Args:
        data: data
        pred: predict value
        name: name of the image
        dirt: directory for storing the image
        skip: number of skipping time step
        intervel: time interval between every two frames
        dpi: resolution of image
    """
    step_num = len(data)
    frame_num = step_num // skip

    vmin = data[0].y.min()
    vmax = data[0].y.max()
    for i in range(1,step_num):
        vmin = min(vmin, data[i].y.min())
        vmax = max(vmax, data[i].y.max())
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    def animate(num):
        ax.cla()
        ax.set_aspect('equal')
        ax.set_axis_off()

        step = num * skip
        xx = data[step].mesh_pos.cpu().detach().numpy()
        yy = pred[step].cpu().detach().numpy()
        idx = data[step].face_index.T.cpu().detach().numpy()
        tri = mtri.Triangulation(xx[:,0], xx[:,1], idx)
        
        ax.tripcolor(tri, yy[:,0], vmin=vmin, vmax=vmax)
        ax.triplot(tri, 'ko-', ms=0.5, lw=0.3)
        ax.set_title('step %d' % (step))
        return fig
    
    ani = animation.FuncAnimation(fig, animate, frames=frame_num, interval=intervel)
    
    os.makedirs(dirt, exist_ok=True)
    ani.save(f'{dirt}/{name}.gif', writer='pillow', dpi=dpi)

def plot_anime_trimesh3d(data, pred, name: str, dirt: str,
                         skip: int=10, intervel: int=100, ratio: float=1.0, dpi: int=100) -> None:
    """ plot predict value on trimesh and create animation.
    Args:
        data: data
        pred: predict value
        name: name of the image
        dirt: directory for storing the image
        skip: number of skipping time step
        intervel: time interval between every two frames
        dpi: resolution of image
    """
    step_num = len(data)
    frame_num = step_num // skip

    vmin = data[0].y.min(0).values
    vmax = data[0].y.max(0).values
    dim = 3
    for i in range(1,step_num):
        for j in range(dim):
            vmin[j] = min(vmin[j], data[i].y[:,j].min())
            vmax[j] = max(vmax[j], data[i].y[:,j].max())
    vmin, vmax = vmin+(0.5-0.5*ratio)*(vmax-vmin), vmin+(0.5+0.5*ratio)*(vmax-vmin)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def animate(num):
        ax.cla()
        ax.set_xlim([vmin[0], vmax[0]])
        ax.set_ylim([vmin[1], vmax[1]])
        ax.set_zlim([vmin[2], vmax[2]])
        ax.set_axis_off()

        step = num * skip
        xx = data[step].mesh_pos.cpu().detach().numpy()
        yy = pred[step].cpu().detach().numpy()
        idx = data[step].face_index.T.cpu().detach().numpy()
        
        ax.plot_trisurf(yy[:,0], yy[:,1], idx, yy[:,2], shade=True)
        ax.set_title('step %d' % (step))
        return fig
    
    ani = animation.FuncAnimation(fig, animate, frames=frame_num, interval=intervel)
    
    os.makedirs(dirt, exist_ok=True)
    ani.save(f'{dirt}/{name}.gif', writer='pillow', dpi=dpi)

def plot_all_nodes(features, graph, statistics, time, dirt, name, framerate=60):
    """
    Creates and saves a .mp4 video with pressure anf flow rate values for 
    all nodes in the graph.

    Arguments:
        features: 3D array with reconstructed features
        graph: the GNN graph
        params: dictionary of parameters
        time: duration of the video in seconds
        outfile_name (string): name of the output video
        framerate (int): framerate. Default -> 60

    """
    nframes = time * framerate

    indices = np.floor(np.linspace(0,features.shape[2]-1,nframes)).astype(int)

    sel_pred_features = features[:,:,indices]
    sel_real_features = graph.ndata['nfeatures'][:,:,indices]

    sel_pred_features[:,0,:] = gnn4s.data.invert_normalize(
        sel_pred_features[:,0,:], 'pressure', statistics, 'features')
    sel_pred_features[:,1,:] = gnn4s.data.invert_normalize(
        sel_pred_features[:,1,:], 'flowrate', statistics, 'features')
    sel_real_features[:,0,:] = gnn4s.data.invert_normalize(
        sel_real_features[:,0,:], 'pressure', statistics, 'features')
    sel_real_features[:,1,:] = gnn4s.data.invert_normalize(
        sel_real_features[:,1,:], 'flowrate', statistics, 'features')
    minp = torch.min(sel_real_features[:,0,:])
    maxp = torch.max(sel_real_features[:,0,:])
    minq = torch.min(sel_real_features[:,1,:])
    maxq = torch.max(sel_real_features[:,1,:])

    fig, ax = plt.subplots(2, dpi = 284)

    nodes = np.arange(features.shape[0])
    scatter_real_p = ax[0].scatter(
        nodes, sel_real_features[:,0,0], color='black', s=1.5, alpha=0.3)
    scatter_pred_p = ax[0].scatter(
        nodes, sel_pred_features[:,0,0], color='red', s=1.5, alpha=1)
    scatter_real_q = ax[1].scatter(
        nodes, sel_real_features[:,1,0], color='black', s=1.5, alpha=0.3)
    scatter_pred_q = ax[1].scatter(
        nodes, sel_pred_features[:,1,0], color='red', s=1.5, alpha=1)
    nodesidxs = np.expand_dims(nodes, axis=1)
    ax[1].set_xlabel('graph node index')
    ax[0].set_ylabel('pressure [mmHg]')
    ax[1].set_ylabel('flowrate [cm^3/s]')

    def animation_frame(i):
        p = sel_real_features[:,0,i]
        p = np.concatenate((nodesidxs, np.expand_dims(p, axis = 1)),axis = 1)
        scatter_real_p.set_offsets(p)
        p = sel_pred_features[:,0,i]
        p = np.concatenate((nodesidxs, np.expand_dims(p, axis = 1)),axis = 1)
        scatter_pred_p.set_offsets(p)
        q = sel_real_features[:,1,i]
        q = np.concatenate((nodesidxs, np.expand_dims(q, axis = 1)),axis = 1)
        scatter_real_q.set_offsets(q)
        q = sel_pred_features[:,1,i]
        q = np.concatenate((nodesidxs, np.expand_dims(q, axis = 1)),axis = 1)
        scatter_pred_q.set_offsets(q)

        # ax[0].set_title('{:.2f} s'.format(float(times[i])))
        ax[0].set_xlim(0,features.shape[0])
        ax[0].set_ylim((minp, maxp))
        ax[1].set_xlim(0,features.shape[0])
        ax[1].set_ylim((minq, maxq))
 
        return scatter_pred_p
    
    os.makedirs(dirt, exist_ok=True)
    ani = animation.FuncAnimation(fig, animation_frame, frames=indices.size, interval=20)
    ani.save(f'{dirt}/{name}.gif', writer='pillow', dpi=100)

def interpolation_data(name_ref, dirt_ref, path_sec, dirt, var_dict):
    for name in name_ref:
        dim = 3
        data_dict_sec = {}
        with h5py.File(path_sec, 'r') as hf:
            for key in hf.keys():
                data_dict_sec[key] = np.array(hf[key])
        node_pos_x_sec = data_dict_sec['node_pos_x']
        node_pos_y_sec = data_dict_sec['node_pos_y']
        node_pos_z_sec = data_dict_sec['node_pos_z']
        mask_sec = data_dict_sec['mask']
        #if 'tri' in data_dict_sec:
        #    tri_sec = data_dict_sec['tri']
        bounds = np.array([node_pos_x_sec.min(),node_pos_x_sec.max(),
                           node_pos_y_sec.min(),node_pos_y_sec.max(),
                           node_pos_z_sec.min(),node_pos_z_sec.max()]).reshape(dim,2)

        data_dict_ref = {}
        with h5py.File(f'{dirt_ref}/{name}_pre.h5', 'r') as hf:
            for key in hf.keys():
                data_dict_ref[key] = np.array(hf[key])

        node_pos_x = data_dict_ref['node_pos_x']
        node_pos_y = data_dict_ref['node_pos_y']
        node_pos_z = data_dict_ref['node_pos_z']
        idx = ((node_pos_x>bounds[0,0]-0.1) & (node_pos_x<bounds[0,1]+0.1) & 
               (node_pos_y>bounds[1,0]-0.1) & (node_pos_y<bounds[1,1]+0.1) & 
               (node_pos_z>bounds[2,0]-0.005) & (node_pos_z<bounds[2,1]+0.005))
        node_pos_x_ref = node_pos_x[idx]
        node_pos_y_ref = node_pos_y[idx]
        node_pos_z_ref = node_pos_z[idx]

        data_dict = {}
        data_dict['node_pos_x'] = node_pos_x_sec
        data_dict['node_pos_y'] = node_pos_y_sec
        data_dict['node_pos_z'] = node_pos_z_sec
        #if 'tri' in data_dict:
        #    data_dict['tri'] = tri_sec

        for key in var_dict['node_label']:
            value_ref = data_dict_ref[key][idx]
            value = griddata((node_pos_x_ref, node_pos_y_ref, node_pos_z_ref), value_ref,
                             (node_pos_x_sec, node_pos_y_sec, node_pos_z_sec), method='linear')
            value = value*mask_sec

            data_dict[key] = value
            
        with h5py.File(f'{dirt}/{name}_intp.h5', "w") as f:
            for key, val in data_dict.items():
                f.create_dataset(key, data=val)

def plot_airfoil_anime(name_list, dirt):
    # rotate matrix
    dim = 3
    th1 = 0.0*np.pi; th2 = -0.15*np.pi; th3 = 0.0*np.pi
    c1 = np.array([1,0,0, 0,np.cos(th1),np.sin(th1), 0,-np.sin(th1),np.cos(th1)]).reshape(dim,3)
    c2 = np.array([np.cos(th2),0,np.sin(th2), 0,1,0, -np.sin(th2),0,np.cos(th2)]).reshape(dim,3)
    c3 = np.array([np.cos(th3),np.sin(th3),0, -np.sin(th3),np.cos(th3),0, 0,0,1]).reshape(dim,3)
    
    for name in name_list:
        # secection
        print(name)
        data_dict = {}
        with h5py.File(f'{dirt}/{name}_intp.h5', 'r') as hf:
            for key in hf.keys():
                data_dict[key] = np.array(hf[key])
        tmp = name.split('_')
        deg, mach = int(tmp[1]), float(tmp[3])
        node_pos_x = data_dict['node_pos_x']
        node_pos_y = data_dict['node_pos_y']
        node_pos_z = data_dict['node_pos_z']
        node_pos = np.concatenate([node_pos_x,node_pos_y,node_pos_z],1)
        node_pos = node_pos @ c1 @ c2 @ c3
        
        key = 'vel_x'
        value = data_dict[key]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(node_pos[:,0], node_pos[:,1], node_pos[:,2], c=value[:,0], cmap='ocean', s=10)
        
        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', shrink=0.65)
        cbar.set_ticks(np.linspace(value[:, 0].min(), value[:, 0].max(), num=11))
        cbar.set_ticklabels(['%.2f' % i for i in cbar.ax.get_yticks()])
        #cbar.remove()

        # airfoil data
        with h5py.File(f'./dataset/airfoil/raw/airfoil_surface.h5', 'r') as hf:
            for key in hf.keys():
                data_dict[key] = np.array(hf[key])
        node_pos_x = data_dict['node_pos_x']
        node_pos_y = data_dict['node_pos_y']
        node_pos_z = data_dict['node_pos_z']
        node_pos = np.concatenate([node_pos_x,node_pos_y,node_pos_z],1)
        node_pos = node_pos @ c1 @ c2 @ c3
        ax.scatter(node_pos[:,0], node_pos[:,1], node_pos[:,2], color=[0.9,0.9,0.9], s=10)
        
        ax.text(-0.15, 0.5, 0.5, f'degree: {deg}, mach: {mach}', fontsize=12, color='white')
        ax.set_aspect('auto')
        ax.view_init(elev=100, azim=-90)
        ax.axis('off')
    
        plt.savefig(f'{dirt}/{name}.jpg', dpi=150)
    
    imgs = []
    for name in name_list:
        tmp = name.split('_')
        deg, mach = int(tmp[1]), float(tmp[3])

        tmp = Image.open(f'{dirt}/{name}.jpg')
        imgs.append(tmp)
        if deg==0 or deg==30:
            for k in range(3):
                imgs.append(tmp)
    
    imgs[0].save(f'{dirt}/airfoil.gif', save_all=True, append_images=imgs, duration=200, loop=0)