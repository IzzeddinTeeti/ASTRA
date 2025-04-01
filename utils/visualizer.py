import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def visualize_adjacency_matrices(adjacency_matrices):
    
    # Convert torch tensor to numpy array for visualization
    adjacency_np = adjacency_matrices.cpu().numpy()
    
    # Extract batch size and number of frames
    batch_size, num_frames, _, _ = adjacency_np.shape
    
    for b in range(batch_size):
        fig, axes = plt.subplots(nrows=(num_frames + 1) // 2, ncols=2, figsize=(12, num_frames * 3))
        axes = axes.ravel()  # Flatten axes for easy indexing
        
        for f in range(num_frames):
            # Create graph from the adjacency matrix of the frame
            G = nx.from_numpy_array(adjacency_np[b, f])
            
            ax = axes[f]
            pos = nx.spring_layout(G)  # Compute node positions
            
            # Extract edge weights for adjusting the width of edges during visualization
            edge_weights = [d['weight'] for _, _, d in G.edges(data=True)]
            
            nx.draw_networkx_nodes(G, pos, ax=ax)
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.5, ax=ax)
            nx.draw_networkx_labels(G, pos, ax=ax)
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f"{d['weight']:.2f}" for i, j, d in G.edges(data=True)}, ax=ax)
            
            ax.set_title(f"Frame {f + 1}")
            ax.axis('off')

        for f in range(num_frames, len(axes)):
            fig.delaxes(axes[f])
        
        plt.tight_layout()
        plt.suptitle(f"Batch {b + 1}", fontsize=16)
        plt.subplots_adjust(top=0.95) 
        plt.show()



def world_to_pixel(world_coords, homographic, data, scale):
    world_coords = world_coords*scale
    ones = np.ones((*world_coords.shape[:-1],1))
    world_homogeneous = np.concatenate((world_coords, ones), axis = -1)
    pixel_final = []
    for world_coords in world_homogeneous:
        pixel_homogeneous = np.dot(world_coords, np.linalg.inv(homographic.T))
        if pixel_homogeneous.ndim == 1:
            pixel_homogeneous = pixel_homogeneous.reshape(1,-1)
        pixels_coord = pixel_homogeneous[:,:2]/(pixel_homogeneous[:,2].reshape(-1,1))
        pixels_coord = np.round(pixels_coord).astype(int)
        # Pixels of UCY crowd data, pixels index is in form (col, row)
        # Pixels of ETH data, pixels indexis in form (row, col) so converting all to (row, col)
        if data not in ('eth', 'hotel'):
            pixels_coord = pixels_coord[:,::-1]
        pixel_final.append(pixels_coord)
    pixel_final = np.array(pixel_final)
    return pixel_final

