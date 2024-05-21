import numpy as np
import sys

def read_xyz(file_path):
    """
    Lit un fichier XYZ et retourne les coordonnées atomiques.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_atoms = int(lines[0].strip())
        atoms = []
        for line in lines[2:num_atoms + 2]:  # Les deux premières lignes ne contiennent pas de coordonnées
            parts = line.split()
            atom_type = parts[0]
            x, y, z = map(float, parts[1:])
            atoms.append((atom_type, x, y, z))
    return atoms

def fit_plane(points):
    """
    Ajuste un plan aux points donnés en utilisant les moindres carrés.
    """
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    u, s, vh = np.linalg.svd(centered_points)
    normal = vh[-1]
    return centroid, normal

def reorient_molecule(atoms):
    """
    Réoriente la molécule pour qu'elle soit dans le plan (xy).
    """
    coordinates = np.array([(x, y, z) for atom_type, x, y, z in atoms])
    centroid, normal = fit_plane(coordinates)
    
    if normal[2] < 0:
        normal = -normal
    
    z_axis = np.array([0, 0, 1])
    v = np.cross(normal, z_axis)
    c = np.dot(normal, z_axis)
    
    if np.allclose(normal, z_axis):
        rotation_matrix = np.eye(3)
    else:
        k = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + k + k @ k * ((1 - c) / (1 + c))
    
    reoriented_coordinates = (coordinates - centroid) @ rotation_matrix.T + centroid
    reoriented_atoms = [(atoms[i][0], *reoriented_coordinates[i]) for i in range(len(atoms))]
    
    return reoriented_atoms

def check_planarity(atoms, tolerance):
    """
    Vérifie si la molécule est plane selon une tolérance spécifiée.
    """
    z_coords = [z for atom_type, x, y, z in atoms]
    height = max(z_coords) - min(z_coords)
    return height <= tolerance

def create_grid(atoms, delta):
    """
    Crée une grille rectangulaire au-dessus de la molécule à 1 angstrom d'altitude.
    """
    # Extraire les coordonnées des atomes
    coordinates = np.array([(x, y, z) for atom_type, x, y, z in atoms])
    
    # Trouver les limites de la grille
    min_coords = np.min(coordinates, axis=0)
    max_coords = np.max(coordinates, axis=0)
    
    # Définir les dimensions de la grille
    x_min, y_min, z_min = min_coords
    x_max, y_max, z_max = max_coords
    z_grid = z_max + 1.0  # La grille est placée 1 angstrom au-dessus de la molécule
    
    # Créer les points de la grille
    x_points = np.arange(x_min, x_max + delta, delta)
    y_points = np.arange(y_min, y_max + delta, delta)
    
    grid_points = [(x, y, z_grid) for x in x_points for y in y_points]
    
    return grid_points

def write_grid_to_xyz(atoms, grid_points, output_file):
    """
    Écrit les coordonnées de la molécule et les points de la grille dans un fichier au format XYZ.
    """
    total_points = len(atoms) + len(grid_points)
    with open(output_file, 'w') as file:
        file.write(f"{total_points}\n")
        file.write("Géométrie de la molécule suivie de la grille\n")
        
        # Écrire les coordonnées des atomes
        for atom_type, x, y, z in atoms:
            file.write(f"{atom_type} {x:.6f} {y:.6f} {z:.6f}\n")
        
        # Écrire les points de la grille
        for x, y, z in grid_points:
            file.write(f"X {x:.6f} {y:.6f} {z:.6f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py input.xyz delta tolerance output.xyz")
        sys.exit(1)
    
    input_file = sys.argv[1]
    delta = float(sys.argv[2])
    tolerance = float(sys.argv[3])
    output_file = sys.argv[4]
    
    atoms = read_xyz(input_file)
    reoriented_atoms = reorient_molecule(atoms)
    if not check_planarity(reoriented_atoms, tolerance):
        print("Erreur : La molécule n'est pas plane selon la tolérance spécifiée.")
        sys.exit(1)
    
    grid_points = create_grid(reoriented_atoms, delta)
    write_grid_to_xyz(reoriented_atoms, grid_points, output_file)
    
    print(f"Grille générée et écrite dans {output_file}")

