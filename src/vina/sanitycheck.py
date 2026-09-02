import glob
import torch
import subprocess
import os
from vina import Vina

def rotation_matrix(alpha, beta, gamma):
    # Create rotation matrices for each axis
    R_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(alpha), -torch.sin(alpha)],
        [0, torch.sin(alpha), torch.cos(alpha)]
    ])

    R_y = torch.tensor([
        [torch.cos(beta), 0, torch.sin(beta)],
        [0, 1, 0],
        [-torch.sin(beta), 0, torch.cos(beta)]
    ])

    R_z = torch.tensor([
        [torch.cos(gamma), -torch.sin(gamma), 0],
        [torch.sin(gamma), torch.cos(gamma), 0],
        [0, 0, 1]
    ])

    # Combine the rotations: R = R_z * R_y * R_x
    R = torch.mm(R_z, torch.mm(R_y, R_x))
    return R
alpha = torch.randint(180, (1,)).deg2rad()
beta = torch.randint(180, (1,)).deg2rad()
gamma = torch.randint(180, (1,)).deg2rad()
R = rotation_matrix(alpha, beta, gamma)

def sdf_xyz(sdf_file):
    with open(sdf_file) as f:
        lines = f.readlines()
    n_atoms = int(lines[3].split()[0])
    xyz = []
    for i in range(4, 4+n_atoms): # atom block
        l = lines[i].split()
        xyz.append([
            float(l[0]), #x
            float(l[1]), #y
            float(l[2])  #z
        ]) 
    return torch.tensor(xyz)

def update_xyz(path, xyz, out_path):
    with open(path) as pdb_f:
        lines = [l.replace("\n", "") for l in pdb_f if l[:6].strip() == "HETATM"]
        unique_els = set(r[76:78].strip() for r in lines)
    new_lines = []
    assert len(lines) == xyz.shape[0], f"nlines != nxyz. pred: {xyz.shape}, true: {len(lines)}, els: {unique_els}"
    for i,l in enumerate(lines):
        if l[:6].strip() == "HETATM":
            x,y,z = xyz[i][0], xyz[i][1], xyz[i][2]
            new_line = f"{l[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{l[54:]}"
        new_lines.append(new_line)
    with open(out_path, "w") as pdb_out:
        pdb_out.write("\n".join(new_lines))

def xyz_from_pdb(pdb_p):
    with open(pdb_p) as pdb_f:
        lines = [l for l in pdb_f if l[:6].strip() == "HETATM"]
    xyz = torch.tensor([
        [
            float(r[30:38].strip()), #x
            float(r[38:46].strip()), #y
            float(r[46:54].strip())  #z
        ] for r in lines
    ])
    return xyz

benchmark_path = "/home/daniil/Downloads/vina_sanitycheck/databases/Vina/inputs"
benchmark_paths = glob.glob(f"{benchmark_path}/*")
adfr = "/home/daniil/Desktop/Ph.D/ADFR/bin"
outputs_path = "/home/daniil/Downloads/vina_sanitycheck/databases/Vina/outputs"
box_size = [25.0,25.0,25.0]
curr_dir = os.getcwd()


for case_path in benchmark_paths:
    case = case_path.split("/")[-1]
    out_dir = f"{outputs_path}/{case}"
    try:
        os.mkdir(out_dir)
    except:
        print(f"{out_dir} already created, skipping creation.")
        continue
    ligand_path = f"{case_path}/{case}_ligand.sdf"
    ligand_pdb = ligand_path.replace(".sdf", ".pdb")
    ligand_pdb_r = ligand_pdb.replace(".pdb", "_r.pdb")
    # rotate ligand:
    subprocess.run(f"obabel {ligand_path} -O {ligand_pdb}", shell=True)
    xyz = xyz_from_pdb(ligand_pdb)
    try:
        rand_lig_xyz = (xyz - xyz.mean(dim=0))@R + xyz.mean(dim=0)
    except Exception as e:
        print("error when rotating.")
        continue
    update_xyz(ligand_pdb, rand_lig_xyz, ligand_pdb_r)

    ligand_pdbqt = ligand_pdb_r.replace(".pdb", ".pdbqt")
    try:
        xyz = xyz_from_pdb(ligand_pdb_r)
    except:
        continue
    rec_p = f"{case_path}/{case}_protein.pdb"
    receptor_H = rec_p.replace(".pdb", "_H.pdb")
    receptor_pdbqt = rec_p.replace(".pdb", ".pdbqt")
    grid_center = xyz.mean(dim=0).tolist()

    os.chdir(case_path)
    subprocess.run(f"{adfr}/prepare_ligand -l {ligand_pdb_r} -o {ligand_pdbqt} -A hydrogens", shell=True)
    os.chdir(curr_dir)
    # reduce receptor:
    subprocess.run(
        f"{adfr}/reduce -Quiet -DB {adfr}/reduce_wwPDB_het_dict.txt {rec_p} > {receptor_H}",
        shell=True
    )
    subprocess.run(
        f"{adfr}/prepare_receptor -r {receptor_H} -o {receptor_pdbqt}",
        shell=True
    )

    # perform docking:
    try:
        v = Vina(sf_name="vina")
        v.set_receptor(receptor_pdbqt)
        v.set_ligand_from_file(ligand_pdbqt)
        v.compute_vina_maps(
            center=grid_center,
            box_size=box_size
        )
        # score the pose:
        energy = v.score()
        print(f"Score before energy minimization: {energy}")
        
        # dock the ligand:
        v.dock(exhaustiveness=32, n_poses=20)
        out_f = f"{out_dir}/{case}.pdbqt"
        v.write_poses(out_f, n_poses=20, overwrite=True)
        return_code = subprocess.run(f"obabel {out_f} -O {out_f.replace('.pdbqt', '.pdb')} -m", shell=True)
    except Exception as e:
        print(f"failed docking. Error: {e}")
