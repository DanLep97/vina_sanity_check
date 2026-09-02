import torch
import glob
import re
import subprocess
from posebusters import PoseBusters
from pathlib import Path
import pandas
import multiprocessing as mp
import logging
import os
import time
import shutil
import signal
logging.basicConfig(level=logging.ERROR)

def compute_rmsd(data):
    target_paths = data[0]
    pred_paths = data[1]
    names = data[3]

    target_paths = list(map(lambda x: x.replace(".pdb", ".sdf"), target_paths))
    pred_paths = list(map(lambda x: x.replace(".pdb", ".sdf"), pred_paths))
    assert len(target_paths) == len(pred_paths) and len(names) == len(target_paths), \
        f"in compute rsmd, target len: {len(target_paths)}, pred len: {len(pred_paths)}"
    rmsds = {}
    for target_path, pred_path, name in zip(target_paths, pred_paths, names):
        try:
            same_order = same_atom_order(target_path, pred_path)
        except Exception:
            same_order = False
        if same_order:
            target_xyz = sdf_xyz(target_path)
            pred_xyz = sdf_xyz(pred_path)
            rmsd = (target_xyz-pred_xyz).pow(2).sum(dim=1).mean().sqrt()
            rmsd = rmsd.item()
        else:
            rmsd = None
        rmsds[name] = rmsd
    return rmsds

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

def sdf_atom_order(sdf_file):
    with open(sdf_file) as f:
        lines = f.readlines()
    n_atoms = int(lines[3].split()[0])
    atoms = []
    for i in range(4, 4+n_atoms): # atom block
        l = lines[i].split()
        atoms.append(l[3].strip())
    return atoms

def same_atom_order(target_path, pred_path):
    # get target atom order:
    target_atom_order = sdf_atom_order(target_path)
    pred_atom_order = sdf_atom_order(pred_path)
    return target_atom_order == pred_atom_order


def bust(true_path, pred_path, cond_file, name, i, tool_name):
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out")

    buster = PoseBusters(config="redock")
    # convert the file into sdf if it's a pdb:
    if ".pdb" in pred_path:
        sdf_pred = pred_path.replace(".pdb", ".sdf")
        if not os.path.exists(sdf_pred):
            subprocess.run(
                f"obabel {pred_path} -O {sdf_pred}",
            shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else:
        sdf_pred = pred_path
    sdf_truth = true_path.replace(".pdb", ".sdf")
    try:
        df = buster.bust([Path(sdf_pred)], Path(sdf_truth), Path(cond_file), full_report=True)
    except Exception as e:
        return None
    df["ligand_detail"] = [name.replace("_trimed", "")]
    df["blind_docking"] = ["_trimed" not in name]
    df["same_atom_order"] = same_atom_order(sdf_truth, sdf_pred)
    df["tool"] = tool_name
    return df

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


def get_busters_inputs(names, preds, pred_files_dir, tool_name):
    true_files = []
    pred_files = []
    cond_files = []
    failed_convertions = 0
    final_names = []
    for i, name in enumerate(names):
        true_file = f"../../databases/QBioLiP/nr_lig/{name.replace('_trimed','')}.pdb"
        cond_file_base = f"../../databases/QBioLiP/nr_rec{'_trimed' if 'trimed' in name else ''}"
        cond_file = f"{cond_file_base}/{name.replace('_trimed', '') if 'trimed' in name else name[:6]}.pdb"
        if len(preds[i].shape) == 2: # oneshot prediction
            pred_file = f"{pred_files_dir}/{name}.pdb"
            update_xyz(true_file, preds[i], pred_file)
            pred_files.append(pred_file.strip())
            true_files.append(true_file)
            cond_files.append(cond_file)
            final_names.append(name)
        else: # ranked predictions
            for j,pred in enumerate(preds[i]):
                pred_file = f"{pred_files_dir}/{name}_{j}.pdb"
                try:
                    update_xyz(true_file, pred, pred_file)
                    pred_files.append(pred_file)
                    true_files.append(true_file)
                    cond_files.append(cond_file)
                    final_names.append(name + f"_{j}")
                except Exception as e:
                    failed_convertions += 1
                    print(e)
    print("Percentage of failed buster input generations:", failed_convertions/len(names)*100)
    return true_files, pred_files, cond_files, final_names, list(range(len(final_names))), [tool_name]*len(final_names)

def compute_centroid_deviation(data):
    if len(data) == 3:
        all_preds, all_targets, names = data
    else:
        all_preds, all_targets, names, ranks = data
    cds = [] # centroid deviations
    for i, (preds, target) in enumerate(zip(all_preds, all_targets)):
        if len(preds.shape) == 2: # one shot prediction like equibind
            cd = (preds.mean(dim=0)-target.mean(dim=0)).norm()
            cds.append(cd)
        else:
            preds_cds = []
            for pred in preds:
                try:
                    cd = (pred.mean(dim=0)-target.mean(dim=0)).norm()
                    preds_cds.append(cd)
                except Exception as e:
                    print("error computing centroid deviation:", e)
                    print("For case: ", names[i])
                    exit()
            cds.append(preds_cds)
    return torch.tensor(cds), names


def xyz_from_sdf(sdf_p, n_atoms=None):
    with open(sdf_p) as sdf_f:
        lines = [l.split() for l in sdf_f]
    lines = [l for l in lines if len(l) == 16] #only xyz coordinates lines
    xyz = torch.tensor([
        [
            float(l[0]), #x
            float(l[1]), #y
            float(l[2]), #z
        ]
    for l in lines])
    if n_atoms is not None:
        n_mols = int(xyz.shape[0]/n_atoms)
        return xyz.view(n_mols, n_atoms, 3)
    return xyz 

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
def align_dpl_outputs(output_path, failed_path):
    output_folders = glob.glob(output_path)
    diff_atm_order_count = 0
    failed_aligns = 0
    # move all failed cases back into outputs
    # this allows to re-run this block of code in case something went wrong.
    failed_folders = glob.glob(failed_path + "/*")
    for failed_folder in failed_folders:
        failed_name = failed_folder.split("/")[-1]
        shutil.move(failed_folder, f"{output_path.replace('/*','')}/{failed_name}")
    for folder in output_folders:
        name = folder.split("/")[-1]
        assembly_name = name[:6]
        pred_files_sdf = f"{folder}/sample_ligand.sdf" # this file has all predictions at ounce
        pred_files_pdb_prefix = f"{folder}/sample_ligand_.pdb"
        # convert each of them into pdbs to further obtain the rotation matrix for each of them:
        if not os.path.exists(pred_files_sdf):
            shutil.move(folder, f"{failed_path}/{name}")
            continue
        sdf2pdbs_command_code = subprocess.run(
            f"obabel {pred_files_sdf} -opdb -O {pred_files_pdb_prefix} -m",
        shell=True, check=True)
        is_trimed = "_trimed" in name
        gt_rec_path = f"../../databases/QBioLiP/nr_rec_trimed/{name.replace('_trimed','')}.pdb" if is_trimed else f"../../databases/QBioLiP/nr_rec/{assembly_name}.pdb"
        gt_lig_path = f"../../databases/QBioLiP/nr_lig/{name.replace('_trimed','')}.pdb"
        with open(gt_lig_path) as lig_f:
            gt_atm_order = [r[78:80].strip() for r in lig_f if r[:6].strip() == "HETATM"]
        to_align = f"{folder}/sample_protein.pdb"
        aligned = f"{folder}/sample_protein_aligned.pdb"
        align_structures(to_align, gt_lig_path, gt_rec_path, aligned, with_matrix=True)
        matrix_path = f"{folder}/matrix.txt"
        # after alignment, get the rototranslational modifications:
        if not os.path.exists(matrix_path):
            failed_aligns += 1
            shutil.move(folder, f"{failed_path}/{name}")
            continue
        with open(matrix_path) as matrix_f:
            lines = [l.split() for l in matrix_f]
        matrix_info = lines[2:5]
        t = torch.tensor([float(l[1]) for l in matrix_info])
        r = torch.tensor([[float(i[j]) for j in range(2,5)] for i in matrix_info])
        pred_files_pdbs = glob.glob(f"{folder}/sample_ligand_*.pdb")
        # using the rototranslational modifications needed to align the pdb, apply it to predicted ligand atoms:
        skip_case = False
        for pdb in pred_files_pdbs:
            with open(pdb) as lig_f:
                atm_order = [r[78:80].strip() for r in lig_f if r[:6].strip() == "HETATM"]
            if len(atm_order) != len(gt_atm_order):
                shutil.move(folder, f"{failed_path}/{name}")
                diff_atm_order_count += 1
                skip_case = True
                break
            xyz = xyz_from_pdb(pdb)
            # apply aligning rototranslation:
            xyz = xyz@r + t
            update_xyz(pdb, xyz, pdb)
        if skip_case:
            continue
    print("Number of cases with unmatching atom order for DPL:", diff_atm_order_count)
    print("Number of failed alignment for DPL:", failed_aligns)

def get_cond_path(name, rec_path, trimmed_rec_path):
    trimmed = "_trimed" in name
    cond_root = trimmed_rec_path if trimmed else rec_path
    cond_name = name.replace("_trimed", "") if trimmed else name[:6]
    return f"{cond_root}/{cond_name}.pdb"

def get_crystal_bust_inputs(rec_path, df):
    true_files = []
    pred_files = []
    cond_files = []
    names = []
    tool_name = "Crystal"

    for i in range(df.shape[0]):
        case = df.iloc[i]
        true_files.append(f"../../databases/QBioLiP/nr_lig/{case.ligand_detail}.pdb")
        pred_files.append(f"../../databases/QBioLiP/nr_lig/{case.ligand_detail}.pdb")
        cond_files.append(get_cond_path(case.ligand_detail, rec_path, rec_path))
        names.append(case.ligand_detail)
    n = len(names)
    return true_files, pred_files, cond_files, \
        names, list(range(n)), [tool_name]*n

def get_dpl_bust_inputs(output_path, rec_path, trimmed_rec_path, buster_root):
    true_files = []
    pred_files = []
    cond_files = []
    failed_cases = 0
    names = []
    tool_name = "DPL"

    output_folders = glob.glob(output_path)
    for i, folder in enumerate(output_folders):
        name = folder.split("/")[-1]
        target_f = f"../../databases/QBioLiP/nr_lig/{name.replace('_trimed','')}.pdb"
        pred_file = f"{folder}/sample_ligand_1.pdb" # best ligand
        to_bust = f"{buster_root}/{name}.pdb"
        cond_path = get_cond_path(name, rec_path, trimmed_rec_path)
        if os.path.exists(pred_file):
            shutil.copyfile(pred_file, to_bust)
            pred_files.append(to_bust)
            names.append(name)
            true_files.append(target_f)
            cond_files.append(cond_path)
        else:
            failed_cases+=1
    print(f"For DPL buster input preparation, number of failed cases: {failed_cases}")
    n = len(names)    
    return true_files, pred_files, cond_files, \
        names, list(range(n)), [tool_name]*n

def get_boltz_bust_inputs(output_path, rec_path, trimmed_rec_path):
    names = []
    true_files = []
    pred_files = []
    cond_files = []
    tool_name = "Boltz-2"

    results = glob.glob(output_path)
    for pdb in results:
        name = pdb.split("/")[-2] # name of the folder
        is_trimed = "_trimed" in name
        assembly_name = name[:6]
        gt_rec_path = f"{trimmed_rec_path}/{name.replace('_trimed','')}.pdb" if is_trimed\
            else f"{rec_path}/{assembly_name}.pdb"
        gt_lig_path = f"../../databases/QBioLiP/nr_lig/{name.replace('_trimed','')}.pdb"
        complex_aligned = pdb.replace(".pdb", "_aligned.pdb")
        # align complex to ground truth receptor:
        align_structures(pdb, gt_lig_path, gt_rec_path, complex_aligned)
        if not os.path.exists(complex_aligned):
            continue
        
        # get only the ligand atoms:
        with open(complex_aligned, "r") as complex_f:
            complex_lines = [l for l in complex_f]
        substrate_records = [l for l in complex_lines if l[:6].strip() == "HETATM" or l[:6].strip() == "CONECT"]
        # save substrate:
        substrate_path = complex_aligned.replace(".pdb", "_substrate.pdb")
        with open(substrate_path, "w") as substrate_f:
            substrate_f.write("".join(substrate_records))
        
        names.append(name)
        true_files.append(gt_lig_path)
        pred_files.append(substrate_path)
        cond_files.append(gt_rec_path)
    n = len(names)
    return true_files, pred_files, cond_files, \
        names, list(range(n)), [tool_name]*n

def get_dynamicbind_bust_inputs(output_path, rec_path, trimmed_rec_path):
    result_folders = glob.glob(output_path)
    result_folders.sort()

    names = []
    true_files = []
    pred_files = []
    cond_files = []
    tool_name = "DynamicBind"

    for folder in result_folders:
        if "affinity_prediction.csv" in folder:
            continue
        cases = glob.glob(f"{folder}/*")
        # target_file = [c for c in cases if re.search(r'')]
        protein_name = [c.split("/")[-1].replace(".pdb", "") for c in cases if re.search(r'^[0-9a-z]{4}_[0-9A-Z_]*.pdb', c.split("/")[-1]) is not None][0]
        ligand_names = [c.split("/")[-1].replace(".sdf", "") for c in cases if re.search(r'^[0-9a-z]{4}_[0-9A-Z_]*_randconf.sdf', c.split("/")[-1]) is not None]
        if len(ligand_names) != 1:
            continue
        target_file = f"{folder}/{ligand_names[0]}.sdf"
        ligand_name = ligand_names[0].replace("_randconf", "")
        name = ligand_name if ligand_name != protein_name else f"{ligand_name}_trimed"
        pred_sdfs = [c for c in cases if re.search(r"rank[0-9]{1,2}_ligand_[a-z_0-9.]*_relaxed.sdf", c.split("/")[-1]) is not None]
        if len(pred_sdfs) < 10:
            continue
        ranks = torch.tensor([int(re.findall(r'(?<=rank)[0-9]{1,2}(?=_)', c)[0]) for c in pred_sdfs])
        if len(ranks) == 0:
            continue
        ranks_sorted = ranks.argsort(descending=False)

        names.append(name)
        true_files.append(target_file)
        pred_files.append(pred_sdfs[ranks_sorted[0]])
        cond_files.append(get_cond_path(name, rec_path, trimmed_rec_path))
    n = len(names)    
    return true_files, pred_files, cond_files, \
        names, list(range(n)), [tool_name]*n

def get_tankbind_bust_inputs(output_path, rec_path, trimmed_rec_path):
    outputs = torch.load(output_path, weights_only=False)
    names = []
    true_files = []
    pred_files = []
    cond_files = []
    tool_name = "TankBind"
    for name, pred in outputs.items():
        names.append(name)
        pred_files.append(f"../../databases/TankBind/outputs/lig_pdb/{name}.pdb")
        true_files.append(f"../../databases/QBioLiP/nr_lig/{name.replace('_trimed','')}.pdb")
        cond_files.append(get_cond_path(name, rec_path, trimmed_rec_path))
    n = len(names)
    return true_files, pred_files, cond_files, \
        names, list(range(n)), [tool_name]*n

def get_vina_bust_inputs(output_path, rec_path, trimmed_rec_path):
    outputs = glob.glob(output_path)

    names = []
    true_files = []
    pred_files = []
    cond_files = []
    failed_cases = 0
    all_cases = 0

    tool_name = "Vina"

    for output in outputs:
        name = output.split("/")[-1].split(".")[0]
        files = glob.glob(f"{output}/*.pdb")
        if len(files) < 10:
            continue
        output_scores = []
        target_file = f"../../databases/QBioLiP/nr_lig/{name.replace('_trimed', '')}.pdb"
        with open(target_file) as target_f:
            target_lines = [l for l in target_f if l[:6].strip() == "HETATM" and l[76:78].strip() != "H"]
        target_atom_names = [
            r[12:16].strip() for r in target_lines 
        ]
        for file in files:
            with open(file) as pdb_f:
                pred_lines = [l for l in pdb_f]
            first_lines = pred_lines[:2]
            try:
                score = pred_lines[1].split()[3:]
            except Exception as e:
                print(f"Error for {file}")
                print(e)
                exit()
            score = float(score[0]) # focus on predicted BA
            pred_lines = [l for l in pred_lines if l[:6].strip() == "HETATM" and l[76:78].strip() != "H"]
            if all(r[12:16].strip() in target_atom_names for r in pred_lines):
                pred_atom_names_correct_indices = [
                    target_atom_names.index(r[12:16].strip())
                    for r in pred_lines
                ] # the output of vina doesn't allways have the same atom order than the original PDB.
                # This is fixed by re-arranging the xyz coordinates based on the order of the target's atoms.
                all_cases += 1
            else:
                failed_cases += 1
                continue
            if not all(
                i in pred_atom_names_correct_indices for i in range(len(pred_lines))
            ) or len(pred_lines)!= len(target_lines):
                failed_cases += 1
                continue
            pred_lines = [
                pred_lines[pred_atom_names_correct_indices.index(i)] for i in range(len(pred_lines))
            ]
            # save the file with correct atom order:
            with open(file, "w") as f:
                f.write("".join(first_lines + pred_lines))
            output_scores.append(score)
        output_scores = torch.tensor(output_scores)
        scores_sorted = output_scores.argsort(descending=False) # the lower the better
        names.append(name)
        cond_files.append(get_cond_path(name, rec_path, trimmed_rec_path))
        pred_files.append(files[scores_sorted[0]])
        true_files.append(target_file)
    print("Failed cases for Vina: ", failed_cases)
    print("All cases for Vina: ", all_cases)
    print("Percentage failed: ", failed_cases/all_cases)
    n = len(names)
    return true_files, pred_files, cond_files, \
        names, list(range(n)), [tool_name]*n

def get_neuralplexer_bust_inputs(output_paths, rec_path, trimed_rec_path):
    output_paths = glob.glob(output_paths)
    true_files = []
    pred_files = []
    cond_files = []
    names = []
    tool_name = "NeuralPLexer"

    # fetch outputs:
    for p in output_paths:
        outputs = glob.glob(f"{p}/*.sdf")
        if len(outputs) == 0:
            continue
        output_ranks = []
        output_files = []
        for o in outputs:
            filename = o.split("/")[-1]
            if filename == "lig_all.sdf":
                continue
            if filename == "lig_ref.sdf":
                true_files.append(o)
            else:
                output_files.append(o)
                output_ranks.append(int(re.search(r"(?<=lig_rank)[0-9]{1,2}(?=_)", filename).group()))
        output_ranks = torch.tensor(output_ranks)
        ranks_sorted = output_ranks.argsort(descending=False)[:20] # the lower the better
        # save best file:
        pred_files.append(output_files[ranks_sorted[0]])
        name = p.split("/")[-1]
        names.append(name)
        cond_file = get_cond_path(name, rec_path, trimed_rec_path)
        cond_files.append(cond_file)
    n = len(names)
    return true_files, pred_files, cond_files, \
        names, list(range(n)), [tool_name]*n

def get_equibind_bust_inputs(
        equibind_data, rec_path, trimed_rec_path
    ):
    true_files = []
    pred_files = []
    cond_files = []
    names = []
    tool_name = "EquiBind"

    outputs = glob.glob(f"{equibind_data}/*")
    for o in outputs:
        name = o.split("/")[-1] 
        pred_file = f"{o}/lig_equibind_corrected.sdf"
        cond_file = get_cond_path(name, rec_path, trimed_rec_path)
        true_file = f"../../databases/QBioLiP/nr_lig/{name.replace('_trimed', '')}.pdb"

        names.append(name)
        true_files.append(true_file)
        pred_files.append(pred_file)
        cond_files.append(cond_file)
    n = len(names)    
    return true_files, pred_files, cond_files, \
        names, list(range(n)), [tool_name]*n

def get_diffdock_bust_inputs(output_folders, rec_path, trimmed_rec_path):
    true_files = []
    pred_files = []
    cond_files = []
    final_names = []
    tool_name = "DiffDock"

    output_folders = glob.glob(output_folders)
    for folder in output_folders:
        # get the target xyz coordinates:
        name = folder.split("/")[-1]
        target_path = f"../../databases/QBioLiP/nr_lig/{name.replace('_trimed', '')}.pdb"
        folder_outputs = glob.glob(f"{folder}/*.sdf")
        if len(folder_outputs) == 0:
            continue
        # get prediction data:   
        folder_confidences = []
        folder_predicted = []
        for sdf_path in folder_outputs:
            # check if confidence is available:
            confidence_match = re.search(r"([-]|(?<=e))\d.\d{1,2}", sdf_path.split("/")[-1])
            if confidence_match is None:
                continue
            confidence = float(confidence_match.group())
            folder_confidences.append(confidence)
            folder_predicted.append(sdf_path)
        # rank the prediction based on the confidence
        folder_confidences = torch.tensor(folder_confidences)
        ranks = folder_confidences.argsort(descending=True)[:20] # the higher the better
        
        # store data
        pred_files.append(folder_predicted[ranks[0]])
        final_names.append(name)
        true_files.append(target_path)
        cond_files.append(get_cond_path(name, rec_path, trimmed_rec_path))

    return true_files, pred_files, cond_files, \
        final_names, list(range(len(final_names))), [tool_name]*len(final_names)

def align_atoms(gt_pdb, pred_pdb):
    # get target atom order:
    with open(gt_pdb) as target_f:
        target_lines = [l for l in target_f if l[:6].strip() == "HETATM" and l[76:78].strip() != "H"]
    target_atom_names = [
        r[12:16].strip() for r in target_lines 
    ]
    # get pred atom order and coordinates:
    with open(pred_pdb) as pred_f:
        pred_lines = [l for l in pred_f if l[:6].strip() == "HETATM" and l[76:78].strip() != "H"]
    pred_atom_names = [
        r[12:16].strip() for r in pred_lines 
    ]
    pred_xyz = torch.tensor([
        [
            float(r[30:38].strip()), #x
            float(r[38:46].strip()), #y
            float(r[46:54].strip())  #z
        ] for r in pred_lines 
    ])
    # organize correct atom order:
    if all(pred_atom_name in target_atom_names for pred_atom_name in pred_atom_names):
        pred_atom_names_correct_indices = [
            target_atom_names.index(pred_atom_name)
            for pred_atom_name in pred_atom_names
        ] # the output of vina doesn't allways have the same atom order than the original PDB.
        pred_xyz = torch.stack([
            pred_xyz[pred_atom_names_correct_indices.index(i)] for i in range(pred_xyz.shape[0])
        ])
        update_xyz(pred_pdb, pred_xyz, pred_pdb)

def get_alphafold_buster_inputs(output_folder):
    # to run this function you need to have tmalign as a terminal command
    targets = []
    preds = []
    conditioned = []
    outputs = glob.glob(output_folder)
    names = []
    failed_cases = 0
    for o in outputs:
        o_name = o.split("/")[-1]
        assembly_name = o_name[:6]
        name = assembly_name + o_name[6:].upper().replace("_TRIMED", "")

        # ground-truth pdb ligand:
        gt_path = f"../../databases/QBioLiP/nr_lig/{name}.pdb" 

        # protein:
        is_trimed = "_trimed" in o_name
        rec_path = f"../../databases/QBioLiP/nr_rec_trimed/{name}.pdb" if is_trimed else f"../../databases/QBioLiP/nr_rec/{assembly_name}.pdb"
        file_name = name if not is_trimed else name + "_trimed"

        # retreive outputs and convert them to pdb:
        # convert AF3 output to pdb, if not already:
        top_rank_cif = f"{o}/{o_name}_model.cif"
        top_rank_pdb_unaligned = top_rank_cif.replace(".cif",".pdb")
        top_rank_pdb_aligned = f"../../databases/alphafold3/pred_files/{file_name}_aligned.pdb"
        if not os.path.exists(top_rank_pdb_unaligned):
            cif2pdb_command_code = subprocess.run(
                f"obabel {top_rank_cif} -O {top_rank_pdb_unaligned}",
            shell=True, check=True)

        if align_structures(top_rank_pdb_unaligned, gt_path, rec_path, top_rank_pdb_aligned) is None:
            failed_cases+=1
            continue

        # output pdb ligand as a separate file:
        try:
            with open(top_rank_pdb_aligned) as pdb_f:
                pdb_atom_records = [l for l in pdb_f if l[:6].strip() == "ATOM"] # everything in AF3 is ATOM, even HETATM
        except:
            failed_cases+=1
            if os.path.exists(top_rank_pdb_aligned):
                os.remove(top_rank_pdb_aligned)
            continue
        # For that reason the ligand is the smallest ATOM chain:
        if len(pdb_atom_records) == 0:
            continue
        ligand_chain = pandas.Series(
            l[21] for l in pdb_atom_records 
        ).value_counts().sort_values().index[0] # count the lowest number of atoms in each chain: that's the ligand's chain
        o_ligand = [l for l in pdb_atom_records if l[21] == ligand_chain]
        o_pdb = f"../../databases/alphafold3/pred_files/{file_name}.pdb"
        o_sdf = o_pdb.replace(".pdb", ".sdf")

        with open(o_pdb, "w") as o_pdb_f:
            o_pdb_f.write("".join(o_ligand))
    
        # add also the sdf file for buster input
        if not os.path.exists(o_sdf):
            cif2pdb_command_code = subprocess.run(
                f"obabel {o_pdb} -O {o_sdf}",
            shell=True, check=True)

        preds.append(o_sdf)
        targets.append(gt_path.replace(".pdb", ".sdf"))
        conditioned.append(rec_path)
        names.append(file_name)
        ## align alphafold3 output with input
    print(f"AlphaFold3: target paths: {len(targets):,}; pred paths: {len(preds):,}; conditioned paths: {len(conditioned):,}, names: {len(names)}")
    print(f"Alphafold3: {failed_cases} failed cases ")
    return targets, preds, conditioned, names, list(range(len(preds))), ["AlphaFold3"]*len(preds)

def get_alphafold_outputs(data):
    #inputs
    pred_paths = data[1]
    target_paths = data[0]

    #outputs
    pred_xyzs = []
    target_xyzs = []
    names = []
    for i, pred_path in enumerate(pred_paths):
        pred_path = pred_path.replace(".sdf", ".pdb")
        target_path = target_paths[i].replace(".sdf", ".pdb")
        with open(pred_path) as pred_f:
            pred_atoms = [l for l in pred_f if l[:6].strip() == "ATOM" and l[76:78].strip() != "H"]
            pred_xyz = torch.tensor([
                [
                    float(r[30:38].strip()), #x
                    float(r[38:46].strip()), #y
                    float(r[46:54].strip())  #z
                ] for r in pred_atoms
            ])
        with open(target_path) as target_f:
            target_atoms = [l for l in target_f if l[:6].strip() == "HETATM" and l[76:78].strip() != "H"]
            target_xyz = torch.tensor([
                [
                    float(r[30:38].strip()), #x
                    float(r[38:46].strip()), #y
                    float(r[46:54].strip())  #z
                ] for r in target_atoms
            ])
        assert pred_xyz.shape[0] > 0 and target_xyz.shape[0] > 0, f"Get AF3 outputs: problem with {pred_path}, no atoms in tensor."
        name = pred_path.split("/")[-1].replace(".pdb","")
        target_xyzs.append(target_xyz)
        pred_xyzs.append(pred_xyz)
        names.append(name)
    print(f"AF3 outputs: {len(target_xyzs)} targets, {len(pred_xyzs)} preds, {len(names)} names")
    return pred_xyzs, target_xyzs, names

def align_structures(
        to_align_path,
        gt_lig_path, #ground truth
        gt_rec_path,
        aligned_path,
        with_matrix = False,
    ):
    # align the model as a monomer alignment based on the chain interacting with the ligand.
    # Make the distogram for ligand-protein interaction:
    # receptor coordinates:
    with open(gt_rec_path) as rec_f:
        rec_atoms = [l for l in rec_f if l[:6].strip() == "ATOM"]
    rec_xyz = torch.tensor([
        [
            float(r[30:38].strip()), #x
            float(r[38:46].strip()), #y
            float(r[46:54].strip())  #z
        ] for r in rec_atoms
    ])
    # ligand coordinates:
    with open(gt_lig_path) as lig_f:
        lig_atoms = [l for l in lig_f if l[:6].strip() == "HETATM"]
    lig_xyz = torch.tensor([
        [
            float(r[30:38].strip()), #x
            float(r[38:46].strip()), #y
            float(r[46:54].strip())  #z
        ] for r in lig_atoms
    ])
    # distogram:
    n_rec = rec_xyz.shape[0] # number of receptor atoms
    n_lig = lig_xyz.shape[0] # number of ligand atoms
    lig_matrix = lig_xyz.tile(n_rec,1).view(n_lig,n_rec,3)
    rec_matrix = rec_xyz.tile(n_lig,1).view(n_lig,n_rec,3)
    lig_rec_distogram = (lig_matrix-rec_matrix).pow(2).sum(dim=2).sqrt()
    # deduce the interacting atoms and chain from the distogram:
    # ADD 1 TO INDICES BECAUSE IN PDB INDEXING STARTS WITH 1!!!!!
    interacting_atom_indices = set(((lig_rec_distogram < 10).nonzero(as_tuple=True)[1]+1).tolist())
    interacting_chains = pandas.Series([l[21] for l in rec_atoms if int(l[6:11].strip()) in interacting_atom_indices])
    interacting_chain_counts = interacting_chains.value_counts()
    # use the chain with the most interactions with the ligand to align this chain,
    # from the prediction, to the ground truth receptor chain:
    try:
        most_interacting_chain = interacting_chain_counts.sort_values().index[-1] # sorted from low to high
    except Exception as e:
        print("error aligning: ", e)
        return
    # the chain naming is the same between predictions and receptor files
    if not with_matrix:
        usalign_command_code = subprocess.run(
            f"usalign {to_align_path} {gt_rec_path} -m prot -mm 0 -o {aligned_path.replace('.pdb','')} -chain1 {most_interacting_chain} -chain2 {most_interacting_chain}",
        shell=True, check=True)
    else:
        matrix_path = "/".join(to_align_path.split("/")[:-1]) + "/matrix.txt"
        usalign_command_code = subprocess.run(
            f"usalign {to_align_path} {gt_rec_path} -m prot -mm 0 -o {aligned_path.replace('.pdb','')} -chain1 {most_interacting_chain} -chain2 {most_interacting_chain} -m {matrix_path}",
        shell=True, check=True)
    return 1

if __name__ == "__main__":
    sdf_atoms = sdf_atom_order("../../databases/Vina/outputs/8u37_1_V5U_B/8u37_1_V5U_B1.sdf")
    xyz = sdf_xyz("../../databases/Vina/outputs/8u37_1_V5U_B/8u37_1_V5U_B1.sdf")
    print(sdf_atoms)
    print(len(sdf_atoms))
    print(xyz)
    print(xyz.shape)
    