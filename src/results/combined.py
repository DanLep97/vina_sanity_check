from repo import bust
import torch
import glob
import os
import subprocess

if __name__ == "__main__":
    benchmark_path = "../../Vina/inputs"

    def get_vina_bust_inputs(output_path):
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
            target_file = f"{benchmark_path}/{name}/{name}_ligand.pdb"
            if not os.path.exists(target_file):
                subprocess.run(f"obabel {target_file.replace('.pdb', '.sdf')} -O {target_file}", shell=True)
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
                output_scores.append(score)
            output_scores = torch.tensor(output_scores)
            scores_sorted = output_scores.argsort(descending=False) # the lower the better
            names.append(name)
            cond_files.append(f"{benchmark_path}/{name}/{name}_protein.pdb")
            pred_files.append(files[scores_sorted[0]])
            true_files.append(target_file)
        if failed_cases > 0:
            print("Failed cases for Vina: ", failed_cases)
            print("All cases for Vina: ", all_cases)
            print("Percentage failed: ", failed_cases/all_cases)
        n = len(names)
        return true_files, pred_files, cond_files, \
            names, list(range(n)), [tool_name]*n
    import sys
    sys.path.append("../preprocess")
    import multiprocessing as mp
    mp.set_start_method("spawn")
    import pandas

    vina_data = "../../databases/Vina/outputs/*"
    vina_buster_inputs = get_vina_bust_inputs(vina_data)

# PoseBusters:
    all_buster_inputs = [
        vina_buster_inputs,
    ]
    with mp.Pool(processes=32) as pool:
        results = []
        for inputs in all_buster_inputs:
            async_results = [
                pool.apply_async(bust, args=item) 
                for item in zip(*inputs)
            ]
            results.append(async_results)
        all_dfs = []
        for async_result_list in results:
            completed_results = []
            for async_result in async_result_list:
                try:
                    # 65 seconds timeout (60 + buffer for file operations)
                    result = async_result.get(timeout=65)
                    if result is not None:
                        completed_results.append(result)
                except mp.TimeoutError:
                    print(f"Task timed out after 65 seconds")
                    continue
                except Exception as e:
                    print(f"Task failed with error: {e}")
                    continue
            if completed_results:
                all_dfs.append(pandas.concat(completed_results))
            else:
                all_dfs.append(pandas.DataFrame())
        vina_bust_df = all_dfs

    torch.save({
        "bust_df": {
            "Vina": vina_bust_df,
        },
    }, "../../databases/results/posebusters_all_df.pt")