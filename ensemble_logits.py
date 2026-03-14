
import os
import glob
import pandas as pd
import torch

def load_logits_csv(filepath):
    df = pd.read_csv(filepath, sep='\t')
    df = df.set_index("filename")
    return df

def ensemble_logits(csv_files, output_csv):
    all_logits = []
    filenames = None

    for path in csv_files:
        df = load_logits_csv(path)
        if filenames is None:
            filenames = df.index
        else:
            filenames = filenames.intersection(df.index)
        all_logits.append(df)

    # 对 filename 做交集匹配
    filenames = sorted(filenames)
    aligned_logits = [df.loc[filenames].drop(columns=["scene_label"]) for df in all_logits]
    avg_logits = sum(df.values for df in aligned_logits) / len(aligned_logits)

    # 构建输出 DataFrame
    class_names = aligned_logits[0].columns
    ensemble_df = pd.DataFrame(avg_logits, columns=class_names)
    ensemble_df.insert(0, "filename", filenames)
    ensemble_df.insert(1, "scene_label", [class_names[i] for i in avg_logits.argmax(axis=1)])

    # 保存
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    ensemble_df.to_csv(output_csv, sep='\t', index=False)
    print(f"[Done] Ensemble logits saved to: {output_csv}")

if __name__ == "__main__":
    # 设置路径
    input_dir = "predictions_2T"  # 所有 teacher logits 所在文件夹
    output_csv = "ensemble/ensemble_logits_2T_1C1P.csv"

    # 自动查找子目录下的所有 logits.csv
    csv_files = glob.glob(os.path.join(input_dir, "*", "*.csv"))
    print("[Info] Found logits files:", csv_files)

    ensemble_logits(csv_files, output_csv)
