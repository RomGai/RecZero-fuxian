import pandas as pd, pathlib

run_dir = pathlib.Path("/home/huangyanwen.hyw/code_linlin/Logic-RL-rating/wandb/offline-run-20250726_131550-3i5298si")
df = pd.read_json(run_dir / "files" / "metrics.json", lines=True)
print(df.columns)                 # 查看有哪些指标
df[['step', 'train/loss', 'eval/acc']].plot(kind='line')