import subprocess

files = [
    'fig1_spec_intro',
    'fig2_GQ_CC_tre08',
    'figSM1_FP_effects',
    'fig34_Kneser_compare',
    'fig56_KPM_split_nospike',
    'fig7_KPM_spike',
    'fig8_heat_capacity',
    'fig9_partition',
]

for file in files:
    subprocess.run(f'jupyter nbconvert --execute --to notebook --inplace --allow-errors --ExecutePreprocessor.timeout=-1 {file}.ipynb',shell=True)

