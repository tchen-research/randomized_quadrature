import subprocess

files = [
    'fig0_spec_intro',
    'fig1_GQ_CC_tre08',
    'fig2_FP_effects',
    'fig3_Kneser_compare',
    'fig4_KPM_split_nospike',
    'fig6_KPM_spike',
    'fig7_heat_capacity',
]

for file in files:
    subprocess.run(f'jupyter nbconvert --execute --to notebook --inplace --allow-errors --ExecutePreprocessor.timeout=-1 {file}.ipynb',shell=True)

