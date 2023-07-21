# fools-gold

cd project/fools-gold/
conda env create -n fools_gold -f environment.yaml
conda activate fools_gold

pip uninstall streamlit
pip install streamlit==1.24.1
pip install boto3==1.26.151
pip install langchain==0.0.232
pip install chromadb-client

conda deactivate
conda remove --name fools_gold --all