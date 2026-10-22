Genera il dataset 

cd OMIRA-q
python -m app.scripts.01_etl_and_visualize
python -m app.scripts.02_preprocess_and_visualize
for s in 0 42 123 6 36; do python -m app.scripts.03_classical_train_and_compare --model xgboost --seed $s; done
for s in 0 42 123 6 36; do python -m app.scripts.03_classical_train_and_compare --model gatv2 --seed $s; done
python -m app.scripts.04_aggregate_seeds




