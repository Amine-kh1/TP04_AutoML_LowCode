# Plan B : une petite recherche AutoML ecrite avec scikit-learn seul.
# Meme principe que FLAML : tant que le budget n'est pas epuise,
# proposer un candidat (famille + reglages), l'evaluer, garder le meilleur.
import sys, json, time, random
from pathlib import Path
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                             HistGradientBoostingClassifier)

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 60 # secondes
rng = random.Random(42)

ESPACE = { # famille -> fonction qui tire une configuration au hasard
    'logistique': lambda: {'C': 10 ** rng.uniform(-3, 2)},
    'rf': lambda: {'n_estimators': rng.choice([100, 200, 400]),
                   'max_depth': rng.choice([6, 10, 16, None]),
                   'min_samples_leaf': rng.choice([1, 2, 5])},
    'extra_tree': lambda: {'n_estimators': rng.choice([100, 200, 400]),
                           'max_depth': rng.choice([6, 10, 16, None])},
    'hist_gb': lambda: {'learning_rate': 10 ** rng.uniform(-2, -0.5),
                        'max_leaf_nodes': rng.choice([15, 31, 63]),
                        'max_iter': rng.choice([100, 200, 400])},
}

CLASSES = {
    'logistique': lambda c: LogisticRegression(max_iter=2000, **c),
    'rf': lambda c: RandomForestClassifier(random_state=42, n_jobs=-1, **c),
    'extra_tree': lambda c: ExtraTreesClassifier(random_state=42, n_jobs=-1, **c),
    'hist_gb': lambda c: HistGradientBoostingClassifier(random_state=42, **c),
}

df = pd.read_csv('data/transactions.csv')
X, y = df.drop(columns='fraude'), df['fraude']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

Path('reports').mkdir(exist_ok=True)
journal = open(f'reports/maison_{budget}s.log', 'w', encoding='utf-8')
t0, n, meilleur = time.time(), 0, None
familles = list(ESPACE)
while time.time() - t0 < budget:
    famille = familles[n % len(familles)] if n < len(familles) else rng.choice(familles)
    config = ESPACE[famille]()
    score = cross_val_score(CLASSES[famille](config), Xtr, ytr, cv=3, scoring='f1').mean()
    n += 1
    journal.write(json.dumps({'record_id': n, 'learner': famille, 'config': config,
                              'validation_loss': 1 - score,
                              'wall_clock_time': round(time.time() - t0, 2)}) + '\n')
    if meilleur is None or score > meilleur[0]:
        meilleur = (score, famille, config)

journal.close()
duree = round(time.time() - t0, 1)
score_val, famille, config = meilleur
modele = CLASSES[famille](config).fit(Xtr, ytr) # reentrainement final
f1 = round(f1_score(yte, modele.predict(Xte)), 4)
print(f'budget {budget} s | {n} essais | famille retenue : {famille} | F1 test = {f1}')
print('meilleure config :', config)

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('tp4-automl')
with mlflow.start_run(run_name=f'maison-{budget}s'):
    mlflow.log_params({'outil': 'sklearn-maison', 'budget_s': budget, 'famille': famille})
    mlflow.log_metrics({'f1': f1, 'f1_validation': round(score_val, 4),
                        'duree_s': duree, 'nb_essais': n})
    mlflow.log_artifact(f'reports/maison_{budget}s.log')
