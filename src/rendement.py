import mlflow
import matplotlib
matplotlib.use('Agg')  # pas de fenetre graphique
import matplotlib.pyplot as plt

mlflow.set_tracking_uri('http://127.0.0.1:5000')
runs = mlflow.search_runs(experiment_names=['tp4-automl'])  # du plus recent au plus ancien

auto = runs[runs['params.outil'].isin(['flaml', 'sklearn-maison'])].copy()
auto['budget'] = auto['params.budget_s'].astype(int)
auto = auto.drop_duplicates(['params.outil', 'budget']).sort_values('budget')  # dernier run par budget

refs = runs[runs['params.outil'] == 'scikit-learn'].drop_duplicates('tags.mlflow.runName')

plt.figure(figsize=(7, 4))
for outil, g in auto.groupby('params.outil'):
    plt.plot(g['budget'], g['metrics.f1'], 'o-', label=f'AutoML ({outil})')
    for b, f in zip(g['budget'], g['metrics.f1']):
        print(f'{outil:15s} budget {b:4d} s -> F1 {f:.4f}')

for _, r in refs.iterrows():
    nom, f = r['tags.mlflow.runName'], r['metrics.f1']
    plt.axhline(f, linestyle='--', color='gray', linewidth=1)
    plt.text(auto['budget'].min(), f + 0.003, f'{nom} ({f:.4f})', fontsize=8, color='gray')
    print(f'{nom:22s} F1 {f:.4f}')

plt.xscale('log')
budgets = sorted(auto['budget'].unique())
plt.xticks(budgets, [str(b) for b in budgets])
plt.minorticks_off()
plt.xlabel('budget de calcul (secondes, echelle log)')
plt.ylabel('F1 sur le jeu de test')
plt.title('Rendement du budget de calcul')
plt.legend(loc='lower right')
plt.savefig('reports/rendement.png', dpi=120, bbox_inches='tight')
print('graphique ecrit : reports/rendement.png')
