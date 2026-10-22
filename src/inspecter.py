import sys
import json
from collections import Counter

fichier = sys.argv[1] if len(sys.argv) > 1 else 'reports/flaml_60s.log'
essais = []
for ligne in open(fichier, encoding='utf-8'):
    ligne = ligne.strip()
    if ligne.startswith('{'):
        e = json.loads(ligne)
        if 'learner' in e and e.get('validation_loss') is not None:
            essais.append(e)  # une ligne = un essai

print(len(essais), 'essais journalises dans', fichier)
print('essais par famille :', dict(Counter(e['learner'] for e in essais)))

meilleurs = {}  # meilleur essai de chaque famille
for e in essais:
    f = e['learner']
    if f not in meilleurs or e['validation_loss'] < meilleurs[f]['validation_loss']:
        meilleurs[f] = e

print('\nMeilleur F1 de validation par famille (F1 = 1 - perte) :')
for f, e in sorted(meilleurs.items(), key=lambda kv: kv[1]['validation_loss']):
    print(f" {f:15s} F1 val = {1 - e['validation_loss']:.4f} (trouve apres {e.get('wall_clock_time', 0):.1f} s)")

gagnant = min(essais, key=lambda e: e['validation_loss'])
print('\nEssai gagnant :', gagnant['learner'], gagnant.get('config'))
