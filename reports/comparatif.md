# Comparatif des approches - projet fraude (TP04)

| Approche | F1 test | Temps humain | Temps machine | Cout estime (source, date) | Explicabilite | Export / reversibilite |
|---|---|---|---|---|---|---|
| Logistique (ref.) | 0.7407 | ~5 min | 0.21 s | 0 (poste local) | Maximale | Total |
| Manuel (TP2) | 0.8669 | ~2 h | 3.75 s | 0 (poste local) | Moyenne/Bonne | Total |
| FLAML 60 s | 0.8582 | ~10 min | 60.4 s | 0 (poste local) | Moyenne/Basse | Total |
| Vertex AI AutoML | non mesure | ~15 min | 1 a 2 h | ~3.15 $ / heure de nœud (Google Cloud Pricing, oct. 2026) | Bonne | Faible / Verrouillage (Vendor Lock-in) |
| SageMaker Autopilot | non mesure | ~15 min | 1 h | ~2.50 $ a 5.00 $ (instances ml.m5.xlarge, AWS Pricing, oct. 2026) | Bonne | Faible / Verrouillage (Vendor Lock-in) |
| Azure Automated ML | non mesure | ~15 min | 30 a 60 min | ~1.50 $ a 4.00 $ (VM Standard_DS3_v2, Azure Pricing, oct. 2026) | Bonne | Faible / Verrouillage (Vendor Lock-in) |

## Commentaire (5 lignes)
1. La régression logistique sert de référence immédiate et transparente, idéale quand l'explicabilité réglementaire stricte et la très faible latence priment.
2. Le réglage manuel convient quand un data scientist expert maîtrise la physique métier du problème et affine minutieusement une architecture ciblée.
3. FLAML local est le meilleur compromis efficacité/coût en phase de R&D pour surpasser ou égaler rapidement un modèle manuel sans frais d'infrastructure.
4. Les plateformes cloud managées s'imposent en entreprise pour l'entraînement distribué sur gros volumes, la collaboration d'équipe et le monitoring continu.
5. Toutefois, le choix du cloud implique d'accepter des coûts récurrents élevés et un risque de dépendance technique (vendor lock-in) vis-à-vis du fournisseur.
