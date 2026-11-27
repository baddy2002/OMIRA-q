
### Differenza: xgboost leggermente più performante
risultati training un po' in favore di xgboost andando un po' contro la letteratura, in particolare Vishwakarma

#### Spiegazioni

##### Dati

- 1. nci-60 ma diversa release rispetto il paper
- 2. preprocessing per quantum scarta molte molecole, lo stesso paper sopra afferma come xgboost e altri ensemble tree models performino meglio su dataset piccoli rispetto a reti profonda

##### Modelli

- 3. Gatv2 vs chemprop 
- 4. Ho mantenuto 6 cell line massimizzando la diversità delle molecole attive, questo per evitare di prendere linee cellulari con correlazioni troppo forti limitando il task ad un riconoscimento attivo/passivo e replicare lo stesso risultato su tutte le cell line (es se prendevo 6 tumori al polmone), tuttavia questo va in favore di xgboost che addestra un modello per ogni label e quindi può concentrarsi sulle singole cell_line mentre Gatv2 è unica per tutte le label e quindi massimizzando la diversità si complica il task. 
- 5. Ottimizzazione Bayesiana fatta nel paper in oggetto, senza anche in quello xgboost vinceva.


### Differenza: equivarianti migliori

se le molecole non sono girate nello spazio o cose così (abbiamo normalizzato) la EGNN non è che può ottere valori migliori


- 1. rumorosità delle coordinate 3D generate (rdkit con MMFF+ETKDG non perfetto)
- 2. possibile natura del task ??? (gruppi farmacofori sono identificabili anche in 2D)
- 3. Flessibilità conformazionale (le molecole possono assumere diverse conformazioni 3D valide)
- 4. la conformazione 3D è più importante per molecole grandi (Man mano che una molecola cresce, aumenta il numero di legami ruotabili)




Sto lavorando sulla classificazione binaria di attività antitumorale su NCI-60 a partire da grafi molecolari. L'architettura base prevede una GNN per estrarre rappresentazioni molecolari, affiancata da un circuito quantistico variazionale per valutare se l'espressività quantistica possa portare vantaggi predittivi o di compattezza del modello.


Nei primi test end-to-end, far passare l'intero flusso e il gradiente esclusivamente dal circuito quantistico si è rivelato problematico: l'ottimizzazione tramite parameter-shift rule su molti parametri ha costi di calcolo molto elevati su CPU, e soffre di vanishing gradients legati ai barren plateaus. 
Approccio con surrogate gradient in cui fw e bw (warmup) e poi solo bw sono gestiti tramite esso, da problemi perchè anche se il modello surrogato è stato addestrato bene su dati iniziali mano mano che la gatv2 viene addestrata cambia la distribuzione degli input al surrogato durante il train che quindi smette di rappresentare una buona approssimazione del layer quantistico
Al momento ho ottenuto i riscontri migliori trattando il layer quantistico come un blocco di arricchimento non lineare che concatena le rappresentazioni con la parte classica, osservando piccoli incrementi rispetto alla baseline, anche l'implementazione di una feature quantistica aggiuntiva per ogni campione prima del train ha portato qualche miglioramento.

"Per il tirocinio vorrei concentrarmi su due fronti:
    - Introdurre Noise nelle simulazioni
    - Scalare la simulazione sul cluster del CINECA (magari sfruttando GPU e librerie accelerate come cuQuantum) per velocizzare il training e testare strategie di mitigazione dei gradienti/ottimizzazione più spinte;
    - Una volta consolidata l'architettura sul simulatore, validare l'inferenza del modello su hardware quantistico reale (QPU).

model,n_seeds,roc_auc_mean,roc_auc_std,auprc_mean,auprc_std,p@50_mean,p@50_std,ef@50_mean,ef@50_std,p@100_mean,p@100_std,ef@100_mean,ef@100_std
dmpnn,1,0.6852,,0.4833,,0.6533,,2.696,,0.565,,2.3166,
egnn3,5,0.7388,0.0155,0.5471,0.0182,0.7413,0.0248,3.0697,0.1149,0.6243,0.0211,2.5847,0.0962
gatv2,5,0.7387,0.01,0.5454,0.0124,0.736,0.0206,3.0669,0.1071,0.6307,0.0293,2.603,0.1496
gatv2_mf,5,0.7364,0.0091,0.5552,0.0152,0.7566,0.0409,3.1673,0.1856,0.631,0.0289,2.6133,0.1343
gatv2_qf,5,0.7483,0.013,0.5501,0.0118,0.754,0.0174,3.1246,0.0898,0.6123,0.0153,2.5161,0.0768
gatv2_qmf,1,0.7309,,0.5113,,0.6767,,2.755,,0.5383,,2.1671,
xgboost,5,0.7259,0.0032,0.5353,0.0017,0.7407,0.0255,3.0666,0.1185,0.5997,0.0077,2.4527,0.04
