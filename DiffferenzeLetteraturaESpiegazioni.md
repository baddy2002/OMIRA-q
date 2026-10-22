
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