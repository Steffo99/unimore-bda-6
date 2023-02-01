[ Stefano Pigozzi | Tema Text Analytics | Big Data Analytics | A.A. 2022/2023 | Unimore ]

# WIP

> ### Sentiment analysis su recensioni Amazon
> 
> Modificare l’esercizio di sentiment analysis sulle review Amazon svolto in classe e verificare l’efficacia del metodo effettuando queste varianti: 
> 
> 1. Utilizzare come tokenizer il “sentiment tokenizer” di Christopher Potts (link disponibile nelle slide del corso);
> 2. Modificare il dataset recuperando anche recensioni a 2, 3 e 4 stelle ed effettuare una classificazione a più classi (es. 5 classi di sentiment corrispondenti al numero di stelle delle recensioni). 
>
> Effettuare quindi un confronto di efficacia tra queste varianti e la versione originale vista in classe.
> 
> Valutare anche l’inclusione di altre feature estratte dai dati, con l’obiettivo di aumentare l’efficacia.
> 
> * E’ necessario effettuare tutti i test su un numero significativo di run (es., almeno 50), scegliendo ogni volta in maniera casuale la composizione di test-set e training-set a partire dall’insieme di post estratti (è possibile utilizzare le feature automatiche di cross validation viste per scikit-learn)
> * E’ possibile (e gradito) estendere in ampiezza la propria analisi:
>    * utilizzare e confrontare una o più delle librerie di ML viste a lezione (NLTK/scikitlearn/XGBoost/Tensorflow) (NOTA: per le tracce 2 e 3 è necessario sperimentare anche almeno una libreria diversa da NLTK)
>    * utilizzare e confrontare diversi classificatori tra quelli offerti (es. quelli citati a lezione in scikit-learn) e una o più delle tecniche citate/viste a lezione (es. codifica del testo tramite TF-IDF, word embeddings per tensorflow, hyper-parameter tuning per scikit-learn, tecniche specifiche per sent. analysis, …)
>    * utilizzare librerie per l’elaborazione del testo alternative (es. SpaCy https://spacy.io/ ) per estrarre feature aggiuntive, valutandone l’effetto sul modello
>    * in generale: studiare, riassumere brevemente e applicare eventuali altre tecniche o strumenti ritenuti utili all’obiettivo (cioè, migliorare l’efficacia del modello proposto).
>
> Consegna: PDF commentato con discussione e codice Python (includere dati e codice anche in un file .txt per facilitarne il testing)
> 
> Per quanto riguarda il codice Python, è possibile (e gradito) produrre e consegnare un notebook jupyter .ipynb
> (https://jupyter.org/) invece di codice .py e relativi commenti separati su PDF (per comodità di consultazione,
> consegnare comunque anche una stampa PDF del notebook oltre al notebook stesso).
