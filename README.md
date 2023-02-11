[ Stefano Pigozzi | Traccia #3 | Tema Text Analytics | Big Data Analytics | A.A. 2022/2023 | Unimore ]

# Confronto tra modelli di sentiment analysis per recensioni Amazon

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

## Sinossi

In questo progetto si è realizzato una struttura che permettesse di mettere a confronto diversi modi per effettuare sentiment analysis, e poi si sono realizzati su di essa alcuni modelli di sentiment analysis con caratteristiche diverse per confrontarli.

## Premessa

### Packaging

Il codice dell'attività è incluso come package Python 3.10 compatibile con PEP518.

> **Warning:**
>
> Il progetto non supporta Python 3.11 per via del mancato supporto di Tensorflow a quest'ultimo.

#### Installazione del package

Per installare il package, è necessario eseguire i seguenti comandi dall'interno della directory del progetto:

```console
$ python3.10 -m venv .venv
$ source venv/bin/activate
$ pip install .
```

##### NLTK

NLTK richiede dipendenze aggiuntive per funzionare, che possono essere scaricate eseguendo il seguente comando su console:

```console
$ ./scripts/download-nltk.sh
```

##### Tensorflow

L'accelerazione hardware di Tensorflow richiede che una scheda grafica NVIDIA con supporto a CUDA sia disponibile sul dispositivo, e che gli strumenti di sviluppo di CUDA siano installati sul sistema operativo.

Per indicare a Tensorflow il percorso degli strumenti di sviluppo di CUDA, è necessario impostare la seguente variabile d'ambiente, sostituendo a `/opt/cuda` il percorso in cui gli strumenti sono installati sul dispositivo:

```console
$ export XLA_FLAGS=--xla_gpu_cuda_data_dir\=/opt/cuda
```

Per più informazioni, si suggerisce di consultare la pagina [Install Tensorflow 2](https://www.tensorflow.org/install) della documentazione di Tensorflow.

#### Esecuzione del programma

Per eseguire il programma principale, è possibile eseguire i seguenti comandi dall'interno della directory del progetto:

```console
$ source venv/bin/activate
$ python3.10 -m unimore_bda_6
```

### Dati

Il codice dell'attività richiede la connessione a un server MongoDB 6 contenente la collezione di recensioni Amazon fornita a lezione.

> **Warning:**
>
> La collezione non è inclusa con il repository, in quanto occupa 21 GB!

Si forniscono alcuni script nella cartella `./data/scripts` per facilitare la configurazione e l'esecuzione di quest'ultima.

#### Esecuzione del database

Per eseguire il database MongoDB come processo utente, salvando i dati nella cartella `./data/db`:

```console
$ ./data/scripts/run-db.sh
```

#### Importazione dei dati da JSON

Per importare il dataset `./data/raw/reviewsexport.json` fornito a lezione nel database MongoDB:

```console
$ ./data/scripts/import-db.sh
```

#### Creazione indici

Per creare indici MongoDB potenzialmente utili al funzionamento efficiente del codice:

```console
$ mongosh < ./data/scripts/index-db.js
```

## Struttura per il confronto
### Configurazione ambiente e iperparametri - `.config`
### Recupero dati dal database - `.database`
### Tokenizzatore astratto - `.tokenizer.base`
### Analizzatore astratto - `.analysis.base`
### Logging - `.log`
### Tester - `.__main__`

## Ri-implementazione dell'esercizio con NLTK - `.analysis.nltk_sentiment`
### Wrapping del tokenizzatore di NLTK - `.tokenizer.nltk_word_tokenize`
### Ri-creazione del tokenizer di Christopher Potts - `.tokenizer.potts`
### Problemi di memoria

## Ottimizzazione di memoria
### Caching - `.database.cache` e `.gathering`

## Implementazione di modelli con Tensorflow - `.analysis.tf_text`
### Creazione di tokenizzatori compatibili con Tensorflow - `.tokenizer.plain` e `.tokenizer.lower`
### Creazione di un modello di regressione - `.analysis.tf_text.TensorflowPolarSentimentAnalyzer`
### Creazione di un modello di categorizzazione - `.analysis.tf_text.TensorflowCategorySentimentAnalyzer`
#### Esplosione del gradiente

## Implementazione di tokenizzatori di HuggingFace - `.tokenizer.hugging`

## Confronto dei modelli

## Conclusione
