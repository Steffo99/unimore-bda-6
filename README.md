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

### Codice e packaging

Il codice dell'attività è incluso come package Python 3.10 compatibile con PEP518.

> **Note**
>
> In questo documento sono riportate parti del codice: in esse, è stato rimosso il codice superfluo come comandi di logging, docstring e commenti, in modo da accorciare la relazione e per mantenere l'attenzione sull'argomento della rispettiva sezione.
>
> Nel titolo di ciascuna sezione è evidenziato il file da cui gli spezzoni di codice provengono: se si necessitano sapere più dettagli sul funzionamento di esso, si consiglia di andare a vedere i file sorgente allegati, che contengono la documentazione necessaria.

> **Warning**
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

> **Warning**
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

## Costruzione di una struttura per il confronto

Al fine di effettuare i confronti richiesti dalla consegna dell'attività, si è deciso di realizzare un package Python che permettesse di confrontare vari modelli di Sentiment Analysis tra loro, con tokenizer, training set e test set diversi tra loro.

Il package, chiamato `unimore_bda_6`, è composto da vari moduli, ciascuno descritto nelle seguenti sezioni.

### Configurazione ambiente e iperparametri - `.config`

Il primo modulo, `unimore_bda_6.config`, definisce le variabili configurabili del package usando [`cfig`], e, se eseguito, mostra all'utente un'interfaccia command-line che le descrive e ne mostra i valori attuali.

Viene prima creato un oggetto [`cfig.Configuration`], che opera come contenitore per le variabili configurabili:

```python
import cfig
config = cfig.Configuration()
```

In seguito, per ogni variabile configurabile viene definita una funzione, che elabora il valore ottenuto dalle variabili di ambiente del contesto in cui il programma è eseguito, convertendolo in un formato più facilmente utilizzabile dal programma.

Si fornisce un esempio di una di queste funzioni, che definisce la variabile per configurare la dimensione del training set:

```python
@config.optional()
def TRAINING_SET_SIZE(val: str | None) -> int:
    """
    The number of reviews from each category to fetch for the training dataset.
    Defaults to `4000`.
    """
    if val is None:
        return 4000
    try:
        return int(val)
    except ValueError:
        raise cfig.InvalidValueError("Not an int.")
```

> In gergo del machine learning / deep learning, queste variabili sono dette iperparametri, perchè configurano la creazione del modello, e non vengono configurati dall'addestramento del modello stesso!

Infine, si aggiunge una chiamata al metodo `cli()` della configurazione, eseguita solo se il modulo viene eseguito come main, che mostra all'utente l'interfaccia precedentemente menzionata:

```python
if __name__ == "__main__":
    config.cli()
```

L'esecuzione del modulo `unimore_bda_6.config`, senza variabili d'ambiente definite, dà quindi il seguente output:

```console
$ python -m unimore_bda_6.config
===== Configuration =====
...
TRAINING_SET_SIZE         = 4000
The number of reviews from each category to fetch for the training dataset.
Defaults to `4000`.
...
===== End =====
```

### Recupero dati dal database - `.database`

Il modulo `unimore_bda_6.database` si occupa della connessione al database [MongoDB] e la collezione contenente il dataset di partenza, del recupero dei documenti in modo bilanciato, della conversione di essi in un formato più facilmente leggibile da Python, e della creazione di cache su disco per permettere alle librerie che lo supportano di non caricare l'intero dataset in memoria durante l'addestramento di un modello.

#### Connessione al database - `.database.connection`

Il modulo `unimore_bda_6.database.connection` si occupa della conessione (e disconnessione) al database utilizzando il package [`pymongo`].

Definisce un context manager che effettua automaticamente la disconnessione dal database una volta usciti dal suo scope:

```python
@contextlib.contextmanager
def mongo_client_from_config() -> t.ContextManager[pymongo.MongoClient]:
    client: pymongo.MongoClient = pymongo.MongoClient(
        host=MONGO_HOST.__wrapped__,
        port=MONGO_PORT.__wrapped__,
    )
    yield client
    client.close()
```

Esso sarà poi utilizzato in questo modo:

```python
with mongo_client_from_config as client:
    ...
```

#### Recupero della collezione - `.database.collections`

Il modulo `unimore_bda_6.database.collection` si occupa di recuperare la collezione `reviews` dal database MongoDB:

```python
def reviews_collection(db: pymongo.MongoClient) -> pymongo.collection.Collection[MongoReview]:
    collection: pymongo.collection.Collection[MongoReview] = db.reviews.reviews
    return collection
```

#### Contenitori di dati - `.database.datatypes`

Il modulo `unimore_bda_6.database.datatypes` contiene contenitori ottimizzati (attraverso l'attributo magico [`__slots__`]) per i dati recuperati dal database, che possono essere riassunti con:

```python
@dataclasses.dataclass
class TextReview:
    text: str
    rating: float

@dataclasses.dataclass
class TokenizedReview:
    tokens: list[str]
    rating: float
```

#### Query su MongoDB - `.database.queries`

Il modulo `unimore_bda_6.database.queries` contiene alcune query pre-costruite utili per operare sulla collezione `reviews`.

##### Working set

Essendo il dataset completo composto da 23 milioni, 831 mila e 908 documenti (23_831_908), effettuare campionamenti su di esso in fase di sviluppo risulterebbe eccessivamente lento e dispendioso, pertanto in ogni query il dataset viene rimpicciolito a un *working set* attraverso l'uso del seguente aggregation pipeline stage, dove `WORKING_SET_SIZE` è sostituito dal suo corrispondente valore nella configurazione (di default 1_000_000):

```javascript
{"$limit": WORKING_SET_SIZE},
```

##### Dataset con solo recensioni 1* e 5* - `sample_reviews_polar`

Per recuperare un dataset bilanciato di recensioni 1* e 5*, viene utilizzata la seguente funzione:

```python
def sample_reviews_polar(collection: pymongo.collection.Collection, amount: int) -> t.Iterator[TextReview]:
    category_amount = amount // 2

    cursor = collection.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$match": {"overall": 1.0}},
        {"$sample": {"size": category_amount}},
        {"$unionWith": {
            "coll": collection.name,
            "pipeline": [
                {"$limit": WORKING_SET_SIZE.__wrapped__},
                {"$match": {"overall": 5.0}},
                {"$sample": {"size": category_amount}},
            ],
        }},
        {"$addFields": {
            "sortKey": {"$rand": {}},
        }},
        {"$sort": {
            "sortKey": 1,
        }}
    ])

    cursor = map(TextReview.from_mongoreview, cursor)

    return cursor
```

L'aggregazione eseguita non è altro che l'unione dei risultati delle seguenti due aggregazioni, i cui risultati vengono poi mescolati attraverso l'ordinamento su un campo contenente il risultato dell'operatore [`$rand`]:

```javascript
db.reviews.aggregate([
    {"$limit": WORKING_SET_SIZE},
    {"$match": {"overall": 1.0}},
    {"$sample": {"size": amount / 2}},
])
// unita a
db.reviews.aggregate([
    {"$limit": WORKING_SET_SIZE},
    {"$match": {"overall": 5.0}},
    {"$sample": {"size": amount / 2}},
])
// e poi mescolate
```

##### Dataset bilanciato con recensioni 1*, 2*, 3*, 4* e 5*

Lo stesso procedimento viene usato per ottenere un dataset bilanciato di recensioni con ogni numero possibile di stelle:

```python
def sample_reviews_varied(collection: pymongo.collection.Collection, amount: int) -> t.Iterator[TextReview]:
    category_amount = amount // 5

    cursor = collection.aggregate([
        {"$limit": WORKING_SET_SIZE.__wrapped__},
        {"$match": {"overall": 1.0}},
        {"$sample": {"size": category_amount}},
        {"$unionWith": {
            "coll": collection.name,
            "pipeline": [
                {"$limit": WORKING_SET_SIZE.__wrapped__},
                {"$match": {"overall": 2.0}},
                {"$sample": {"size": category_amount}},
                {"$unionWith": {
                    "coll": collection.name,
                    "pipeline": [
                        {"$limit": WORKING_SET_SIZE.__wrapped__},
                        {"$match": {"overall": 3.0}},
                        {"$sample": {"size": category_amount}},
                        {"$unionWith": {
                            "coll": collection.name,
                            "pipeline": [
                                {"$limit": WORKING_SET_SIZE.__wrapped__},
                                {"$match": {"overall": 4.0}},
                                {"$sample": {"size": category_amount}},
                                {"$unionWith": {
                                    "coll": collection.name,
                                    "pipeline": [
                                        {"$limit": WORKING_SET_SIZE.__wrapped__},
                                        {"$match": {"overall": 5.0}},
                                        {"$sample": {"size": category_amount}},
                                    ],
                                }}
                            ],
                        }}
                    ],
                }}
            ],
        }},
        {"$addFields": {
            "sortKey": {"$rand": {}},
        }},
        {"$sort": {
            "sortKey": 1,
        }}
    ])

    cursor = map(TextReview.from_mongoreview, cursor)

    return cursor
```

### Tokenizzatore astratto - `.tokenizer.base` e `.tokenizer.plain`

Si è realizzata una classe astratta che rappresentasse un tokenizer qualcunque, in modo da avere la stessa interfaccia a livello di codice indipendentemente dal package di tokenizzazione utilizzato:

```python
class BaseTokenizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def tokenize(self, text: str) -> t.Iterator[str]:
        "Convert a text `str` into another `str` containing a series of whitespace-separated tokens."
        raise NotImplementedError()

    def tokenize_review(self, review: TextReview) -> TokenizedReview:
        "Apply `.tokenize` to the text of a `TextReview`, converting it in a `TokenizedReview`."
        tokens = self.tokenize(review.text)
        return TokenizedReview(rating=review.rating, tokens=tokens)
```

Si sono poi realizzate due classi di esempio che implementassero i metodi astratti della precedente: `PlainTokenizer` e `LowerTokenizer`, che semplicemente separano il testo in tokens attraverso la funzione builtin [`str.split`] di Python, rispettivamente mantenendo e rimuovendo la capitalizzazione del testo.

```python
class PlainTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> t.Iterator[str]:
        tokens = text.split()
        return tokens

class LowercaseTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> t.Iterator[str]:
        text = text.lower()
        tokens = text.split()
        return tokens
```

### Analizzatore astratto - `.analysis.base`

Allo stesso modo, si è realizzato una classe astratta per tutti i modelli di Sentiment Analysis:

```python
class BaseSentimentAnalyzer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, training_dataset_func: CachedDatasetFunc, validation_dataset_func: CachedDatasetFunc) -> None:
        "Train the analyzer with the given training and validation datasets."
        raise NotImplementedError()

    @abc.abstractmethod
    def use(self, text: str) -> float:
        "Run the model on the given input, and return the predicted rating."
        raise NotImplementedError()

    def evaluate(self, evaluation_dataset_func: CachedDatasetFunc) -> EvaluationResults:
        """
        Perform a model evaluation by calling repeatedly `.use` on every text of the test dataset and by comparing its resulting category with the expected category.
        """
        er = EvaluationResults()
        for review in evaluation_dataset_func():
            er.add(expected=review.rating, predicted=self.use(review.text))
        return er
```

Si può notare che il metodo `evaluate` inserisce i risultati di ciascuna predizione effettuata in un oggetto di tipo `EvaluationResults`.

Esso tiene traccia della matrice di confusione per la valutazione, e da essa è in grado di ricavarne i valori di richiamo e precisione per ciascuna categoria implementata dal modello; inoltre, calcola l'errore medio assoluto e quadrato tra previsioni e valori effettivi:

```python
class EvaluationResults:
    def __init__(self):
        self.confusion_matrix: dict[float, dict[float, int]] = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        "Confusion matrix of the evaluation. First key is the expected rating, second key is the output label."

        self.absolute_error_total: float = 0.0
        "Sum of the absolute errors committed in the evaluation."

        self.squared_error_total: float = 0.0
        "Sum of the squared errors committed in the evaluation."

    def keys(self) -> set[float]:
        "Return all processed categories."
        keys: set[float] = set()
        for expected, value in self.confusion_matrix.items():
            keys.add(expected)
            for predicted, _ in value.items():
                keys.add(predicted)
        return keys

    def evaluated_count(self) -> int:
        "Return the total number of evaluated reviews."
        total: int = 0
        for row in self.confusion_matrix.values():
            for el in row.values():
                total += el
        return total

    def perfect_count(self) -> int:
        """
        Return the total number of perfect reviews.
        """
        total: int = 0
        for key in self.keys():
            total += self.confusion_matrix[key][key]
        return total

    def recall_count(self, rating: float) -> int:
        "Return the number of reviews processed with the given rating."
        total: int = 0
        for el in self.confusion_matrix[rating].values():
            total += el
        return total

    def precision_count(self, rating: float) -> int:
        "Return the number of reviews for which the model returned the given rating."
        total: int = 0
        for col in self.confusion_matrix.values():
            total += col[rating]
        return total

    def recall(self, rating: float) -> float:
        "Return the recall for a given rating."
        try:
            return self.confusion_matrix[rating][rating] / self.recall_count(rating)
        except ZeroDivisionError:
            return float("inf")

    def precision(self, rating: float) -> float:
        "Return the precision for a given rating."
        try:
            return self.confusion_matrix[rating][rating] / self.precision_count(rating)
        except ZeroDivisionError:
            return float("inf")

    def add(self, expected: float, predicted: float) -> None:
        "Count a new prediction."
        self.confusion_matrix[expected][predicted] += 1
```

Si è inoltre realizzata un'implementazione di esempio della classe astratta, `ThreeCheat`, che "prevede" che tutte le recensioni siano di 3.0*, in modo da verificare facilmente la correttezza della precedente classe:

```python
class ThreeCheat(BaseSentimentAnalyzer):
    def train(self, training_dataset_func: CachedDatasetFunc, validation_dataset_func: CachedDatasetFunc) -> None:
        pass

    def use(self, text: str) -> float:
        return 3.0
```

### Logging - `.log`

Si è configurato il modulo [`logging`] di Python affinchè esso scrivesse report sull'esecuzione:

- nello stream stderr della console, in formato colorato e user-friendly
- sul file `./data/logs/last_run.tsv`, in formato machine-readable

Il livello di logging viene regolato attraverso la costante magica [`__debug__`] di Python, il cui valore cambia in base alla presenza dell'opzione di ottimizzazione [`-O`] dell'interprete Python; senza quest'ultima, i log stampati su console saranno molto più dettagliati.

### Tester - `.__main__`

Infine, si è preparato un tester che effettuasse ripetute valutazioni di efficacia per ogni combinazione di funzione di campionamento, tokenizzatore, e modello di Sentiment Analysis, con una struttura simile alla seguente:

```python
# Pseudo-codice non corrispondente al main finale
if __name__ == "__main__":
    for sample_func in [sample_reviews_polar, sample_reviews_varied]:
        for SentimentAnalyzer in [ThreeCheat, ...]:
            for Tokenizer in [PlainTokenizer, LowercaseTokenizer, ...]:
                for run in range(TARGET_RUNS):
                    model = SentimentAnalyzer(tokenizer=Tokenizer())
                    model.train(training_set=sample_func(amount=TRAINING_SET_SIZE), validation_set_func=sample_func(amount=VALIDATION_SET_SIZE))
                    model.evaluate(evaluation_set_func=sample_func(amount=EVALUATION_SET_SIZE))
```

Le valutazioni di efficacia vengono effettuate fino al raggiungimento di `TARGET_RUNS` addestramenti e valutazioni riuscite, o fino al raggiungimento di `MAXIMUM_RUNS` valutazioni totali (come descritto più avanti, l'addestramento di alcuni modelli potrebbe fallire e dover essere ripetuto).

## Ri-implementazione dell'esercizio con NLTK

Come prima cosa, si è ricreato l'esempio di sentiment analysis realizzato a lezione all'interno del package `unimore_bda_6`.

### Wrapping del tokenizzatore di NLTK - `.tokenizer.nltk_word_tokenize`

Si è creata una nuova sottoclasse di `BaseTokenizer`, `NLTKWordTokenizer`, che usa la tokenizzazione inclusa con NLTK.

Per separare le parole in token, essa chiama [`nltk.word_tokenize`], funzione built-in che sfrutta i tokenizer [Punkt] e [Treebank] per dividere rispettivamente in frasi e parole la stringa passata come input.

La lista di tokens viene poi passata a [`nltk.sentiment.util.mark_negation`], che aggiunge il suffisso `_NEG` a tutti i token che si trovano tra una negazione e un segno di punteggiatura, in modo che la loro semantica venga preservata anche all'interno di un contesto *bag of words*, in cui le posizioni dei token vengono ignorate.

(È considerato negazione qualsiasi token che finisce con `n't`, oppure uno dei seguenti token: `never`, `no`, `nothing`, `nowhere`, `noone`, `none`, `not`, `havent`, `hasnt`, `hadnt`, `cant`, `couldnt`, `shouldnt`, `wont`, `wouldnt`, `dont`, `doesnt`, `didnt`, `isnt`, `arent`, `aint`.)

```python
class NLTKWordTokenizer(BaseTokenizer):
    def tokenize(self, text: str) -> t.Iterator[str]:
        tokens = nltk.word_tokenize(text)
        nltk.sentiment.util.mark_negation(tokens, shallow=True)
        return tokens
```

### Costruzione del modello - `.analysis.nltk_sentiment`

Si è creata anche una sottoclasse di `BaseSentimentAnalyzer`, `NLTKSentimentAnalyzer`, che utilizza per la classificazione un modello di tipo [`nltk.sentiment.SentimentAnalyzer`].

```python
class NLTKSentimentAnalyzer(BaseSentimentAnalyzer):
    def __init__(self, *, tokenizer: BaseTokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model: nltk.sentiment.SentimentAnalyzer = nltk.sentiment.SentimentAnalyzer()
        self.trained: bool = False

    ...
```

Esattamente come il modello realizzato a lezione, in fase di addestramento esso:

1. Prende il testo di ogni recensione del training set
2. Lo converte in una lista di token
3. Conta le occorrenze totali di ogni token della precedente lista per determinare quelli che compaiono in almeno 4 recensioni diverse
4. Utilizza questi token frequenti per identificare le caratteristiche ("features") da usare per effettuare la classificazione

Successivamente:

5. Identifica la presenza delle caratteristiche in ciascun elemento del training set
6. Addestra un classificatore Bayesiano semplice ("naive Bayes") perchè determini la probabilità che data una certa feature, una recensione abbia un certo numero di stelle

```python
    ...

    def _add_feature_unigrams(self, dataset: t.Iterator[TokenizedReview]) -> None:
        "Register the `nltk.sentiment.util.extract_unigram_feats` feature extrator on the model."
        tokenbags = map(lambda r: r.tokens, dataset)
        all_words = self.model.all_words(tokenbags, labeled=False)
        unigrams = self.model.unigram_word_feats(words=all_words, min_freq=4)
        self.model.add_feat_extractor(nltk.sentiment.util.extract_unigram_feats, unigrams=unigrams)

    def _add_feature_extractors(self, dataset: t.Iterator[TextReview]):
        "Register new feature extractors on the `.model`."
        dataset: t.Iterator[TokenizedReview] = map(self.tokenizer.tokenize_review, dataset)
        self._add_feature_unigrams(dataset)

    def __extract_features(self, review: TextReview) -> tuple[Features, str]:
        "Convert a TextReview to a (Features, str) tuple. Does not use `SentimentAnalyzer.apply_features` due to unexpected behaviour when using iterators."
        review: TokenizedReview = self.tokenizer.tokenize_review(review)
        return self.model.extract_features(review.tokens), str(review.rating)

    def train(self, training_dataset_func: CachedDatasetFunc, validation_dataset_func: CachedDatasetFunc) -> None:
        if self.trained:
            raise AlreadyTrainedError()
        self._add_feature_extractors(training_dataset_func())
        featureset: t.Iterator[tuple[Features, str]] = map(self.__extract_features, training_dataset_func())
        self.model.classifier = nltk.classify.NaiveBayesClassifier.train(featureset)
        self.trained = True

    ...
```

Infine, implementa la funzione `use`, che:

1. tokenizza la stringa ottenuta in input
2. utilizza il modello precedentemente addestrato per determinare la categoria ("rating") di appartenenza

```python
    ...

    def use(self, text: str) -> float:
        if not self.trained:
            raise NotTrainedError()
        tokens = self.tokenizer.tokenize(text)
        rating = self.model.classify(instance=tokens)
        rating = float(rating)
        return rating
```

#### Problemi di RAM

L'approccio utilizzato da [`nltk.sentiment.SentimentAnalyzer`] si è rivelato problematico, in quanto non in grado di scalare a dimensioni molto grandi di training set: i suoi metodi non gestiscono correttamente gli iteratori, meccanismo attraverso il quale Python può realizzare lazy-loading di dati, e richiedono invece che l'intero training set sia caricato contemporaneamente in memoria in una [`list`].

Per permetterne l'esecuzione su computer con 16 GB di RAM, si è deciso di impostare la dimensione predefinita del training set a `4000` documenti.

### Ri-creazione del tokenizer di Christopher Potts - `.tokenizer.potts`

> 1. Utilizzare come tokenizer il “sentiment tokenizer” di Christopher Potts (link disponibile nelle slide del corso);

Per realizzare il punto 1 della consegna, si sono creati due nuovi tokenizer, `PottsTokenizer` e `PottsTokenizerWithNegation`, che implementano il [tokenizer di Christopher Potts] rispettivamente senza marcare e marcando le negazioni sui token attraverso [`ntlk.sentiment.util.mark_negation`].

Essendo il tokenizer originale scritto per Python 2, e non direttamente immediatamente compatibile con `BaseTokenizer`, si è scelto di studiare il codice originale e ricrearlo in un formato più adatto a questo progetto.

Prima di effettuare la tokenizzazione, il tokenizer normalizza l'input:

1. convertendo tutte le entità HTML come `&lt;` nel loro corrispondente unicode `<`
2. convertendo il carattere `&` nel token `and`

Il tokenizer effettua poi la tokenizzazione usando espressioni regolari definite in `words_re_string` per catturare token di diversi tipi, in ordine:

* emoticon testuali `:)`
* numeri di telefono statunitensi `+1 123 456 7890`
* tag HTML `<b>`
* username stile Twitter `@steffo`
* hashtag stile Twitter `#Big_Data_Analytics`
* parole con apostrofi `i'm`
* numeri `-9000`
* parole senza apostrofi `data`
* ellissi `. . .`
* gruppi di caratteri non-whitespace `🇮🇹`

Dopo aver tokenizzato, il tokenizer processa il risultato:

1. convertendo il testo a lowercase, facendo attenzione però a non cambiare la capitalizzazione delle emoticon per non cambiare il loro significato (`:D` è diverso da `:d`)

Il codice riassunto del tokenizer è dunque:

```python
class PottsTokenizer(BaseTokenizer):
    emoticon_re_string = r"""[<>]?[:;=8][\-o*']?[)\](\[dDpP/:}{@|\\]"""
    emoticon_re = re.compile(emoticon_re_string)

    words_re_string = "(" + "|".join([
        emoticon_re_string,
        r"""(?:[+]?[01][\s.-]*)?(?:[(]?\d{3}[\s.)-]*)?\d{3}[\-\s.]*\d{4}""",
        r"""<[^>]+>""",
        r"""@[\w_]+""",
        r"""#+[\w_]+[\w'_-]*[\w_]+""",
        r"""[a-z][a-z'_-]+[a-z]""",
        r"""[+-]?\d+(?:[,/.:-]\d+)?""",
        r"""[\w_]+""",
        r"""[.](?:\s*[.])+""",
        r"""\S+""",
    ]) + ")"

    words_re = re.compile(words_re_string, re.I)

    digit_re_string = r"&#\d+;"
    digit_re = re.compile(digit_re_string)

    alpha_re_string = r"&\w+;"
    alpha_re = re.compile(alpha_re_string)

    amp = "&amp;"

    @classmethod
    def html_entities_to_chr(cls, s: str) -> str:
        "Internal metod that seeks to replace all the HTML entities in s with their corresponding characters."
        # First the digits:
        ents = set(cls.digit_re.findall(s))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    s = s.replace(ent, chr(entnum))
                except (ValueError, KeyError):
                    pass
        # Now the alpha versions:
        ents = set(cls.alpha_re.findall(s))
        ents = filter((lambda x: x != cls.amp), ents)
        for ent in ents:
            entname = ent[1:-1]
            try:
                s = s.replace(ent, chr(html.entities.name2codepoint[entname]))
            except (ValueError, KeyError):
                pass
            s = s.replace(cls.amp, " and ")
        return s

    @classmethod
    def lower_but_preserve_emoticons(cls, word):
        "Internal method which lowercases the word if it does not match `.emoticon_re`."
        if cls.emoticon_re.search(word):
            return word
        else:
            return word.lower()

    def tokenize(self, text: str) -> t.Iterator[str]:
        text = self.html_entities_to_chr(text)
        tokens = self.words_re.findall(text)
        tokens = map(self.lower_but_preserve_emoticons, tokens)
        tokens = list(tokens)
        return tokens
```

## Implementazione di modelli con Tensorflow - `.analysis.tf_text`
### Caching - `.database.cache` e `.gathering`
### Creazione di un modello di regressione - `.analysis.tf_text.TensorflowPolarSentimentAnalyzer`
### Creazione di un modello di categorizzazione - `.analysis.tf_text.TensorflowCategorySentimentAnalyzer`
### Esplosione del gradiente

## Implementazione di tokenizzatori di HuggingFace - `.tokenizer.hugging`

## Confronto dei modelli

## Conclusione



[`cfig`]: https://cfig.readthedocs.io
[MongoDB]: https://www.mongodb.com
[`$sample`]: https://www.mongodb.com/docs/manual/reference/operator/aggregation/sample/
[`$rand`]: https://www.mongodb.com/docs/v6.0/reference/operator/aggregation/rand/
[`__debug__`]: https://docs.python.org/3/library/constants.html#debug__
[`-O`]: https://docs.python.org/3/using/cmdline.html#cmdoption-O
[`str.split`]: https://docs.python.org/3/library/stdtypes.html?highlight=str%20split#str.split
[`nltk.tokenize.word_tokenize`]: https://www.nltk.org/api/nltk.tokenize.word_tokenize.html?highlight=word_tokenize#nltk.tokenize.word_tokenize
[Punkt]: https://www.nltk.org/api/nltk.tokenize.PunktSentenceTokenizer.html#nltk.tokenize.PunktSentenceTokenizer
[Treebank]: https://www.nltk.org/api/nltk.tokenize.TreebankWordTokenizer.html#nltk.tokenize.TreebankWordTokenizer
[`nltk.sentiment.util.mark_negation`]: https://www.nltk.org/api/nltk.sentiment.util.html?highlight=nltk+sentiment+util+mark_negation#nltk.sentiment.util.mark_negation
[`nltk.sentiment.SentimentAnalyzer`]: https://www.nltk.org/api/nltk.sentiment.sentiment_analyzer.html?highlight=nltk+sentiment+sentimentanalyzer#nltk.sentiment.sentiment_analyzer.SentimentAnalyzer
[`list`]: https://docs.python.org/3/library/stdtypes.html?highlight=list#list
[tokenizer di Christopher Potts]: http://sentiment.christopherpotts.net/tokenizing.html
