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

In questo progetto si è realizzata una struttura che permettesse di mettere a confronto diversi modi per effettuare sentiment analysis, e poi si sono realizzati su di essa alcuni modelli di sentiment analysis con caratteristiche diverse, al fine di confrontarli.

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

NLTK ha dipendenze aggiuntive che non possono essere scaricate tramite `pip`.

Esse possono essere scaricate eseguendo su un terminale lo script fornito assieme al progetto:

```console
$ ./scripts/download-nltk.sh
```

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

Per eseguire il database MongoDB come processo utente, salvando i dati nella cartella `./data/db`, è possibile eseguire il seguente comando:

```console
$ ./data/scripts/run-db.sh
```

Se [`jq`] è installato sul sistema, è possibile sfruttarlo per ottenere logs più human-friendly con il seguente comando:

```console
$ ./data/scripts/run-db.sh | jq '.msg'
```

#### Importazione dei dati da JSON

Per importare il dataset `./data/raw/reviewsexport.json` fornito a lezione nel database MongoDB è disponibile il seguente script:

```console
$ ./data/scripts/import-db.sh
```

#### Creazione indici

Per creare indici MongoDB potenzialmente utili al funzionamento efficiente del database è possibile eseguire il seguente comando:

```console
$ mongosh < ./data/scripts/index-db.js
```

## Costruzione di una struttura per il confronto

Al fine di effettuare i confronti richiesti dalla consegna dell'attività, si è deciso di realizzare un package Python che permettesse di confrontare vari modelli di Sentiment Analysis tra loro, con tokenizer, training set e evaluation set (spesso detto *test set*) diversi tra loro.

Il package, chiamato `unimore_bda_6`, è composto da vari moduli, ciascuno descritto nelle seguenti sezioni.

### Configurazione ambiente e iperparametri - `.config`

Il primo modulo, `unimore_bda_6.config`, definisce le variabili configurabili del package usando la libreria [`cfig`], e, se eseguito, mostra all'utente un'interfaccia command-line che le descrive e ne mostra i valori attuali.

Viene prima creato un oggetto [`cfig.Configuration`], che opera come contenitore per le variabili configurabili:

```python
import cfig
config = cfig.Configuration()
```

In seguito, viene definita una funzione per ogni variabile configurabile, che elabora il valore ottenuto dalle variabili di ambiente del contesto in cui il programma è eseguito, convertendolo in un formato più facilmente utilizzabile dal programma.

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

(Nel gergo del machine learning / deep learning, queste variabili sono dette iperparametri, perchè configurano il modello, e non vengono alterate nell'addestramento del modello stesso.)

Infine, si aggiunge una chiamata al metodo `cli()` della configurazione, eseguita solo se il modulo viene eseguito direttamente, che mostra all'utente l'interfaccia precedentemente menzionata:

```python
if __name__ == "__main__":
    config.cli()
```

L'esecuzione del modulo `unimore_bda_6.config`, senza variabili d'ambiente definite, darà quindi il seguente output:

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

Il modulo `unimore_bda_6.database` si occupa della connessione al database [MongoDB], del recupero della collezione contenente il dataset di partenza, del recupero dei documenti nella corretta distribuzione, della conversione di essi in un formato più facilmente leggibile da Python, e della creazione di cache su disco per permettere alle librerie che lo supportano di non caricare l'intero dataset in memoria durante l'addestramento di un modello.

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

Esso è utilizzabile nel seguente modo:

```python
with mongo_client_from_config() as client:
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

Il modulo `unimore_bda_6.database.datatypes` contiene contenitori ottimizzati (attraverso l'attributo magico [`__slots__`]) per i dati recuperati dal database, che possono essere riassunti con le seguenti classi circa equivalenti:

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

Essendo il dataset completo composto da 23 milioni, 831 mila e 908 documenti (`23_831_908`), effettuare campionamenti su di esso in fase di sviluppo risulta eccessivamente lento e dispendioso, pertanto ad ogni query il dataset viene rimpicciolito ad un *working set* attraverso l'uso del seguente aggregation pipeline stage, dove `WORKING_SET_SIZE` è sostituito dal suo corrispondente valore nella configurazione (di default `1_000_000`):

```javascript
{"$limit": WORKING_SET_SIZE},
```

##### Dataset con solo recensioni 1* e 5* - `sample_reviews_polar`

Per recuperare un dataset bilanciato di recensioni 1* e 5*, viene utilizzata la seguente funzione con relativa query MongoDB:

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

##### Dataset bilanciato con recensioni 1*, 2*, 3*, 4* e 5* - `sample_reviews_varied`

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

Si è realizzata una classe astratta che rappresentasse un tokenizer qualunque, in modo da avere la stessa interfaccia a livello di codice indipendentemente dal package di tokenizzazione utilizzato:

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

Si sono poi realizzate due classi triviali che ne implementano i metodi astratti, `PlainTokenizer` e `LowercaseTokenizer`, che separano il testo in tokens attraverso la funzione builtin [`str.split`] di Python, rispettivamente mantenendo e rimuovendo la capitalizzazione del testo.

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
    def train(self, training_dataset_func: CachedDatasetFunc) -> None:
        "Train the analyzer with the given training dataset."
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

Il metodo `evaluate` inserisce i risultati di ciascuna predizione effettuata in un oggetto di tipo `EvaluationResults`.

Esso tiene traccia della matrice di confusione per un'iterazione di valutazione, e da essa è in grado di ricavare i valori di richiamo e precisione per ciascuna categoria supportata dal modello; inoltre, calcola l'errore medio assoluto e quadrato tra previsioni e valori effettivi:

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

Si è poi realizzata un'implementazione triviale della classe astratta, `ThreeCheat`, che identifica tutte le recensioni come aventi una valutazione di di 3.0*, in modo da verificare facilmente la correttezza della precedente classe:

```python
class ThreeCheat(BaseSentimentAnalyzer):
    def train(self, training_dataset_func: CachedDatasetFunc) -> None:
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
                    model.train(training_set=sample_func(amount=TRAINING_SET_SIZE))
                    model.evaluate(evaluation_set=sample_func(amount=EVALUATION_SET_SIZE))
```

Le valutazioni di efficacia vengono effettuate fino al raggiungimento di `TARGET_RUNS` addestramenti e valutazioni riuscite, o fino al raggiungimento di `MAXIMUM_RUNS` valutazioni totali (come descritto più avanti, l'addestramento di alcuni modelli potrebbe fallire e dover essere ripetuto).

Il tester inoltre genera il file `./data/logs/results.tsv`, a cui viene aggiunta una riga per ciascuna valutazione effettuata che ne contiene un riepilogo dei risultati:

- la funzione di campionamento e costruzione dataset utilizzata
- il sentiment analyzer utilizzato
- il tokenizer utilizzato
- il numero di run richieste per raggiungere quei risultati
- lo scarto assoluto medio
- lo scarto quadratico medio
- il numero di valutazioni corrette effettuate
- i valori di recall per le recensioni di 1*, 2*, 3*, 4*, e 5*
- i valori di precision per le recensioni di 1*, 2*, 3*, 4*, e 5*


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
3. Conta le occorrenze totali di ogni token della precedente lista per selezionare quelli che compaiono in almeno 4 recensioni diverse
4. Utilizza i token selezionati nel passo precedente per identificare le caratteristiche ("features") da usare per effettuare la classificazione

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

    def train(self, training_dataset_func: CachedDatasetFunc) -> None:
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

### Ri-creazione del tokenizer di Christopher Potts - `.tokenizer.potts`

Per realizzare il punto 1 della consegna, si sono creati due nuovi tokenizer, `PottsTokenizer` e `PottsTokenizerWithNegation`, che implementano il [tokenizer di Christopher Potts] rispettivamente senza marcare e marcando le negazioni sui token attraverso [`ntlk.sentiment.util.mark_negation`].

Essendo il tokenizer originale scritto per Python 2, e non immediatamente compatibile con `BaseTokenizer`, si è scelto di studiare il codice originale e ricrearlo in un formato più adatto a questo progetto.

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

Dopo aver tokenizzato, il tokenizer processa il risultato convertendo il testo a lowercase, facendo attenzione però a non cambiare la capitalizzazione delle emoticon per non cambiare il loro significato (`:D` è diverso da `:d`).

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

## Implementazione di modelli con Tensorflow+Keras - `.analysis.tf_text`

Visti i problemi riscontrati con NLTK, si è deciso di realizzare nuovi modelli utilizzando stavolta [Tensorflow], il package per il deep learning sviluppato da Google, unito a [Keras], API di Tensorflow che permette la definizione di modelli di deep learning attraverso un linguaggio ad alto livello.

Tensorflow prende il nome dai *tensori*, le strutture matematiche su cui si basa, che consistono in una maggiore astrazione delle matrici o degli array, e che vengono implementate dalla libreria stessa nella classe [`tensorflow.Tensor`].

### Aggiunta di un validation set

La documentazione di Tensorflow suggerisce, in fase di addestramento di modello, di includere un *validation set*, un piccolo dataset su cui misurare le metriche del modello ad ogni epoca di addestramento, in modo da poter verificare in tempo reale che non si stia verificando underfitting o overfitting.

Si è quindi deciso di includerlo come parametro di `BaseSentimentAnalyzer.train`:

```python
    ...

    @abc.abstractmethod
    def train(self, training_dataset_func: CachedDatasetFunc, validation_dataset_func: CachedDatasetFunc) -> None:
        """
        Train the analyzer with the given training and validation datasets.
        """
        raise NotImplementedError()

    ...
```

Si è anche aggiornato il `.__main__` e la `.config` per supportare questa nuova funzionalità:

```python
# Pseudo-codice non corrispondente al main finale
if __name__ == "__main__":
    for sample_func in [sample_reviews_polar, sample_reviews_varied]:
        for SentimentAnalyzer in [ThreeCheat, NLTKSentimentAnalyzer, ...]:
            for Tokenizer in [PlainTokenizer, LowercaseTokenizer, PottsTokenizer, PottsTokenizerWithNegation, ...]:
                for run in range(TARGET_RUNS):
                    model = SentimentAnalyzer(tokenizer=Tokenizer())
                    model.train(training_set=sample_func(amount=TRAINING_SET_SIZE), validation_set=sample_func(amount=VALIDATION_SET_SIZE))
                    model.evaluate(evaluation_set=sample_func(amount=EVALUATION_SET_SIZE))
```

### Caching - `.database.cache` e `.gathering`

Per essere efficienti, i modelli di Tensorflow richiedono che i dati vengano inseriti in un formato molto specifico: un'istanza della classe [`tensorflow.data.Dataset`].

I dataset, per essere creati, richiedono però che gli venga dato in input un *generatore* (funzione che crea un iteratore quando chiamata), e non un *iteratore* (oggetto con un puntatore al successivo) come quello restituito dalle query di MongoDB, in quanto Tensorflow necessita di ricominciare l'iterazione da capo dopo ogni epoca di addestramento.

Un modo semplice per ovviare al problema sarebbe stato raccogliere in una [`list`] l'iteratore creato da MongoDB, ma ciò caricherebbe l'intero dataset contemporaneamente in memoria, ricreando il problema riscontrato con NLTK.

Si è allora adottata una soluzione alternativa: creare una cache su disco composta un file per ciascun documento recuperato da MongoDB, in modo che quando Tensorflow necessita di ritornare al primo documento, possa farlo ritornando semplicemente al primo file.

```python
def store_cache(reviews: t.Iterator[TextReview], path: str | pathlib.Path) -> None:
    "Store the contents of the given `Review` iterator to different files in a directory at the given path."
    path = pathlib.Path(path)
    path.mkdir(parents=True)
    for index, document in enumerate(reviews):
        document_path = path.joinpath(f"{index}.pickle")
        with open(document_path, "wb") as file:
            pickle.dump(document, file)

def load_cache(path: str | pathlib.Path) -> CachedDatasetFunc:
    "Load the contents of a directory into a `Review` generator."
    path = pathlib.Path(path)

    def data_cache_loader():
        document_paths = path.iterdir()
        for document_path in document_paths:
            document_path = pathlib.Path(document_path)
            with open(document_path, "rb") as file:
                result: TextReview = pickle.load(file)
                yield result

    return data_cache_loader

def delete_cache(path: str | pathlib.Path) -> None:
    "Delete the given cache directory."
    path = pathlib.Path(path)
    shutil.rmtree(path)
```

Si è poi creata una classe `Caches` che si occupa di creare, gestire, ed eliminare le cache dei tre dataset nelle cartelle `./data/training`, `./data/validation` e `./data/evaluation`:

```python
@dataclasses.dataclass
class Caches:
    """
    Container for the three generators that can create datasets.
    """

    training: CachedDatasetFunc
    validation: CachedDatasetFunc
    evaluation: CachedDatasetFunc

    @classmethod
    @contextlib.contextmanager
    def from_database_samples(cls, collection: pymongo.collection.Collection, sample_func: SampleFunc) -> t.ContextManager["Caches"]:
        "Create a new caches object from a database collection and a sampling function."

        reviews_training = sample_func(collection, TRAINING_SET_SIZE.__wrapped__)
        reviews_validation = sample_func(collection, VALIDATION_SET_SIZE.__wrapped__)
        reviews_evaluation = sample_func(collection, EVALUATION_SET_SIZE.__wrapped__)

        store_cache(reviews_training, "./data/training")
        store_cache(reviews_validation, "./data/validation")
        store_cache(reviews_evaluation, "./data/evaluation")

        training_cache = load_cache("./data/training")
        validation_cache = load_cache("./data/validation")
        evaluation_cache = load_cache("./data/evaluation")

        yield Caches(training=training_cache, validation=validation_cache, evaluation=evaluation_cache)

        delete_cache("./data/training")
        delete_cache("./data/validation")
        delete_cache("./data/evaluation")

    ...
```

### Creazione del modello base - `.analysis.tf_text.Tensorflow

Si è determinata una struttura comune che potesse essere usata per tutti i tipi di Sentiment Analyzer realizzati con Tensorflow:

```python
class TensorflowSentimentAnalyzer(BaseSentimentAnalyzer, metaclass=abc.ABCMeta):
    ...
```

#### Formato del modello

Essa richiede che le sottoclassi usino un modello `tensorflow.keras.Sequential`, ovvero con un solo layer di neuroni in input e un solo layer di neuroni in output:

```python
    ...

    @abc.abstractmethod
    def _build_model(self) -> tensorflow.keras.Sequential:
        "Create the `tensorflow.keras.Sequential` model that should be executed by this sentiment analyzer."
        raise NotImplementedError()

    ...
```

#### Conversione da-a tensori

Dato che i modelli di Tensorflow richiedono che ciascun dato fornito in input o emesso in output sia un'istanza di `tensorflow.Tensor`, le sottoclassi devono anche definire metodi per convertire le stelle delle recensioni in un equivalente `tensorflow.Tensor` e viceversa:

```python
    ...

    @abc.abstractmethod
    def _rating_to_input(self, rating: float) -> tensorflow.Tensor:
        "Convert a review rating to a `tensorflow.Tensor`."
        raise NotImplementedError()

    @abc.abstractmethod
    def _prediction_to_rating(self, prediction: tensorflow.Tensor) -> float:
        "Convert the results of `tensorflow.keras.Sequential.predict` into a review rating."
        raise NotImplementedError()

    ...
```

Attraverso di essi, la classe è in grado di costruire il [`tensorflow.data.Dataset`] necessario al modello:

```python
    ...

    @staticmethod
    def _tokens_to_tensor(tokens: t.Iterator[str]) -> tensorflow.Tensor:
        "Convert an iterator of tokens to a `tensorflow.Tensor`."
        tensor = tensorflow.convert_to_tensor(
            [list(tokens)],
            dtype=tensorflow.string,
            name="tokens"
        )
        return tensor

    def _build_dataset(self, dataset_func: CachedDatasetFunc) -> tensorflow.data.Dataset:
        "Create a `tensorflow.data.Dataset` from the given `CachedDatasetFunc`."
        def dataset_generator():
            for review in dataset_func():
                review: TextReview
                review: TokenizedReview = self.tokenizer.tokenize_review(review)
                tokens: tensorflow.Tensor = self._tokens_to_tensor(review.tokens)
                rating: tensorflow.Tensor = self._rating_to_input(review.rating)
                yield tokens, rating

        dataset = tensorflow.data.Dataset.from_generator(
            dataset_generator,
            output_signature=(
                tensorflow.TensorSpec(shape=(1, None,), dtype=tensorflow.string, name="tokens"),
                self._ratingtensor_shape(),
            ),
        )
        dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tensorflow.data.AUTOTUNE)
        return dataset

    ...
```

#### Lookup delle stringhe

I modelli di deep learning di Tensorflow non sono in grado di processare stringhe direttamente; esse devono essere prima convertite in formato numerico.

All'inizializzazione, la struttura base crea un layer di tipo [`tensorflow.keras.layers.StringLookup`], che prende in input una lista di token e la converte in una lista di numeri interi, assegnando a ciascun token un numero diverso:

```python
    ...

    def __init__(self, *, tokenizer: BaseTokenizer):
        ...
        self.string_lookup_layer = tensorflow.keras.layers.StringLookup(max_tokens=TENSORFLOW_MAX_FEATURES)
        ...

    ...
```

Ciò comporta che, prima dell'addestramento del modello, il layer deve essere adattato, ovvero deve essere costruito un vocabolario che associa ogni possibile termine ad un numero; qualsiasi token al di fuori da questo vocabolario verrà convertito in `0`.

Per esempio, `["ciao", "come", "stai", "?"]` potrebbe essere convertito in `[1, 2, 0, 3]` se il modello non è stato adattato con il token `"stai"`.

#### Addestramento

La struttura base `TensorflowSentimentAnalyzer` uniforma la fase di addestramento per tutti i modelli realizzandola attraverso le seguenti fasi:

1. Creazione di `tensorflow.data.Dataset` dalla cache del training set e del validation set
2. Adattamento del layer di string lookup
3. Fitting del modello per `TENSORFLOW_EPOCHS` epoche

```python
    ...

    def train(self, training_dataset_func: CachedDatasetFunc, validation_dataset_func: CachedDatasetFunc) -> None:
        training_set = self._build_dataset(training_dataset_func)
        validation_set = self._build_dataset(validation_dataset_func)

        vocabulary = training_set.map(lambda tokens, rating: tokens)
        self.string_lookup_layer.adapt(vocabulary)

        self.history: tensorflow.keras.callbacks.History | None  = self.model.fit(
            training_set,
            validation_data=validation_set,
            epochs=TENSORFLOW_EPOCHS.__wrapped__,
            callbacks=[
                tensorflow.keras.callbacks.TerminateOnNaN()
            ],
        )

        if len(self.history.epoch) < TENSORFLOW_EPOCHS.__wrapped__:
            self.failed = True
            raise TrainingFailedError()
        else:
            self.trained = True

    ...
```

##### Esplosione del gradiente

Il metodo `train` si occupa anche di gestire una situazione particolare: quella in cui l'errore del modello sul training set diventi `NaN` per via del fenomeno di [esplosione del gradiente].

Grazie al callback `tensorflow.keras.callbacks.TerminateOnNaN`, nel momento in cui viene riconosciuto che l'errore è diventato `NaN`, l'addestramento viene interrotto, e viene sollevato un `TrainingFailedError`.

Si è quindi aggiornato il main per gestire l'eccezione e ricominciare l'addestramento da capo qualora essa si verificasse:

```python
# Pseudo-codice non corrispondente al main finale
if __name__ == "__main__":
    for sample_func in [sample_reviews_polar, sample_reviews_varied]:
        for SentimentAnalyzer in [ThreeCheat, NLTKSentimentAnalyzer, ...]:
            for Tokenizer in [PlainTokenizer, LowercaseTokenizer, PottsTokenizer, PottsTokenizerWithNegation, ...]:
                runs = 0
                successful_runs = 0
                while True:
                    if runs >= MAXIMUM_RUNS or successful_runs >= TARGET_RUNS:
                        break
                    runs += 1
                    model = SentimentAnalyzer(tokenizer=Tokenizer())
                    try:
                        model.train(training_set=sample_func(amount=TRAINING_SET_SIZE), validation_set=sample_func(amount=VALIDATION_SET_SIZE))
                    except TrainingFailedError:
                        continue
                    model.evaluate(evaluation_set=sample_func(amount=EVALUATION_SET_SIZE))
                    successful_runs += 1
```

#### Utilizzo

Anche l'utilizzo del modello è uniformato da `TensorflowSentimentAnalyzer`:

```python
    ...

    def use(self, text: str) -> float:
        tokens = self.tokenizer.tokenize(text)
        tokens = self._tokens_to_tensor(tokens)
        prediction = self.model.predict(tokens, verbose=False)
        prediction = self._prediction_to_rating(prediction)
        return prediction

    ...
```

1. Il testo passato in input viene tokenizzato dal tokenizzatore selezionato;
2. i token vengono trasformati in un tensore di stringhe;
3. il tensore di stringhe viene passato al modello, il primo layer del quale effettua string lookup;
4. il modello emette un output in forma di tensore;
5. il tensore viene convertito nel numero di stelle predetto.

### Creazione di un modello di regressione - `.analysis.tf_text.TensorflowPolarSentimentAnalyzer`

Uno dei due tipi di modello di deep learning realizzati è un modello di regressione, ovvero un modello che dà in output un singolo valore a virgola mobile `0 < y < 1` rappresentante la confidenza che la recensione data sia positiva, il cui complementare `z = 1 - y` rappresenta la confidenza che la recensione data sia negativa:

```python
class TensorflowPolarSentimentAnalyzer(TensorflowSentimentAnalyzer):
    def _ratingtensor_shape(self) -> tensorflow.TensorSpec:
        spec = tensorflow.TensorSpec(shape=(1,), dtype=tensorflow.float32, name="rating_value")
        return spec

    ...
```

Si considera valida la predizione in cui il modello ha più confidenza: positiva, o 5.0*, se `y >= 0.5`, oppure negativa, o 1.0*, se `y < 0.5`:

```python
    ...

    def _prediction_to_rating(self, prediction: numpy.array) -> float:
        rating: float = prediction[0, 0]
        rating = 1.0 if rating < 0.5 else 5.0
        return rating

    ...
```

Seguendo le best practices per i modelli di questo tipo, si normalizza il valore in input a un numero `0.0 < x < 1.0`:


```python
    ...

    def _rating_to_input(self, rating: float) -> tensorflow.Tensor:
        normalized_rating = (rating - 1) / 4
        tensor = tensorflow.convert_to_tensor(
            [normalized_rating],
            dtype=tensorflow.float32,
            name="rating_value"
        )
        return tensor

    ...
```

Infine, si costruiscono i layer del modello di deep learning:

1. il primo layer, [`tensorflow.keras.layers.Embedding`], impara a convertire i tensori di interi di dimensione variabile che riceve in input in tensori di numeri a virgola mobile di dimensione fissa in cui ciascun valore rappresenta un significato delle parole;

2. il secondo (e quarto e sesto) layer, [`tensorflow.keras.layers.Dropout`], imposta casualmente a `0.0` il 25% dei valori contenuti nei tensori che riceve in input, rendendo "più indipendenti" le correlazioni apprese dallo strato precedente di neuroni e così evitando l'overfitting;

3. il terzo layer, [`tensorflow.keras.layers.GlobalAveragePooling1D`], calcola l'influenza media di ciascun significato sulla confidenza del modello relativamente a una determinata recensione

4. il quinto (e sesto) layer, [`tensorflow.keras.layers.Dense`], sono strati di neuroni interconnessi in grado di apprendere semplici collegamenti tra significati e sentimenti

```python
    ...

    def _build_model(self) -> tensorflow.keras.Sequential:
        model = tensorflow.keras.Sequential([
            self.string_lookup_layer,
            tensorflow.keras.layers.Embedding(
                input_dim=TENSORFLOW_MAX_FEATURES.__wrapped__ + 1,
                output_dim=TENSORFLOW_EMBEDDING_SIZE.__wrapped__,
            ),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.GlobalAveragePooling1D(),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.Dense(8),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.Dense(1, activation=tensorflow.keras.activations.sigmoid),
        ])

        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(clipnorm=1.0),
            loss=tensorflow.keras.losses.MeanAbsoluteError(),
        )

        log.debug("Compiled model: %s", model)
        return model
```

### Creazione di un modello di categorizzazione - `.analysis.tf_text.TensorflowCategorySentimentAnalyzer`

L'altro tipo di modello realizzato è invece un modello di categorizzazione one-of, ovvero un modello che dà in output cinque diversi valori a virgola mobile, ciascuno rappresentante la confidenza che la data recensione appartenga a ciascuna delle date categorie:

```python
class TensorflowCategorySentimentAnalyzer(TensorflowSentimentAnalyzer):
    def _ratingtensor_shape(self) -> tensorflow.TensorSpec:
        spec = tensorflow.TensorSpec(shape=(1, 5), dtype=tensorflow.float32, name="rating_one_hot")
        return spec

    ...
```

Si considera valida la predizione nella quale il modello ha confidenza più alta:

```python
    ...

    def _prediction_to_rating(self, prediction: tensorflow.Tensor) -> float:
        best_prediction = None
        best_prediction_index = None

        for index, prediction in enumerate(iter(prediction[0])):
            if best_prediction is None or prediction > best_prediction:
                best_prediction = prediction
                best_prediction_index = index

        result = float(best_prediction_index) + 1.0
        return result

    ...
```

Questa volta, si utilizza l'encoding *one-hot* per gli input del modello in modo da creare una separazione netta tra le cinque possibili categorie in cui una recensione potrebbe cadere (1*, 2*, 3*, 4*, 5*).

Esso consiste nel creare un tensore di cinque elementi, ciascuno rappresentante una categoria, e di impostarlo a 1.0 se la recensione appartiene a una categoria o a 0.0 se essa non vi appartiene.

```python
    ...

    def _rating_to_input(self, rating: float) -> tensorflow.Tensor:
        tensor = tensorflow.convert_to_tensor(
            [[
                1.0 if rating == 1.0 else 0.0,
                1.0 if rating == 2.0 else 0.0,
                1.0 if rating == 3.0 else 0.0,
                1.0 if rating == 4.0 else 0.0,
                1.0 if rating == 5.0 else 0.0,
            ]],
            dtype=tensorflow.float32,
            name="rating_one_hot"
        )
        return tensor

    ...
```

Infine, si costruisce un modello molto simile al precedente, ma con 5 neuroni in output, il cui valore viene normalizzato attraverso la funzione *softmax*:

```python
    ...

    def _build_model(self) -> tensorflow.keras.Sequential:
        model = tensorflow.keras.Sequential([
            self.string_lookup_layer,
            tensorflow.keras.layers.Embedding(
                input_dim=TENSORFLOW_MAX_FEATURES.__wrapped__ + 1,
                output_dim=TENSORFLOW_EMBEDDING_SIZE.__wrapped__,
            ),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.GlobalAveragePooling1D(),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.Dense(8),
            tensorflow.keras.layers.Dropout(0.25),
            tensorflow.keras.layers.Dense(5, activation="softmax"),
        ])

        model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(clipnorm=1.0),
            loss=tensorflow.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tensorflow.keras.metrics.CategoricalAccuracy(),
            ]
        )

        return model

```

## Implementazione di tokenizzatori di HuggingFace - `.tokenizer.hugging`

Come ultima funzionalità, si implementa la possibilità di importare tokenizzatori presenti su [HuggingFace] con la classe astratta `HuggingTokenizer`:

```python
class HuggingTokenizer(BaseTokenizer, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()
        self.hug: tokenizers.Tokenizer = self._build_hugging_tokenizer()

    @abc.abstractmethod
    def _build_hugging_tokenizer(self) -> tokenizers.Tokenizer:
        raise NotImplementedError()

    def tokenize(self, text: str) -> t.Iterator[str]:
        return self.hug.encode(text).tokens
```

Utilizzandola, si implementa il tokenizzatore [`bert-base-cased`] per testarne l'efficacia:

```python
class HuggingBertTokenizer(HuggingTokenizer):
    def _build_hugging_tokenizer(self):
        return tokenizers.Tokenizer.from_pretrained("bert-base-cased")
```

## Regolazione degli iperparametri

Il tester è stato eseguito alcune volte con diverse configurazioni di parametri per verificarne il corretto funzionamento e determinare empiricamente gli iperparametri migliori da utilizzare durante la run vera e propria.

Si riportano i parametri regolati assieme ai valori a cui essi sono stati impostati.

### `TRAINING_SET_SIZE`

> Il numero di recensioni di ogni categoria da recuperare per formare il training set.

L'approccio all'addestramento utilizzato da [`nltk.sentiment.SentimentAnalyzer`] si è rivelato problematico, in quanto non in grado di scalare per dimensioni molto grandi di training set: i suoi metodi non sembrano gestire correttamente gli iteratori, meccanismo attraverso il quale Python può realizzare lazy-loading di dati.

Inoltre, si è notato che il problema di [esplosione del gradiente](#esplosione-del-gradiente) si verifica tanto più di frequente quanto è grande il training set.

Per questi due motivi si è deciso di limitare la dimensione del training set a `4_000` documenti per categoria.

### `VALIDATION_SET_SIZE`

> Il numero di recensioni di ogni categoria da recuperare per formare il validation set.

Si è scelto di creare un validation set della dimensione di un decimo del training set, ovvero di `400` documenti per categoria.

### `EVALUATION_SET_SIZE`

> Il numero di recensioni di ogni categoria da recuperare per formare il test set.

Durante la sperimentazione manuale, si è notato che i risultati della valutazione del test set giungevano a convergenza dopo l'elaborazione di circa `1_000` documenti, pertanto si è impostato l'iperparametro a quel numero.

### `WORKING_SET_SIZE`

> Il numero di recensioni del database da considerare.
> 
> Si suggerisce di impostarlo a un numero basso per evitare rallentamenti nell'esecuzione delle query.

Si è determinato che `5_000_000` fosse un buon numero che permettesse di avere ottima casualità nel dataset senza comportare tempi di campionamento troppo lunghi.

### `TENSORFLOW_EMBEDDING_SIZE`

> La dimensione del tensore degli embeddings da usare nei modelli Tensorflow.

Si sono testati vari valori per questo iperparametro, e non sono state notate differenze significative nei risultati ottenuti; perciò, l'iperparametro è stato impostato a un valore di `12`, leggermente superiore a quello minimo di `8` suggerito dalla documentazione di Tensorflow.

### `TENSORFLOW_MAX_FEATURES`

> Il numero massimo di features da usare nei modelli Tensorflow.

Come per il parametro precedente, non si sono notate particolari differenze, quindi si è scelto di rimanere sul sicuro permettendo fino a `300_000` token diversi di essere appresi.

### `TENSORFLOW_EPOCHS`

> Il numero di epoche per cui addestrare i modelli Tensorflow.

Si è notato che qualsiasi addestramento successivo alla terza epoca risultava in un aumento nella loss dei modelli, probabilmente dovuta all'occorrenza di overfitting in essi.

Per prevenire il fenomeno si è allora deciso di impostare il numero massimo di epoche a `3`.

## Confronto dei modelli

Si sono effettuate 5 esecuzioni del tester, totalizzando 245 run dei modelli.

I risultati grezzi ottenuti sono disponibili all'interno dei file `./data/logs/results-success2.tsv`,  `./data/logs/results-success3.tsv`,  `./data/logs/results-success4.tsv`,  `./data/logs/results-success5.tsv`, e `./data/logs/results-success6.tsv`. 

Si riportano invece direttamente all'interno di questa relazione i risultati cumulativi ottenuti, ottenuti effettuando la media tra i risultati ottenuti nelle cinque esecuzioni.

### Scarto

#### Recensioni 1* e 5* - `sample_reviews_polar`

| Analyzer | Tokenizer | Scarto assoluto medio | Scarto quadratico medio |
|---|---|--:|--:|
| `ThreeCheat` | `PlainTokenizer` | 2.000 | 4.000 |
| `ThreeCheat` | `LowercaseTokenizer` | 2.000 | 4.000 |
| `ThreeCheat` | `NLTKWordTokenizer` | 2.000 | 4.000 |
| `ThreeCheat` | `PottsTokenizer` | 2.000 | 4.000 |
| `ThreeCheat` | `PottsTokenizerWithNegation` | 2.000 | 4.000 |
| `ThreeCheat` | `HuggingBertTokenizer` | 2.000 | 4.000 |
| `NLTKSentimentAnalyzer` | `PlainTokenizer` | 1.015 | 4.061 |
| `NLTKSentimentAnalyzer` | `LowercaseTokenizer` | 0.981 | 3.923 |
| `NLTKSentimentAnalyzer` | `NLTKWordTokenizer` | 1.086 | 4.346 |
| `NLTKSentimentAnalyzer` | `PottsTokenizer` | 0.985 | 3.939 |
| `NLTKSentimentAnalyzer` | `PottsTokenizerWithNegation` | 1.107 | 4.429 |
| `NLTKSentimentAnalyzer` | `HuggingBertTokenizer` | 1.118 | 4.474 |
| `TensorflowPolarSentimentAnalyzer` | `PlainTokenizer` | 0.573 | 2.291 |
| `TensorflowPolarSentimentAnalyzer` | `LowercaseTokenizer` | 0.522 | 2.086 |
| `TensorflowPolarSentimentAnalyzer` | `NLTKWordTokenizer` | 0.534 | 2.134 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizer` | 0.502 | 2.010 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.465 | 1.859 |
| `TensorflowPolarSentimentAnalyzer` | `HuggingBertTokenizer` | 0.526 | 2.106 |
| `TensorflowCategorySentimentAnalyzer` | `PlainTokenizer` | 0.503 | 2.013 |
| `TensorflowCategorySentimentAnalyzer` | `LowercaseTokenizer` | 0.514 | 2.058 |
| `TensorflowCategorySentimentAnalyzer` | `NLTKWordTokenizer` | 0.506 | 2.026 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizer` | 0.490 | 1.958 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.513 | 2.051 |
| `TensorflowCategorySentimentAnalyzer` | `HuggingBertTokenizer` | 0.530 | 2.118 |

#### Recensioni 1*, 2*, 3*, 4* e 5* - `sample_reviews_varied`

| Analyzer | Tokenizer | Scarto assoluto medio | Scarto quadratico medio |
|---|---|--:|--:|
| `ThreeCheat` | `PlainTokenizer` | 1.200 | 2.000 |
| `ThreeCheat` | `LowercaseTokenizer` | 1.200 | 2.000 |
| `ThreeCheat` | `NLTKWordTokenizer` | 1.200 | 2.000 |
| `ThreeCheat` | `PottsTokenizer` | 1.200 | 2.000 |
| `ThreeCheat` | `PottsTokenizerWithNegation` | 1.200 | 2.000 |
| `ThreeCheat` | `HuggingBertTokenizer` | 1.200 | 2.000 |
| `NLTKSentimentAnalyzer` | `PlainTokenizer` | 1.254 | 3.096 |
| `NLTKSentimentAnalyzer` | `LowercaseTokenizer` | 1.274 | 3.188 |
| `NLTKSentimentAnalyzer` | `NLTKWordTokenizer` | 1.267 | 3.116 |
| `NLTKSentimentAnalyzer` | `PottsTokenizer` | 1.263 | 3.128 |
| `NLTKSentimentAnalyzer` | `PottsTokenizerWithNegation` | 1.280 | 3.164 |
| `NLTKSentimentAnalyzer` | `HuggingBertTokenizer` | 1.340 | 3.441 |
| `TensorflowPolarSentimentAnalyzer` | `PlainTokenizer` | 1.287 | 3.149 |
| `TensorflowPolarSentimentAnalyzer` | `LowercaseTokenizer` | 1.261 | 3.045 |
| `TensorflowPolarSentimentAnalyzer` | `NLTKWordTokenizer` | 1.252 | 3.006 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizer` | 1.218 | 2.870 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizerWithNegation` | 1.229 | 2.915 |
| `TensorflowPolarSentimentAnalyzer` | `HuggingBertTokenizer` | 1.208 | 2.834 |
| `TensorflowCategorySentimentAnalyzer` | `PlainTokenizer` | 0.866 | 1.583 |
| `TensorflowCategorySentimentAnalyzer` | `LowercaseTokenizer` | 0.821 | 1.523 |
| `TensorflowCategorySentimentAnalyzer` | `NLTKWordTokenizer` | 0.813 | 1.446 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizer` | 0.792 | 1.441 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.839 | 1.516 |
| `TensorflowCategorySentimentAnalyzer` | `HuggingBertTokenizer` | 0.821 | 1.513 |

### Accuracy

#### Recensioni 1* e 5* - `sample_reviews_polar`

| Analyzer | Tokenizer | Accuracy |
|---|---|---:|
| `ThreeCheat` | `PlainTokenizer` | 0.000 |
| `ThreeCheat` | `LowercaseTokenizer` | 0.000 |
| `ThreeCheat` | `NLTKWordTokenizer` | 0.000 |
| `ThreeCheat` | `PottsTokenizer` | 0.000 |
| `ThreeCheat` | `PottsTokenizerWithNegation` | 0.000 |
| `ThreeCheat` | `HuggingBertTokenizer` | 0.000 |
| `NLTKSentimentAnalyzer` | `PlainTokenizer` | 0.746 |
| `NLTKSentimentAnalyzer` | `LowercaseTokenizer` | 0.755 |
| `NLTKSentimentAnalyzer` | `NLTKWordTokenizer` | 0.728 |
| `NLTKSentimentAnalyzer` | `PottsTokenizer` | 0.754 |
| `NLTKSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.723 |
| `NLTKSentimentAnalyzer` | `HuggingBertTokenizer` | 0.720 |
| `TensorflowPolarSentimentAnalyzer` | `PlainTokenizer` | 0.857 |
| `TensorflowPolarSentimentAnalyzer` | `LowercaseTokenizer` | 0.870 |
| `TensorflowPolarSentimentAnalyzer` | `NLTKWordTokenizer` | 0.867 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizer` | 0.874 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.884 |
| `TensorflowPolarSentimentAnalyzer` | `HuggingBertTokenizer` | 0.868 |
| `TensorflowCategorySentimentAnalyzer` | `PlainTokenizer` | 0.874 |
| `TensorflowCategorySentimentAnalyzer` | `LowercaseTokenizer` | 0.871 |
| `TensorflowCategorySentimentAnalyzer` | `NLTKWordTokenizer` | 0.873 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizer` | 0.878 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.872 |
| `TensorflowCategorySentimentAnalyzer` | `HuggingBertTokenizer` | 0.868 |

#### Recensioni 1*, 2*, 3*, 4* e 5* - `sample_reviews_varied`

| Analyzer | Tokenizer | Accuracy |
|---|---|---:|
| `ThreeCheat` | `PlainTokenizer` | 0.200 |
| `ThreeCheat` | `LowercaseTokenizer` | 0.200 |
| `ThreeCheat` | `NLTKWordTokenizer` | 0.200 |
| `ThreeCheat` | `PottsTokenizer` | 0.200 |
| `ThreeCheat` | `PottsTokenizerWithNegation` | 0.200 |
| `ThreeCheat` | `HuggingBertTokenizer` | 0.200 |
| `NLTKSentimentAnalyzer` | `PlainTokenizer` | 0.340 |
| `NLTKSentimentAnalyzer` | `LowercaseTokenizer` | 0.335 |
| `NLTKSentimentAnalyzer` | `NLTKWordTokenizer` | 0.329 |
| `NLTKSentimentAnalyzer` | `PottsTokenizer` | 0.339 |
| `NLTKSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.326 |
| `NLTKSentimentAnalyzer` | `HuggingBertTokenizer` | 0.321 |
| `TensorflowPolarSentimentAnalyzer` | `PlainTokenizer` | 0.332 |
| `TensorflowPolarSentimentAnalyzer` | `LowercaseTokenizer` | 0.337 |
| `TensorflowPolarSentimentAnalyzer` | `NLTKWordTokenizer` | 0.339 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizer` | 0.343 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.339 |
| `TensorflowPolarSentimentAnalyzer` | `HuggingBertTokenizer` | 0.346 |
| `TensorflowCategorySentimentAnalyzer` | `PlainTokenizer` | 0.398 |
| `TensorflowCategorySentimentAnalyzer` | `LowercaseTokenizer` | 0.428 |
| `TensorflowCategorySentimentAnalyzer` | `NLTKWordTokenizer` | 0.427 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizer` | 0.444 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.414 |
| `TensorflowCategorySentimentAnalyzer` | `HuggingBertTokenizer` | 0.427 |

### Recall

#### Recensioni 1* e 5* - `sample_reviews_polar`

| Analyzer | Tokenizer | Recall 1* | Recall 5* | Recall Avg |
|---|---|---:|---:|---|
| `ThreeCheat` | `PlainTokenizer` | 0.000 | 0.000 | 0.000 |
| `ThreeCheat` | `LowercaseTokenizer` | 0.000 | 0.000 | 0.000 |
| `ThreeCheat` | `NLTKWordTokenizer` | 0.000 | 0.000 | 0.000 |
| `ThreeCheat` | `PottsTokenizer` | 0.000 | 0.000 | 0.000 |
| `ThreeCheat` | `PottsTokenizerWithNegation` | 0.000 | 0.000 | 0.000 |
| `ThreeCheat` | `HuggingBertTokenizer` | 0.000 | 0.000 | 0.000 |
| `NLTKSentimentAnalyzer` | `PlainTokenizer` | 0.646 | 0.847 | 0.746 |
| `NLTKSentimentAnalyzer` | `LowercaseTokenizer` | 0.660 | 0.849 | 0.755 |
| `NLTKSentimentAnalyzer` | `NLTKWordTokenizer` | 0.615 | 0.842 | 0.728 |
| `NLTKSentimentAnalyzer` | `PottsTokenizer` | 0.648 | 0.860 | 0.754 |
| `NLTKSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.600 | 0.846 | 0.723 |
| `NLTKSentimentAnalyzer` | `HuggingBertTokenizer` | 0.601 | 0.840 | 0.720 |
| `TensorflowPolarSentimentAnalyzer` | `PlainTokenizer` | 0.874 | 0.840 | 0.857 |
| `TensorflowPolarSentimentAnalyzer` | `LowercaseTokenizer` | 0.879 | 0.860 | 0.870 |
| `TensorflowPolarSentimentAnalyzer` | `NLTKWordTokenizer` | 0.871 | 0.862 | 0.867 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizer` | 0.884 | 0.865 | 0.874 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.898 | 0.870 | 0.884 |
| `TensorflowPolarSentimentAnalyzer` | `HuggingBertTokenizer` | 0.882 | 0.854 | 0.868 |
| `TensorflowCategorySentimentAnalyzer` | `PlainTokenizer` | 0.891 | 0.857 | 0.874 |
| `TensorflowCategorySentimentAnalyzer` | `LowercaseTokenizer` | 0.879 | 0.864 | 0.871 |
| `TensorflowCategorySentimentAnalyzer` | `NLTKWordTokenizer` | 0.880 | 0.867 | 0.873 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizer` | 0.890 | 0.865 | 0.878 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.858 | 0.886 | 0.872 |
| `TensorflowCategorySentimentAnalyzer` | `HuggingBertTokenizer` | 0.872 | 0.863 | 0.868 |

#### Recensioni 1*, 2*, 3*, 4* e 5* - `sample_reviews_varied`

| Analyzer | Tokenizer | Recall 1* | Recall 2* | Recall 3* | Recall 4* | Recall 5* | Recall Avg |
|---|---|---:|---:|---:|---:|---:|--:|
| `ThreeCheat` | `PlainTokenizer` | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.200 |
| `ThreeCheat` | `LowercaseTokenizer` | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.200 |
| `ThreeCheat` | `NLTKWordTokenizer` | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.200 |
| `ThreeCheat` | `PottsTokenizer` | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.200 |
| `ThreeCheat` | `PottsTokenizerWithNegation` | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.200 |
| `ThreeCheat` | `HuggingBertTokenizer` | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.200 |
| `NLTKSentimentAnalyzer` | `PlainTokenizer` | 0.384 | 0.302 | 0.182 | 0.121 | 0.710 | 0.340 |
| `NLTKSentimentAnalyzer` | `LowercaseTokenizer` | 0.376 | 0.239 | 0.200 | 0.146 | 0.713 | 0.335 |
| `NLTKSentimentAnalyzer` | `NLTKWordTokenizer` | 0.386 | 0.245 | 0.170 | 0.131 | 0.713 | 0.329 |
| `NLTKSentimentAnalyzer` | `PottsTokenizer` | 0.406 | 0.251 | 0.196 | 0.127 | 0.714 | 0.339 |
| `NLTKSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.361 | 0.203 | 0.207 | 0.111 | 0.749 | 0.326 |
| `NLTKSentimentAnalyzer` | `HuggingBertTokenizer` | 0.304 | 0.271 | 0.176 | 0.131 | 0.725 | 0.321 |
| `TensorflowPolarSentimentAnalyzer` | `PlainTokenizer` | 0.840 | 0.000 | 0.000 | 0.000 | 0.820 | 0.332 |
| `TensorflowPolarSentimentAnalyzer` | `LowercaseTokenizer` | 0.858 | 0.000 | 0.000 | 0.000 | 0.828 | 0.337 |
| `TensorflowPolarSentimentAnalyzer` | `NLTKWordTokenizer` | 0.816 | 0.000 | 0.000 | 0.000 | 0.881 | 0.339 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizer` | 0.870 | 0.000 | 0.000 | 0.000 | 0.845 | 0.343 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.839 | 0.000 | 0.000 | 0.000 | 0.858 | 0.339 |
| `TensorflowPolarSentimentAnalyzer` | `HuggingBertTokenizer` | 0.858 | 0.000 | 0.000 | 0.000 | 0.870 | 0.346 |
| `TensorflowCategorySentimentAnalyzer` | `PlainTokenizer` | 0.431 | 0.366 | 0.402 | 0.347 | 0.443 | 0.398 |
| `TensorflowCategorySentimentAnalyzer` | `LowercaseTokenizer` | 0.522 | 0.332 | 0.385 | 0.447 | 0.456 | 0.428 |
| `TensorflowCategorySentimentAnalyzer` | `NLTKWordTokenizer` | 0.531 | 0.307 | 0.403 | 0.400 | 0.493 | 0.427 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizer` | 0.514 | 0.405 | 0.408 | 0.384 | 0.511 | 0.444 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.520 | 0.326 | 0.381 | 0.344 | 0.497 | 0.414 |
| `TensorflowCategorySentimentAnalyzer` | `HuggingBertTokenizer` | 0.488 | 0.311 | 0.379 | 0.414 | 0.543 | 0.427 |

### Precision

#### Recensioni 1* e 5* - `sample_reviews_polar`

| Analyzer | Tokenizer | Precision 1* | Precision 5* | Precision Avg |
|---|---|---:|---:|---|
| `ThreeCheat` | `PlainTokenizer` | - | - | - |
| `ThreeCheat` | `LowercaseTokenizer` | - | - | - |
| `ThreeCheat` | `NLTKWordTokenizer` | - | - | - |
| `ThreeCheat` | `PottsTokenizer` | - | - | - |
| `ThreeCheat` | `PottsTokenizerWithNegation` | - | - | - |
| `ThreeCheat` | `HuggingBertTokenizer` | - | - | - |
| `NLTKSentimentAnalyzer` | `PlainTokenizer` | 0.808 | 0.705 | 0.757 |
| `NLTKSentimentAnalyzer` | `LowercaseTokenizer` | 0.814 | 0.715 | 0.765 |
| `NLTKSentimentAnalyzer` | `NLTKWordTokenizer` | 0.795 | 0.686 | 0.741 |
| `NLTKSentimentAnalyzer` | `PottsTokenizer` | 0.822 | 0.710 | 0.766 |
| `NLTKSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.796 | 0.679 | 0.738 |
| `NLTKSentimentAnalyzer` | `HuggingBertTokenizer` | 0.789 | 0.678 | 0.734 |
| `TensorflowPolarSentimentAnalyzer` | `PlainTokenizer` | 0.845 | 0.870 | 0.858 |
| `TensorflowPolarSentimentAnalyzer` | `LowercaseTokenizer` | 0.864 | 0.877 | 0.871 |
| `TensorflowPolarSentimentAnalyzer` | `NLTKWordTokenizer` | 0.866 | 0.872 | 0.869 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizer` | 0.869 | 0.883 | 0.876 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.874 | 0.896 | 0.885 |
| `TensorflowPolarSentimentAnalyzer` | `HuggingBertTokenizer` | 0.860 | 0.879 | 0.870 |
| `TensorflowCategorySentimentAnalyzer` | `PlainTokenizer` | 0.862 | 0.888 | 0.875 |
| `TensorflowCategorySentimentAnalyzer` | `LowercaseTokenizer` | 0.866 | 0.878 | 0.872 |
| `TensorflowCategorySentimentAnalyzer` | `NLTKWordTokenizer` | 0.870 | 0.879 | 0.874 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizer` | 0.868 | 0.888 | 0.878 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.883 | 0.863 | 0.873 |
| `TensorflowCategorySentimentAnalyzer` | `HuggingBertTokenizer` | 0.864 | 0.872 | 0.868 |

#### Recensioni 1*, 2*, 3*, 4* e 5* - `sample_reviews_varied`

| Analyzer | Tokenizer | Precision 1* | Precision 2* | Precision 3* | Precision 4* | Precision 5* | Precision Avg |
|---|---|---:|---:|---:|---:|---:|---|
| `ThreeCheat` | `PlainTokenizer` | - | - | 0.200 | - | - | 0.200 |
| `ThreeCheat` | `LowercaseTokenizer` | - | - | 0.200 | - | - | 0.200 |
| `ThreeCheat` | `NLTKWordTokenizer` | - | - | 0.200 | - | - | 0.200 |
| `ThreeCheat` | `PottsTokenizer` | - | - | 0.200 | - | - | 0.200 |
| `ThreeCheat` | `PottsTokenizerWithNegation` | - | - | 0.200 | - | - | 0.200 |
| `ThreeCheat` | `HuggingBertTokenizer` | - | - | 0.200 | - | - | 0.200 |
| `NLTKSentimentAnalyzer` | `PlainTokenizer` | 0.487 | 0.341 | 0.294 | 0.352 | 0.305 | 0.356 |
| `NLTKSentimentAnalyzer` | `LowercaseTokenizer` | 0.465 | 0.306 | 0.325 | 0.393 | 0.298 | 0.357 |
| `NLTKSentimentAnalyzer` | `NLTKWordTokenizer` | 0.430 | 0.305 | 0.306 | 0.347 | 0.305 | 0.338 |
| `NLTKSentimentAnalyzer` | `PottsTokenizer` | 0.507 | 0.313 | 0.302 | 0.369 | 0.298 | 0.358 |
| `NLTKSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.463 | 0.294 | 0.293 | 0.333 | 0.305 | 0.338 |
| `NLTKSentimentAnalyzer` | `HuggingBertTokenizer` | 0.490 | 0.318 | 0.328 | 0.357 | 0.279 | 0.354 |
| `TensorflowPolarSentimentAnalyzer` | `PlainTokenizer` | 0.335 | - | - | - | 0.335 | 0.335 |
| `TensorflowPolarSentimentAnalyzer` | `LowercaseTokenizer` | 0.344 | - | - | - | 0.341 | 0.343 |
| `TensorflowPolarSentimentAnalyzer` | `NLTKWordTokenizer` | 0.359 | - | - | - | 0.323 | 0.341 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizer` | 0.338 | - | - | - | 0.352 | 0.345 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.345 | - | - | - | 0.342 | 0.344 |
| `TensorflowPolarSentimentAnalyzer` | `HuggingBertTokenizer` | 0.334 | - | - | - | 0.360 | 0.347 |
| `TensorflowCategorySentimentAnalyzer` | `PlainTokenizer` | 0.549 | 0.326 | 0.301 | 0.360 | 0.574 | 0.422 |
| `TensorflowCategorySentimentAnalyzer` | `LowercaseTokenizer` | 0.555 | 0.395 | 0.348 | 0.347 | 0.569 | 0.443 |
| `TensorflowCategorySentimentAnalyzer` | `NLTKWordTokenizer` | 0.566 | 0.367 | 0.341 | 0.358 | 0.568 | 0.440 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizer` | 0.567 | 0.393 | 0.350 | 0.383 | 0.583 | 0.455 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.537 | 0.356 | 0.362 | 0.323 | 0.539 | 0.424 |
| `TensorflowCategorySentimentAnalyzer` | `HuggingBertTokenizer` | 0.549 | 0.350 | 0.331 | 0.374 | 0.580 | 0.437 |

### F1-measure

Si calcola la F1-measure sui valori medi di recall e precision ottenuti da ciascun modello.

#### Recensioni 1* e 5* - `sample_reviews_polar`

| Analyzer | Tokenizer | F1-measure |
|---|---|---:|
| `ThreeCheat` | `PlainTokenizer` | - |
| `ThreeCheat` | `LowercaseTokenizer` | - |
| `ThreeCheat` | `NLTKWordTokenizer` | - |
| `ThreeCheat` | `PottsTokenizer` | - |
| `ThreeCheat` | `PottsTokenizerWithNegation` | - |
| `ThreeCheat` | `HuggingBertTokenizer` | - |
| `NLTKSentimentAnalyzer` | `PlainTokenizer` | 0.751 |
| `NLTKSentimentAnalyzer` | `LowercaseTokenizer` | 0.760 |
| `NLTKSentimentAnalyzer` | `NLTKWordTokenizer` | 0.735 |
| `NLTKSentimentAnalyzer` | `PottsTokenizer` | 0.760 |
| `NLTKSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.730 |
| `NLTKSentimentAnalyzer` | `HuggingBertTokenizer` | 0.727 |
| `TensorflowPolarSentimentAnalyzer` | `PlainTokenizer` | 0.857 |
| `TensorflowPolarSentimentAnalyzer` | `LowercaseTokenizer` | 0.870 |
| `TensorflowPolarSentimentAnalyzer` | `NLTKWordTokenizer` | 0.868 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizer` | 0.875 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.884 |
| `TensorflowPolarSentimentAnalyzer` | `HuggingBertTokenizer` | 0.869 |
| `TensorflowCategorySentimentAnalyzer` | `PlainTokenizer` | 0.875 |
| `TensorflowCategorySentimentAnalyzer` | `LowercaseTokenizer` | 0.872 |
| `TensorflowCategorySentimentAnalyzer` | `NLTKWordTokenizer` | 0.874 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizer` | 0.878 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.872 |
| `TensorflowCategorySentimentAnalyzer` | `HuggingBertTokenizer` | 0.868 |

#### Recensioni 1*, 2*, 3*, 4* e 5* - `sample_reviews_varied`

| Analyzer | Tokenizer | F1-measure |
|---|---|---:|
| `ThreeCheat` | `PlainTokenizer` | 0.200 |
| `ThreeCheat` | `LowercaseTokenizer` | 0.200 |
| `ThreeCheat` | `NLTKWordTokenizer` | 0.200 |
| `ThreeCheat` | `PottsTokenizer` | 0.200 |
| `ThreeCheat` | `PottsTokenizerWithNegation` | 0.200 |
| `ThreeCheat` | `HuggingBertTokenizer` | 0.200 |
| `NLTKSentimentAnalyzer` | `PlainTokenizer` | 0.348 |
| `NLTKSentimentAnalyzer` | `LowercaseTokenizer` | 0.346 |
| `NLTKSentimentAnalyzer` | `NLTKWordTokenizer` | 0.334 |
| `NLTKSentimentAnalyzer` | `PottsTokenizer` | 0.348 |
| `NLTKSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.332 |
| `NLTKSentimentAnalyzer` | `HuggingBertTokenizer` | 0.337 |
| `TensorflowPolarSentimentAnalyzer` | `PlainTokenizer` | 0.334 |
| `TensorflowPolarSentimentAnalyzer` | `LowercaseTokenizer` | 0.340 |
| `TensorflowPolarSentimentAnalyzer` | `NLTKWordTokenizer` | 0.340 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizer` | 0.344 |
| `TensorflowPolarSentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.342 |
| `TensorflowPolarSentimentAnalyzer` | `HuggingBertTokenizer` | 0.346 |
| `TensorflowCategorySentimentAnalyzer` | `PlainTokenizer` | 0.410 |
| `TensorflowCategorySentimentAnalyzer` | `LowercaseTokenizer` | 0.435 |
| `TensorflowCategorySentimentAnalyzer` | `NLTKWordTokenizer` | 0.433 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizer` | 0.450 |
| `TensorflowCategorySentimentAnalyzer` | `PottsTokenizerWithNegation` | 0.419 |
| `TensorflowCategorySentimentAnalyzer` | `HuggingBertTokenizer` | 0.432 |

### Commento

Tutte le misure effettuate, dallo scarto, all'accuracy, alla F1-measure, indicano come più efficaci i modelli Tensorflow, favorendo leggermente `TensorflowPolarSentimentAnalyzer` quando il dataset è `sample_reviews_polar`, e invece favorendo fortemente `TensorflowCategorySentimentAnalyzer` quando il dataset è `sample_reviews_varied`.

Mantenendo fisso invece l'analyzer, e confrontando tra loro i tokenizer, si nota che non c'è chiarezza su quali siano meglio degli altri; l'unico pattern individuabile è che il `PottsTokenizer` tende ad avere metriche leggermente migliori rispetto agli altri, ottenendo spesso poco più di un punto percentuale di differenza rispetto agli altri tokenizer.

La combinazione migliore pertanto si direbbe quella del `TensorflowCategorySentimentAnalyzer` con il `PottsTokenizer`.


<!--Collegamenti-->

[`jq`]: https://jqlang.github.io/jq/
[`cfig`]: https://cfig.readthedocs.io
[`cfig.Configuration`]: https://cfig.readthedocs.io/en/latest/reference.html#cfig.config.Configuration
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
[Tensorflow]: https://www.tensorflow.org
[`tensorflow.data.Dataset`]: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
[Keras]: https://www.tensorflow.org/api_docs/python/tf/keras
[`tensorflow.Tensor`]: https://www.tensorflow.org/api_docs/python/tf/Tensor
[`tensorflow.keras.layers.StringLookup`]: https://www.tensorflow.org/api_docs/python/tf/keras/layers/StringLookup
[esplosione del gradiente]: https://towardsdatascience.com/the-vanishing-exploding-gradient-problem-in-deep-neural-networks-191358470c11
[`tensorflow.keras.layers.Embedding`]: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
[`tensorflow.keras.layers.Dropout`]: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
[`tensorflow.keras.layers.Dense`]: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
[HuggingFace]: https://huggingface.co
[`bert-base-cased`]: https://huggingface.co/bert-base-cased
