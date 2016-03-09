# Implementierung \label{head:implementierung}

## Kontextdiagramm

Im Kontextdiagramm der Abbildung \ref{fig:kontextdiagramm} sind die Klassenstruktur und deren Abhängigkeiten, ebenso wie die verwendeten und resultierenden Artefakte abgebildet. Das Projekt ist in vier Module unterteilt, welche in Form von *Python*-Dateien abgebildet werden.

![Kontextdiagramm [Hodel] \label{fig:kontextdiagramm}](images/Kontextdiagramm.pdf)

Das Modul *preprocessor.py* dient als Adapter der Originalbilder zum Netzwerk.

Das Modul *metric.py* beinhaltet Klassen, um Trainingsvorgänge in einer *MongoDB* aufzuzeichnen.

Das Modul *network.py* repräsentiert das eigentliche *kNN*. Es beinhaltet die Klasse *Network*, welche Instanzen der Klassen *FullyConnectedLayer* und *AutoencoderLayer* zu einem *kNN* verbindet und diese trainieren kann. Dafür wird die Klasse *BatchProcessor* und auch optional die Klasse *MetricRecorder* verwendet. Ein trainiertes *kNN* wird als *Artefakt* in einer Datei persistent gespeichert.

Das Modul *cleaner.py* beinhaltet die Klasse *Cleaner*, mit welcher ein gespeichertes *kNN* geladen und damit beliebige Bilder bereinigt werden können. Die Klasse *BatchCleaner* macht Verwendung der Klasse *Cleaner* und kann Bilder eines ganzen Ordners bereinigen. Es besteht ebenfalls die Möglichkeit eine komprimierte Datei im von *Kaggle* vorgegebenen Dateiformat zu erstellen. Mit dieser kann das *kNN* anschließend auf *Kaggle* bewertet werden.

## Preprocessor

Das Modul *preprocessor.py* beinhaltet die Klassen *Processor* und *BatchProcessor*.

### Die Klasse Processor

Das Generieren der Subbilder wird mit der *Python*-Klasse *Processor* implementiert. Der Klasse wird der Pfad zum verrauschten Bild, der Pfad zum bereinigten Bild, sowie Parameter für die Anzahl der berücksichtigten Nachbarpixel übergeben.

Über die Methode *get_dataset* können nun die Subbilder als *tupel*, mit den jeweiligen Zielpixeln verbunden, ausgegeben werden. Es sind dafür drei verschiedene Algorithmen implementiert. Dabei ist der erste Algorithmus in reinem *Python*-Code realisiert. Der zweite Algorithmus verwendet spezielle *numpy*-Matrix-Operationen. Die beiden Algorithmen unterscheiden sich erheblich in der Performance, was in Kapitel \ref{head:sliding_window} beschrieben ist. Bei Beiden werden alle Subbilder in sortierter Form als *numpy.ndarray* ausgegeben.

Die dritte Version unterscheidet sich insofern, dass sie zufällig nur ein Subbild und das dazugehörige Zielpixel ausgibt. Es werden diesmal nicht alle Datensätze gleichzeitig ausgegeben. Diese Version ist in der Methode *get_random_patch* implementiert.

### Die Klasse BatchProcessor

Durch das Generieren von Subbildern für jedes Pixel aller zum Trainieren verwendeten Bilder, wird eine Menge an Daten generiert, welche nicht gesamtheitlich im Arbeitsspeicher Platz findet. Somit muss ein Weg gefunden werden, die Daten in kleinere Gruppen zu unterteilen, um iterativ das *kNN* trainieren zu können.

Die Klasse *BatchProcessor* dient genau diesem Zweck. Sie ist als *Python*-Iterator-Klasse implementiert. Dies geschieht nach einer *Python*-Konvention, die besagt, dass eine Klasse, welche über die Methoden *\_\_iter\_\_* und *next* verfügt, automatisch zu einem Iterator wird.

Der Klasse *BatchProcessor* werden die Ordner der verunreinigten sowie der bereinigten Bilder angegeben. Es wird für jedes Bild eine *Processor*-Instanz angelegt. Zusätzlich wird die Größe der Gruppe von Subbildern, welche pro Iteration generiert werden sollen, angegeben (*Batchsize*). Der *BatchProcessor* berechnet darauf, wie viele Gruppen für die Subbilder aller Bilder in den angegebenen Ordnern anhand der Gruppengröße benötigt werden.

Wird der *BatchProcessor* als Iterator verwendet, wird bei jeder Iteration die Methode *next* aufgerufen, welche immer genau so viele Subbilder wie gewünscht zurückgibt.

#### Zufälligkeit

Der *Backpropagation-Algorithmus* verlangt, dass die Trainingsdaten nach jedem Durchlauf zufällig durchmischt werden (siehe Kapitel \ref{head:backprop}). Da nicht alle Trainingsdaten im Arbeitsspeicher Platz finden, werden immer nur die Subbilder einer Gruppe (*Batch*) zusammen durchmischt. Um die Gruppen pro Iteration neu aufzustellen, werden die Bilder, repräsentiert durch *Processor*-Instanzen, nach jeder Iteration neu gemischt. Wie in Kapitel \ref{head:evaluierung} beschrieben, ist die Durchmischung dadurch nicht optimal.

Auf Grund dieser Tatsache ist eine weitere zufällige Variante implementiert worden. Diese wird *Totaler Zufall* genannt. Beim *Totalen Zufall* wählt der *BatchProcessor* zufällig einen *Processor* aus und lässt sich ein Subbild inklusive Zielpixel ausgeben (Aufruf der Methode *get_random_patch*). Dieser Prozess wird so oft wiederholt, bis die gewünschte Gruppengröße erreicht ist.

Beim *Totalen-Zufall* werden die Subbilder bereits bei ihrer Generierung zufällig angeordnet. Dies ist zum einen schneller, da die Durchmischung nicht nachträglich gemacht werden muss, und ermöglicht zum anderen eine zufällige Durchmischung der Subbilder aller Bilder. In Abbildung \ref{fig:batch_vs_fully} wird im Trainingsverlauf der Unterschied sichtbar.

## Netzwerk \label{head:network}

### Verwendetes Material

Der Aufbau der Klassen *Network* und *FullyConnectedLayer* wurde aus der Datei *network3.py*, welche von Michael Nielson zusammen mit seinem Online-Buch "Neural Networks and Deep Learning" [@nielsen_2015] veröffentlicht wurde, übernommen und angepasst. Vor allem wurde daraus der *Stochastic-Gradient-Descent*-Algorithmus, die *L2-Regularisation* sowie der *Dropout*-Mechanismus daraus entnommen.

Hinzugefügt wurden der *RMSprop*-Algorithmus, der *Momentum*-Mechanismus, die Datenübergabe mittels Iterator-Klasse, das early-stopping, die Trainingsveralaufaufzeichung, sowie die gesamte Klasse *AutoencoderLayer*. Der Programmcode des *AutoencoderLayer* wurde überwiegend dem Tutorial "Stacked Denoising Autoencoders (SdA)" der *Theano*-Webseite [@deeplearning.net-2015] entnommen und der Struktur der eingens erstellten Basisklasse *Layer* angepasst.

### Persistente Speicherung

Allen Klassen im Modul *Network* werden die Methoden *\_\_setstate\_\_* und *\_\_getstate\_\_* überschrieben. Dies ist die von *Python* vorgegebene Art festzulegen, welche Attribute wie serialisiert werden sollen. Die Serialisierung wird mit Hilfe der *Python*-Bibliothek *cPickle* durchgeführt.

Es ist nicht unerlässlich diese Methoden zu überschreiben. Wird dies jedoch nicht getan, kann nicht garantiert werden, dass bei einer Weiterentwicklung der Klasse, alte, serialisierte Objekte korrekt geladen werden können. Diese Methoden dienen vor allem als Adapter zur Garantie der Kompatibilität auf Zeit.

### Die Klasse Network

Den Kern des *kNN* bildet die Klasse *Network*. Beim Instanziieren wird ihr eine Liste von Schicht-Klassen mitgegeben. Dadurch werden die Schichten miteinander so verknüpft, dass die Ausgangsneuronen der vorstehenden Schicht zu den Eingangsneuronen der darauffolgenden Schicht werden (siehe Codebeispiel \ref{lst:binding}). Die Klasse *Network* ist damit das Bindeglied modular zusammenstellbarer Schicht-Klassen zu einem Netzwerk.

~~~~~~~{#lst:binding .python caption="Verbinden der Schichten"}
for j in xrange(1, len(self.layers)):
  prev_layer, layer  = self.layers[j-1], self.layers[j]
  layer.set_input(prev_layer.output, prev_layer.output_dropout, self.mbs)
~~~~~~~

Ebenfalls ist in der Klasse *Network* der Trainingsalgorithmus implementiert. Dafür ist die Methode *train* zuständig. Der Methode *train* werden die Trainings- und Validierungsdaten in Form von *BatchProcessor*-Instanzen sowie die Hyperparameter für die Lernrate, *L2-Regularisation* sowie die Art des *Gradientenabstiegsverfahrens* mitgegeben. Optional kann zusätzlich eine Instanz der Klasse *MetricRecorder* mitgegeben werden. Ist dies der Fall, werden nach jeder Validierung die Zwischenergebnisse aufgezeichnet. Der Trainingsablauf ist in Form eines Flussdiagramms in Abbildung \ref{fig:trainingsprozess} grob skizziert.

![Trainingsprozess Implementation [Hodel] \label{fig:trainingsprozess}](images/Training-Flowchart.pdf)

Werden beim Instanziieren Schichten vom Typ *AutoencoderLayer* mitgegeben, können diese mittels der Methode *pretrain_autoencoders* vorausgehend trainiert werden. Diese Methode erkennt automatisch alle *AutoencoderLayer* und ruft bei diesen schrittweise deren Methode *train* auf (siehe Codebeispiel \ref{lst:autoencoder}).

~~~~~~~{#lst:autoencoder .python caption="Schichtweises trainieren der Autoencoder"}
aes = [layer for layer in self.layers
       if isinstance(layer, AutoencoderLayer)]
for index, ae in enumerate(aes):
  ae.train(tdata=tdata, mbs=mbs, eta=eta, epochs=epochs,
           layers=aes[:index], metric_recorder=metric_recorder,
           level=index)
~~~~~~~

Sind mehrere *AutoencoderLayer* vorhanden, handelt es sich um einen *Stacked-Denoising-Autoencoder*. Hier werden der nächsten Schicht alle vorhergehenden Schichten mitgegeben, damit die Trainingsdaten zunächst von den vorgehenden Schichten verarbeitet werden können. Die *Netzwerk* Klasse übernimmt hier abermals die Rolle des Bindeglieds.

\FloatBarrier

### Die Basisklasse Layer

Die Klasse *Layer* dient als Basisklasse für mögliche Schichten. Die wichtigste Methode spielt dabei die Methode *set_inpt*. Diese wird von der Klasse *Network* dazu verwendet, die Schichten miteinander zu verbinden. Zusätzlich werden die Methoden zur Persistenten Speicherung implementiert.

Die Basisklasse dient vor allem zur Übersicht, nicht aber als ein aus statischen Sprachen bekanntes Interface. Fehlt eine benötigte Methode, wird die *Exception* *NotImplementet* aufgerufen.

#### Das Attribut self.params

Beim Trainieren greift die Klasse *Network* über das Attribut *self.params* direkt auf die Gewichte und Bias der Schichtklassen zu. Da in *Python* Instanz-Attribute direkt in der Konstruktor-Methode *\_\_init\_\_* definiert werden, gibt es keine Möglichkeit Instanz-Attribute in einer Basisklasse zu definieren.

### Die Klasse FullyConnectedLayer \label{head:fully-connected}

Die Klasse *FullyConnectedLayer* repräsentiert eine Schicht, bei der alle Ausgangsneuronen der vorgehenden Schicht allen eigenen Eingangsneuronen zugeordnet werden. Dem *FullyConnectedLayer* kann beim Instanziieren die Aktivierungsfunktion sowie die *Dropout* Prozentzahl, *p_dropout*, mitgegeben werden.

Ist der *p_dropout* größer als $0.0$ gewählt, werden beim Trainieren der unsichtbaren Schicht, wie in Kapitel \ref{head:dropout} beschrieben, zufällig diverse Neuronen deaktiviert.

Mit dem Parameter *activation\_fn* kann die Art der Neuronen definiert werden. Zur Auswahl stehen die Aktivierungsfunktionen *sigmoid* und *ReLU*. Diese sind im Modul *network.py* als alleinstehende Funktionen definiert.

#### Gewichte und Bias

Für die Gewichte und Bias werden zwei *Theano-Shared-Variablen* (*self.w* und *self.b*) initialisiert.  Die *Theano-Shared-Variablen* sind Variablen, die zwischen dem *CPU*-Arbeitsspeicher und dem *GPU*-Arbeitsspeicher geteilt und synchronisiert werden. Die Initialwerte der beiden Vektoren werden durch den Algorithmus von Glorot & Bengio [@GlorotBB11], wie in Codebeispiel \ref{lst:init}, gesetzt.

~~~~~~~{#lst:init .python caption="Initialisieren der Gewichte und Bias"}
self.w = tshared(self.rnd.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), 'w')
self.b = tshared(self.rnd.normal(loc=0.0, scale=1.0, size=(n_out,)), 'b')
~~~~~~~

Das Attribut *self.params* ist eine Liste, welche Referenzen auf die Gewichts- und Biasvektoren beinhaltet. Von außen, bzw. von der Klasse *Network*, wird nur über *self.params* auf die Gewichte und Bias zugegriffen.

#### Verknüpfung der Schicht

Die Methode *set_input* dient zur Verknüpfung der jeweiligen Schicht. Die Eingabeverknüpfung wird als Parameter in Form eines symbolischen *Theano-Tensors* des Typs *tensor.dmatrix* übergeben.

Die Eingabeverknüpfung wird im Attribut *self.inpt* gespeichert.
Des Weiteren wird das Attribut *self.output* definiert, in dem die Aktivierungsfunktion (*self.activation\_fn*) auf das Resultat der Eingabefunktion für die Eingabeverknüpfung (*self.inpt*), die Gewichte (*self.w*) und die Bias (*self.b*) angewendet wird. Dies entspricht dem Schema der Gleichung \ref{eq:aktivierungsfunktion} in Kapitel Aktivierungsfunktion \ref{head:aktivierungsfunktion}.

#### Dropout

Zusätzlich zur Ausgabe *self.output* wird das Attribut *self.output_dropout* definiert. Hier wird zuerst die Eingabeverknüpfung, mit einer zufälligen Binominalverteilung maskiert. Somit werden zufällige Eingabeverknüpfungen deaktiviert. Diese maskierte Eingabeverknüpfung wird im Attribut *self.inpt_dropout* gespeichert. Der Parameterwert *p_dropout* definiert, wie viel Prozent der Neuronen deaktiviert werden sollen. Liegt der Wert bei Null, ist die *dropout*-Funktionalität deaktiviert.

~~~~~~~{#lst:dropout .python caption="Maskierung der Layer mit einer Binominalverteilung"}
srng = shared_randomstreams.RandomStreams(rnd.randint(999999))
mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
layer*T.cast(mask, theano.config.floatX)
~~~~~~~

#### Kostenfunktion

Die Kostenfunktion *cost* verwendet die von *Theano* zur Verfügung gestellte Funktion *Theano.Tensor.nnet.binary_crossentropy*, welche auf die Ausgabeverknüpfung *self.output_dropout* und dem Zielvektor (*Network.y*) angewendet wird. Die Kostenfunktion wird in der Methode *train* der Klasse *Network* verwendet. Es wird die Kostenfunktion der Ausgangsschicht, die letzte Schicht-Klasse des Netzes, verwendet.

#### Präzision

Für die Berechnung der Präzision bei der Validierung mit Validierungsdaten, wird nicht die *binary_crossentropy* sondern, wie vom Wettbewerb [@kaggleDDD] vorgeschrieben, der *Root-Mean-Square-Error*, verwendet. Hier wird nun als Ausgabeverknüpfung das Attribut *self.output* verwendet. *Dropout* ist nur beim Training von Relevanz.

Ein wesentliches Detail liegt in der "*Minibatch*-weisen" Bereinigung. Ein Bild hat meistens nicht genau so viele Pixel, dass diese ohne Rest durch die *Minibatchgröße* geteilt werden können. Um trotzdem alle Pixel durch *Minibatch* bereinigen zu können, werden diese mit schwarzen Subbildern ergänzt, welche danach wieder entfernt werden.

### AutoencoderLayer \label{head:autoencoder-layer}

Der *AutoencoderLayer* baut auf dem *FullyConnectedLayer* auf. Es wird jedoch keine Vererbung eingesetzt.

Die wichtigste Eigenschaft vom *AutoencoderLayer* ist, dass dieser zwei "Gesichter" besitzt. Wird der *AutoencoderLayer* in einem Netzwerk verwendet, ist die unsichtbare Schicht gleichzeitig auch die Ausgangsschicht.

Wird der *AutoencoderLayer* durch die eigene Methode *train* trainiert, wird intern der unsichtbaren Schicht eine neue Ausgangsschicht angefügt. Diese besitzt die gleiche Anzahl Neuronen wie die Eingangsschicht. Beim Trainieren werden die Gewichte und der Bias der unsichtbaren Schicht angepasst. Dies ist in Abbildung \ref{fig:stacked-autoencoder} in Kapitel \ref{head:stacked-autoencoder} visualisiert. Der *Stacked-Denoising-Autoencoer* kann durch das Verbinden mehrerer *AutoencoderLayer* erreicht werden.

Weitere Arten wie der *Sparse-Autoencoder* oder *Deep-Autoencoder* wurden aus Zeitgründen nicht umgesetzt.

#### Verrauschen der Eingabeverknüpfung

Das Verrauschen der Eingabeverknüpfung geschieht in gleicher Weise, wie der *Dropout*-Mechanismus. In der Methode *get_corrupted_input* wird der Eingabevektor *self.inpt* mit einer binominal verteilten Maske, bestehend aus Prozentzahlen, multipliziert und zurückgegeben. Dadurch werden die Werte zufällig verändert.

~~~~~~~{#lst:corrupt .python caption="Verrauschen der Eingangsschicht mit einer Binominalverteilung"}
self.theano_rng.binomial(size=self.inpt.shape, n=1,
                         p=1 - self.corruption_level,
                         dtype=theano.config.floatX) * self.inpt
~~~~~~~

#### Berechnung der unsichtbaren Schicht (encode)

Zur Berechnung der unsichtbaren Schicht dient die Methode *get_hidden_values*. Diese verwendet die Aktivierungsfunktion *sigmoid* und wendet diese auf eine übergebene Eingangsverknüpfung an. Bevor die Eingangsverknüpfung übergeben wird, wird diese mit Hilfe der Methode *get_corrupted_input* verrauscht. Dadurch handelt es sich um einen *Denoising-Autoencoder*.

~~~~~~~{#lst:hidden_layer .python caption="Berechnung der unsichtbaren Schicht"}
sigmoid(T.dot(inpt, self.w) + self.b)
~~~~~~~

#### Bereinigen des Inputs während dem Training (decode)

Während dem Trainieren des *AutoencoderLayer* wird die unsichtbare Schicht durch die Methode *get_reconstructed_input* zu einem Ausgabevektor mit gleicher Form des Eingabevektor berechnet. Dazu wird abermals die Aktivierungsfunktion *sigmoid* verwendet. Als Gewichte und Bias werden hierbei jedoch nicht mehr *self.w* und *self.b*, sondern *self.w\_prime* und *self.b\_prime* verwendet, wobei der Gewichtsvektor *self.w\_prime* dem gespiegelten Gewichtsvektor *self.w* entspricht. Dadurch sind *self.w\_prime* und *self.w* hart aneinander gekoppelt. Der Biasvektor *self.b\_prime* ist eigenständig und wird mit Nullwerten initialisiert.

~~~~~~~{#lst:denois .python caption="Decodierung der unsichtbaren Schicht"}
sigmoid(T.dot(hidden, self.w_prime) + self.b_prime)
~~~~~~~

#### Das Training

Der *AutoencoderLayer* kann durch die Methode *train* trainiert werden. Dazu müssen die Trainingsdaten in Form eines *BatchProcessor* übergeben werden. Validierungsdaten sind nicht notwendig, da die Trainingsdaten zur Laufzeit verrauscht werden. Wie bei der Methode *train* der Klasse *Network*, müssen die Epochen-Anzahl, Lernrate und Größe der *Minibatchs* mitgegeben werden. Zusätzlich kann eine Liste von Referenzen zu vorhergehenden *AutoencoderLayer*-Instanzen mitgegeben werden. Ist diese Liste vorhanden, werden die Trainingsdaten zuerst von den vorausgehenden *AutoencoderLayer* mit der Methode *forward* modifiziert.

Nun wird eine symbolische *Theano-Tensor-Matrix-Variable* *x* erstellt, welche sogleich durch die Methode *set_inpt* als Eingabevektor gesetzt wird.
Danach werden die symbolischen Funktionen zur Kostenberechnung und für die Modifikationsregeln der Gewichte und Bias mit der Methode *get_cost_updates* zurückgegeben. Die Methode *get_cost_updates* definiert die Kostenfunktion mit Hilfe der Methoden *get_hidden_values* und *get_reconstructed_input* und der schon in der Klasse *Network* verwendeten *binary_crossentropy*. Die Modifikationsregeln werden durch den *RMSprop*-Algorithmus generiert.

~~~~~~~{#lst:train .python caption="Kompilieren der Theano Funktion zur berechnung der Trainingskosten pro Minibatch"}
cost, updates = self.get_cost_updates(eta=eta)
train_mb = theano.function(
    [index], cost, updates=updates,
    givens={ x: training_x[index * mbs: (index + 1) * mbs] })
~~~~~~~

Mit Hilfe der Modifikationsregeln und der Kostenfunktion wird nun die Trainingsfunktion als kompilierte *Theano*-Funktion definiert. Dabei wird eine *Theano-Shared-Variable* für die Trainingsdaten verwendet. Diese wird mit der symbolischen Variable *x* verbunden. Die Funktion besitzt ebenfalls einen Parameter für den *Minibatch*-Index. So wird aus der *Theano-Shared-Variable* der Trainingsdaten immer nur ein *Minibatch* pro Aufruf verarbeitet. Die Ausgabe der kompilierten Funktion ist der Fehler (Kosten) des aktuell trainierten *Minibatches*.

~~~~~~~{#lst:train_loop .python caption="Iteration durch Trainingsdaten. Modifikation durch vorhergehende Layer. Aktualisieren der Shared-Variable. Trainieren der Minibatches"}
for train_x, _ in tdata:
  for l in layers: train_x = l.forward(train_x)
  training_x.set_value(train_x, borrow=True)
  for batch_index in xrange(n_train_batches):
    c.append(train_mb(batch_index))
~~~~~~~

Als Nächstes wird pro Epoche durch alle Trainingsdaten iteriert, wobei die *Theano-Shared-Variable* kontinuierlich aktualisiert wird. Vor der Aktualisierung werden die Trainingsdaten mit vorhandenen vorhergehenden Schichten bearbeitet. Dann wird für alle *Minibatches* die kompilierte *Theano-Funktion* zum Trainieren aufgerufen. Die Kosten werden aufsummiert und nach jeder Epoche deren Durchschnitt durch den *MetricRecorder* aufgezeichnet.

## Metric

Das Modul *metric.py* dient der automatischen Aufzeichnung und komfortablen, nachträglichen Analyse der Trainingsverläufe.

Zur Konfiguration der Datenbankverbindung und Identifikation der Trainingsvorgänge wird eine *JSON*-Datei verwendet. Die Struktur der Datei ist vom Projekt *Spearmint* übernommen um Kompatibilität herzustellen. *Spearmint* verwendet ebenfalls eine *MongoDB* um Zwischenergebnisse zu speichern. Da *Spearmint* nur die Endresultate, nicht aber den Trainingsverlauf aufzeichnet, wurde die Klasse *MetricRecorder* implementiert, mit welcher zusätzlich der Trainingsverlauf in eine eigene *MongoDB*-Kollektion eingetragen wird.

### MetricRecorder \label{head:metric-recorder}

Dem *MetricRecorder* wird beim Instanziieren der Pfad zur *JSON*-Konfigurationsdatei mitgegeben. Darin ist die Datenbankverbindung sowie der Experimentname eingetragen. Mit Hilfe dieser Information baut der *MetricRecorder* eine Verbindung mit der *MongoDB* auf und erstellt zwei Kollektionen mit den Namen *experiment\_name.metrics* und *experiment\_name.trainings*, wobei *experiment_name* dynamisch dem in der Konfigurationsdatei festgelegten Experimentnamen entspricht. Pro Experiment werden dadurch eigene Kollektionen erstellt. Diese Logik wurde ebenfalls aus dem Projekt *Spearmint* übernommen. Dieses erstellt die Kollektionen *experiment_name.jobs* und *experiment_name.hypers* für die Hyperparametersuche.

Die Kollektion *experiment\_name.trainings* enthält pro Trainingsgang einen Eintrag mit der gesamten Konfiguration des *kNN* und den Hyperparameter zum Trainieren. Dieser kann durch die Methode *record\_training\_info* gespeichert werden.

Der *MetricRecorder* startet bei der Instanziierung automatisch einen Timer. Wird nun ein neuer Messpunkt durch das Aufrufen der Methode *record* erfasst, werden die Trainings- und Validierungskosten, die vergangene Zeit sowie die aktuelle Iteration und Epoche in die Kollektion *experiment_name.metrics* geschrieben.

Nach einem Trainingsgang beinhaltet die Kollektion *experiment\_name.trainings* einen Eintrag mit einer eineindeutigen *job\_id*, unter welchen mehrere Messpunkte in der Kollektion *experiment_name.metrics* bestehen.

#### Vorteil der automatischen Aufzeichnung

Dadurch, dass der *MetricRecorder* automatisch eine Identifikation generiert und die Konfiguration des Netzwerks sowie die Trainingsverläufe aufzeichnet, können beliebige *kNN*-Konfigurationen trainiert werden, ohne dass bei jedem neuen Trainingslauf darauf geachtet werden muss, dass die Parameterkonfigurationen aufgeschrieben werden. Dies bewirkt, dass keine Daten verloren gehen.

#### Spezifikation der Kollektionen

---------------------   ----------------------------------------------------
_id                     MongoDB-Id

job_id                  Identifikator des Trainingsgangs

algorithmus             Art des Trainingsalgorithmus (sgd, rmsprop)

dropouts                Liste mit den Dropoutwerten für jeden Layer

eta                     Startwert der linear absteigenden Lernrate

eta_min                 Endwert der linear absteigenden Lernrate

layers                  String, welcher die Layerkonfiguration angibt

lambda                  L2-Regularisation Koeffizient

mini_batch_size         Größe der Minibatches

momentum                Momentum Koeffizient

improvement_threshold   Prozentzahl, wie viel besser die Validierung sein
                        muss, um die Geduld zu erhöhen (early-stopping)

patience_increase       Multiplikator für die Erhöhung der Geduld
                        (early-stopping)

training_data           Anzahl der Trainingsdaten

validation_data         Anzahl der Validierungsdaten

validation_frequency    Anzahl der Validierungen pro Epoche
--------------------    ----------------------------------------------------

  : Kollektion experiment_name.trainings \label{table:trainings}

---------------------   ----------------------------------------------------
_id                     MongoDB-Id

job_id                  Identifikation des Trainingsverlaufs

cost                    Trainingskosten

validation_accuracy     Validierungskosten

epoch                   Aktuelle Epoche

eta                     Verwendete Lernrate

iteration               Aktuelle Iteration (Minibatch)

second                  Vergangene Sekunden bis zur Aufzeichnung
---------------------   ----------------------------------------------------

  : Kollektion experiment_name.metrics \label{table:metrics}

### MetricPlayer \label{head:metric-player}

Die Klasse *MetricPlayer* dient zur komfortablen Auswertung und Visualisierung der aufgezeichneten Trainingsgänge. Bei der Instanziierung wird dieselbe *JSON*-Datei wie beim *MetricRecorder* und *Speramint* mitgegeben.

Mit der Methode *get_records* können die Trainingsdaten in Form eines *Pandas DataFrame* zurückgegeben werden.

Des Weiteren kann mit der Methode *plot* direkt ein Trainingsgang visualisiert werden. Mit der Methode *compair_plot* können mehrere *job_id*s mitgegeben werden, welche gemeinsam visualisiert ausgegeben werden.

Alle Visualisierungen werden mit Hilfe der *Python*-Bibliothek *matplotlib* erstellt. Die in Kapitel \ref{head:evaluierung} beschriebenen Plots sind alle mit der Klasse *MetricPlayer* erstellt worden.

#### Vorteil beim Auswerten

Dadurch, dass die Trainingsverläufe in einer *MongoDB* aufgezeichnet werden, können auf dem eigenen Laptop mit Hilfe des *ipython notebooks* [@ipython-notebook] und der Klasse *MetricPlayer* die Trainingsverläufe bereits während dem Training visualisiert werden, obwohl das Training auf dem Server *deepgreen02* an der HTW-Berlin durchgeführt wird. Hierzu wird wiederholt auf die Abbildung \ref{fig:training-kontext} verwiesen.

## Cleaner \label{head:cleaner}

Im Modul *cleaner.py* wird der Bereinigungsprozess implementiert. Dafür werden die Module *network.py* und *preprocessor.py* vorausgesetzt.

### Die Klasse Cleaner

Der Klasse *Cleaner* muss bei der Instanziierung der Pfad zu einer durch *cPickle* serialisierten Instanz der Klasse *Network* mitgegeben werden, welche sogleich geladen wird. Von der geladenen *Network* Instanz wird die Metainformation über die Eingangsgröße des *kNN* ausgelesen. Diese ist wichtig, damit das zu bereinigende Bild in korrekt große Subbilder unterteilt wird.

Die Klasse *Cleaner* stellt die Methoden *clean*, *clean_and_show*, *clean_and_save* sowie *to_submission_format* zur Verfügung. Wird *clean* mit dem Pfad zum verunreinigten Bild aufgerufen, wird das Bild mit einer *Processor*-Instanz geladen und in die Subbilder unterteilt. Die Subbilder befinden sich in sortierter Reihenfolge. Diese Subbilder werden nun durch die Methode *predict* der *Netwerk*-Instanz, in gleicher Reihenfolge, in bereinigte Pixel umgewandelt. In diesem Vorgang wird aus jedem Subbild ein Pixel des bereinigten Bildes. Diese Pixel werden nun wieder zu einem vollständigen Bild zusammengefügt und ausgegeben.

Die Methoden *clean_and_save* sowie *clean_and_show* verwenden beide die Methode *clean* und erweitern diese zum einfachen Anzeigen oder Abspeichern des bereinigten Bildes.

### Die Klasse BatchCleaner \label{head:batch-cleaner}

Die Klasse *BatchCleaner* verwendet die Klasse *Cleaner* um damit Bilder eines Ordners bereinigen zu können. Beim Instanziieren muss ein Pfad zum Ordner mit den verunreinigten Bildern, sowie der Pfad zur serialisierten Modelldatei angegeben werden. Danach können über die zur Verfügung gestellten Methoden *clean_and_save* sowie *clean_for_submission* alle Bilder bereinigt und abgespeichert oder bereinigt und im vom *Kaggle-Wettbewerb* vorgegebenen Format zur Einreichung ausgegeben werden.
