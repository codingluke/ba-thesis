# Implementierung

## Kontextdiagramm


Im Kontextdiagramm der Abbildung \ref{fig:kontextdiagramm} sind die Klassenstruktur und deren Abhängigkeiten, ebenso wie die resultierenden Artefakte abgebildet. Das Projekt ist in vier Module unterteilt, welche in Form von *Python*-Dateien abgebildet werden.

![Kontextdiagramm \label{fig:kontextdiagramm}](images/kontextdiagramm.png)

Das Modul *preprocessor.py* dient als Adapter der Originalbilder zum Netzwerk.

Das Modul *metric.py* beinhaltet Klassen um Trainingsvorgänge in einer *MongoDB* aufzuzeichnen.

Das Modul *network.py* repräsentiert das eigentliche kNN. Es beinhaltet die Klasse *Network*, welche Instanzen der Klassen *FullyConnectedLayer* und *AutoencoderLayer* zu einem kNN verbindet und diese trainieren kann. Dafür wird die Klasse *BatchProcessor* und auch optional die Klasse *MetricRecorder* verwendet. Ein trainiertes kNN wird als *Artefakt* in einer Datei persistent gespeichert.

Das Modul *cleaner.py* beinhaltet die Klasse *Cleaner* mit welcher ein gespeichertes kNN geladen und damit beliebige Bilder bereinigt werden können. Die Klasse *BatchCleaner* macht Verwendung der Klasse *Cleaner* und kann Bilder eines ganze Ordners bereinigen. Es besteht ebenfalls die Möglichkeit eine komprimierte Datei im von *Kaggle* vorgegebenen Dateiformat zu erstellen. Mit dieser kann das kNN anschließend auf *Kaggle* Bewertet werden.

## Preprocessor

### Klasse Processor

Das Generieren der Subbilder wird mit der *Python*-Klasse *Processor* Implementiert. Der Klasse wird der Pfad zum verrauschten Bild, der Pfad zum bereinigten Bild, sowie Parameter für die Anzahl der berücksichtigten Nachbarpixel übergeben.

Über die Methode *get_dataset* können nun die Subbilder als *tupel*, mit den jeweiligen Zielpixel verbunden, ausgegeben werden. Es sind drei verschiedene Algorithmen implementiert. Dabei ist der erste Algorithmus in reinem *Python*-Code realisiert. Der zweite Algorithmus verwendet spezielle *numpy*-Matrix-Operationen. Die beiden Algorithmen unterscheiden sich erheblich in der Performance was im Kapitel \ref{head:sliding_window} beschrieben ist. Bei beiden werden alle Subbilder in sortierter Form als *numpy.ndarray* ausgegeben.

Die dritte Version unterscheidet sich insofern, dass sie zufällig nur ein Subbild und das dazugehörige Zielpixel ausgibt. Es werden also nicht alle Datensätze ausgegeben. Diese ist über die Methode *get_random_patch* implementiert.

### Klasse BatchProcessor

Durch das Generieren von Subbilder für jedes Pixel aller zum Trainieren verwendete Bilder, wird eine Menge an Daten generiert, welche nicht gesamt im Arbeitsspeicher Platz findet. Somit muss einen Weg gefunden werden, die Daten in kleinere Gruppen zu unterteilen um iterativ das *kNN* trainieren zu können.

Die Klasse *BatchProcessor* dient genau diesem Zweck. Sie ist als *Python*-Iterator-Klasse implementiert. Dies geschieht nach einer *Python*-Konvention welche besagt, dass eine Klasse welche über die Methoden *\_\_iter\_\_* und *next* verfügt, automatisch zu einem Iterator wird.

Der Klasse *BatchProcessor* werden die Ordner der verunreinigten sowie der bereinigten Bilder angegeben. Es wird für jedes Bild eine *Processor*-Instanz angelegt. Zusätzlich wird die Größe der Gruppe von Subbilder, welche pro Iteration generiert werden sollen angegeben (Batchsize). Der *BatchProcessor* berechnet darauf, wie viele Gruppen für die Subbilder aller Bilder in den angegebenen Ordner anhand der Gruppengröße benötigt werden.

Wird der *BatchProcessor* als Iterator verwendet, werden bei jeder Iteration die Methode *\_\_next\_\_* aufgerufen, welche immer genau so viele Subbilder wie gewünscht zurück gibt.

**Zufälligkeit**

Der *Backpropagation-Algorithmus* verlangt, dass die Trainingsdaten nach jedem Durchlauf zufällig durchmischt werden. Da nicht alle Trainingsdaten im Arbeitsspeicher Platz finden, werden immer nur die Subbilder einer Gruppe (*Batch*) zusammen durchmischt. Um die Gruppen pro Iteration neu aufzustellen, werden die Bilder, repräsentiert durch *Processor*-Instanzen, nach jeder Iteration neu gemischt. Wie im Kapitel \ref{head:evaluation} zu sehen ist, ist die Durchmischung dadurch nicht optimal.

Es ist auf Grund dieser Tatsache eine weitere Zufällige Variante implementiert. Diese wird *Totaler Zufall* genannt. Beim *Totalen Zufall* wählt der *BatchProcessor* zufällig ein *Processor* aus und lässt sich ein Subbild inklusive Zielpixel ausgeben (Aufruf der Methode *get_random_patch*). Diesen Prozess wird so oft wiederholt, bis die gewünschte Gruppengröße erreicht ist.

Beim *Totalen-Zufall* werden die Subbilder bereits bei ihrer Generierung zufällig angeordnet. Dies ist zum einen schneller, da die Durchmischung nicht nachträglich gemacht werden muss, und ermöglicht zum anderen eine zufällige Durchmischung über alle Subbilder aller Bilder. In Abbildung \ref{fig:batch_vs_fully} wird im Trainingsverlauf der Unterschied sichtbar.

## Netzwerk \label{head:network}

### Persistente Speicherung

Allen Klassen im Modul *Network* werden die Methoden *\_\_setstate\_\_* und *\_\_getstate\_\_* überschrieben. Dies ist die von *Python* vorgegebene Art festzulegen welche Attribute, wie Serialisiert werden sollen. Die Serialisierung wird mit Hilfe der *Python*-Bibliothek *cPickle* durchgeführt.

Es ist nicht unerlässlich diese Methoden zu überschreiben, wird dies jedoch nicht getan, kann nicht garantiert werden, dass bei einer Weiterentwicklung der Klasse, alte, serialisierte Objekte korrekt geladen werden können. Diese Methoden dienen vor allem als Adapter zur Garantie der Kompatibilität auf Zeit.

### Die Klasse Network

Der Kern des kNN, bildet die Klasse *Network*. Beim Instanziieren wird ihr eine Liste von Schicht-Klassen mitgegeben. Dadurch werden die Schichten miteinander so verknüpft, dass die Ausgangsneuronen der vorstehenden Schicht zu den Eingangsneuronen der darauffolgenden Schicht werden. Die Klasse *Network* ist damit das Bindeglied für Modular zusammenstellbaren Schicht-Klassen zu einem Netzwerk.

Ebenfalls ist die Klasse *Network* der Trainingsalgorithmus implementiert. Dafür ist die Methode *train* zuständig. Der Methode *train* werden die Trainings- und Validierungsdaten in Form von *BatchProcessor*-Instanzen sowie die Hyperparameter für die Lernrate, L2-Regularisation und auch die Art des Gradientenabstiegsverfahrens mitgegeben. Optional, kann auch eine Instanz der Klasse *MetricRecorder* mitgegeben werden. Ist dies der Fall, werden nach jeder Validierung die Zwischenergebnisse aufgezeichnet. Der Trainingsablauf ist in Form eines Flussdiagramms in Abbildung \ref{fig:trainingsprozess} grob skizziert.

![Trainingsprozess Implementation \label{fig:trainingsprozess}](images/trainings-flowchart.png)

Werden bei beim instanziieren Schichten vom Typ *AutoencoderLayer* mitgegeben, können diese mit der Methode *pretrain_autoencoders* vorausgehend trainiert werden. Die Methode erkennt automatisch alle *AutoencoderLayer* und ruft bei diesen Schrittweise deren Methode *train* auf. Sind mehrere *AutoencoderLayer* vorhanden handelt es sich um einen *Stacked-Autoencoder*. Hier werden der nächsten Schicht alle vorhergehenden Schichten mitgegeben, damit die Trainingsdaten zuerst von den vorgehenden Schichten verarbeitet werden können. Die *Netzwerk* Klasse übernimmt hier abermals die Rolle des Bindeglieds.

\FloatBarrier

### Basisklasse Layer

Die Klasse *Layer* dient als Basisklasse für mögliche Schichten. Die wichtigste Methode spielt dabei die Methode *set_inpt*. Diese wird von der Klasse *Network* dazu verwendet, die Schichten miteinander zu verbinden.

Beim Trainieren greift die Klasse *Network* über das Attribut *self.params* direkt auf die Gewichte und Bias der Schichtklassen zu. Da in *Python* Instanz-Attribute direkt in der Konstruktor-Methode *\_\_init\_\_* definiert werden, gibt es keine Möglichkeit Instanz-Attribute in einer Basisklasse zu definieren.

Die Basisklasse dient vor allem zur Übersicht, nicht aber als ein von statischen Sprachen bekanntes Interface.

### Klasse FullyConnectedLayer \label{head:fully-connected}

Die Klasse *FullyConnectedLayer* repräsentiert eine Schicht bei der alle Ausgangsneuronen der vorgehenden Schicht allen eigenen Eingangsneuronen zugeordnet werden. Dem FullyConnectedLayer kann beim instanziieren die Aktivierungsfunktion sowie eine *Dropout* Prozentzahl mitgegeben werden.

Ist der *Dropout* Prozentsatz größer als $0.0$ gesetzt, wird beim Trainieren der unsichtbaren Schicht wie im Kapitel \ref{head:dropout} beschrieben zufällig diverse Neuronen deaktiviert.

Mit dem Parameter *activation\_fn* kann die Art der Neuronen definiert werden. Zur Auswahl stehen die Aktivierungsfunktionen *Sigmoid* und *ReLU*.

**Gewichte und Bias**

Für die Gewichte und Bias werden zwei *Theano-Shared-Variablen* (*self.w* und *self.b*) initialisiert.  Die *Theano-Shared-Variablen* sind Variablen, zwischen dem CPU-Arbeitsspeicher und dem GPU-Arbeitsspeicher geteilt und synchronisiert werden. Die Initialwerte der beiden Vektoren werden durch die den Algorithmus von Glorot & Bengio [@GlorotBB11] gesetzt.

Das Attribut *self.params* ist eine Liste, welche Referenzen auf die Gewichts- und Biasvektoren beinhaltet. Von außen, bzw. von der Klasse *Network*, wird nur über *self.params* auf die Gewichte und Bias zugegriffen.

**Verknüpfung der Schicht**

Die Methode *set_input* dient zur Verknüpfung des *Layer*. Die Eingabeverknüpfung wird als Parameter in Form eines symboischen *Theano-Tensors* des Typs *Tensor.dmatrix* übergeben.

Die Eingabeverknüpfung wird nun im Attribut *self.inpt* gespeichert.
Des weiteren wird das Attribut *self.output* definiert, in dem die Aktivierungsfunktion (*self.activation\_fn*) mit dem Resultat der Eingabefunktion für die Eingabeverknüpfung (*self.inpt*), die Gewichte (*self.w*) und die Bias (*self.b*) aufgerufen wird. Dies entspricht dem Schema der Gleichung \ref{eq:aktivierungsfunktion} im Kapitel Aktivierungsfunktion \ref{head:aktivierungsfunktion}.

**Dropout**

Zusätzlich zur Ausgabe *self.output* wird das Attribut *self.output_dropout* definiert. Hier wird zuerst die Eingabeverknüpfung, mit einer zufälligen Binominalverteilung maskiert. Somit werden zufällige Eingabeverknüpfungen deaktiviert. Diese maskierte Eingabeverknüpfung wird im Attribut *self.inpt_dropout* gespeichert. Der Parameterwert *p_dropout* definiert, wie viele Prozent der Neuronen deaktiviert werden sollen. Liegt der Wert bei Null, ist die *dropout*-Funktionalität deaktiviert.

Nun wird analog zu *self.output* das Attribut *self.output_dropout* definiert. Nur wird dieses Mal die maskierte Eingabeverknüpfung verwendet.

**Kostenfunktion**

Als Kostenfunktion *cost* verwendet die von *Theano* zur Verfügung gestellten Funktion *Theano.Tensor.nnet.binary_crossentropy*. Die *binary_crossentropy* Funktion wird auf die Ausgabeverknüpfung *self.output_dropout* angewendet und vergleicht diese mit dem Zielvektor (*Network.y*). Die Kostenfunktion wird in der Methode *train* der Klasse *Network* verwendet. Dabei spielt nur die Kostenfunktion der letzten Schicht, eine Rolle.

**Präzision**

Für die Berechnung der Präzision bei der Validierung mit Validierungsdaten, wird nicht die *binary_crossentropy* sondern, wie vom Kaggle Wettbewerb vorgeschrieben, der *root-mean-square-error* verwendet. Hier wird nun als Ausgabeverknüpfung das Attribut *self.output* verwendet. *Dropout* ist nur beim Training von Relevanz. Wie die Kostenfunktion wird diese ebenfalls beim Trainieren von der Klasse *Network* verwendet und wird nur in der letzen Schicht im Netzwerk benötigt.

### AutoencoderLayer \label{head:autoencoder-layer}

Der *AutoencoderLayer* baut auf dem *FullyConnectedLayer* auf. Es wird jedoch keine Vererbung eingesetzt.

Die wichtigste Eigenschaft vom *AutoencoderLayer* ist, dass dieser zwei *Gesichter* besitzt. Wird der *AutoencoderLayer* in einem Netzwerk verwendet, ist die unsichtbare Schicht gleichzeitig auch die Ausgangsschicht.

Wird der *AutoencoderLayer* durch die eigene Methode *train* trainiert, wird intern der unsichtbaren Schicht eine neue Ausgangsschicht angefügt. Diese besitzt die gleiche Anzahl Neuronen wie die Eingangsschicht. Beim Trainieren werden die Gewichte und der Bias der unsichtbaren Schicht angepasst.

Dadurch ist der *AutoencoderLayer* ein *FullyConnectedLayer*, welcher unabhängig vom Netzwerk, in welchem er sich befinden, im voraus trainiert werden kann. Bei dem vorhergehende Trainieren handelt es sich um unbeaufsichtigtes Trainieren zur besseren Initialisierung der Gewichte und Bias.

Weitere Arten wie der *Sparse-Autoencoder* oder *Deep-Autoencoder* wurden nicht umgesetzt. Der *Stacked-Autoencoder* kann durch das Verbinden mehrerer *AutoencoderLayer* erreicht werden.

**Verrauschen (denoising) der Eingabeverknüpfung**

Das Verrauschen der Eingabeverknüpfung geschieht auf gleicher Art, wie der Dropout Mechanismus. In der Methode *get_corrupted_input* wird dem Eingabevektor *self.inpt* mit einer binominal verteilten Maske, bestehend aus Prozentzahlen, multipliziert und zurückgegeben. Dadurch werden die Werte zufällig verändert.

**Berechnung der unsichtbaren Schicht**

Zur Berechnung der unsichtbaren Schicht dient die Methode *get_hidden_values* Diese Verwendet die Aktivierungsfunktion *Sigmoid* und wendet diese auf eine übergebene Eingangsverknüpfung (z.B. der zuvor verrauschten) an.

**Bereinigen des Inputs wärend dem Training**

Während dem Trainieren des *AutoencoderLayer* wird die unsichtbare Schicht durch die Methode *get_reconstructed_input* zu einem Ausgabevektor mit leichter Form des Eingabevektor berechnet. Dazu wird die Aktivierungsfunktion *Sigmoid* verwendet. Als Gewichte und Bias werden hier nicht mehr *self.w* und *self.b*, sondern *self.w\_prime* und *self.b\_prime* verwendet, wobei der Gewichtsvektor *self.w\_prime* der gespiegelte Gewichtsvektor *self.w* ist. Dadurch sind *self.w\_prime* und *self.w* hart aneinander gekoppelt. Der Biasvektor *self.b\_prime* ist eigenständig und wird mit Nullwerten initialisiert.

**Das Training**

Der *AutoencoderLayer* kann durch die Methode *train* trainiert werden. Dazu müssen die Trainingsdaten in Form eines *BatchProcessor* übergeben werden. Validierungsdaten sind nicht notwendig, da die Trainingsdaten zur laufzeit verrauscht werden. Wie bei der Methode *train* der Klasse *Network* müssen die Epochen-Anzahl, Lernrate und Größe der *Minibatchs* mitgegeben werden. Zusätzlich kann eine Liste von Referenzen zu vorhergehenden *AutoencoderLayer*-Instanzen mitgegeben werden. Ist diese Liste vorhanden, werden die Trainingsdaten zuerst von den vorausgehenden *AutoencoderLayer* mit der Methode *forward* bearbeitet.

Nun wird eine symbolische *Theano-Tensor-Matrix-Variable* *x* erstellt, welche sogleich durch die Methode *set_inpt* als Eingabevektor gesetzt wird.
Danach werden die symbolischen Funktionen zur Kostenberechnung und für die Modifikationsregeln der Gewichte und Bias mit der Methode *get_cost_updates* zurückgegeben. Die Methode *get_cost_updates* definiert die Kostenfunktion mit Hilfe der Methoden *get_hidden_values* und *get_reconstructed_input* und der schon in der Klasse *Network* verwendeten *binary_crossentropy*. Die Modifikationsregeln werden durch den *rmsprop*-Algorithmus generiert.

Mit Hilfe der Modifikationsregeln und der Kostenfunktion wird nun die Trainingsfunktion als kompilierte Theano-Funktion definiert. Dabei wird eine *Theano-Shared-Variable* für die Trainingsdaten verwendet. Diese wird mit der symbolischen Variable *x* verbunden. Die Ausgabe der Funktion sind die Kosten und die Aktualisierungsregeln die Modifikationsregeln. Die Funktion besitzt ebenfalls einen Parameter für den Minibatch Index. So werden aus der Theano-Shared-Variable der Trainingsdaten immer nur einen *Minibatch* pro aufruf verarbeitet.

Nun wird pro Epoche durch alle Trainingsdaten iteriert, wobei die *Theano-Shared-Variable* kontinuierlich aktualisiert wird. Vor der Aktualisierung werden die Trainingsdaten mit vorhandenen vorhergehenden Schichten bearbeitet. Dann wird für alle Minibatchs die kompilierte *Theano-Funktion* zum Trainieren aufgerufen. Die kosten werden aufsummiert und nach jeder Epoche deren Durchschnitt durch den *MetricRecorder* aufgezeichnet.

## Metric

Das Modul *metric.py* dient zur automatischer Aufzeichnung und komfortablen, nachträglicher Analyse der Trainingsverläufe.

Zur Konfiguration der Datenbankverbindung und Identifikation der Trainingsvorgänge dient eine JSON Datei. Die Struktur der Datei wurde vom Projekt *Spearmint* übernommen um Kompatibilität herzustellen. *Spearmint* verwendet ebenfalls eine *MongoDB* um Zwischenergebnisse zu speichern. Da *Spearmint* nur die Endresultate, nicht aber den Trainingsverlauf aufzeichnet, wurde die Klasse *MetricRecorder* implementiert, mit welche zusätzlich der Trainingsverlauf in eine eigene *MongoDB* Kollektion eingetragen werden.

### MetricRecorder \label{head:metric-recorder}

Dem *MetricRecorder* wird beim Instanziieren eine Pfad zu einer *JSON* Konfigurationsdatei mitgegeben. Darin wird die Datenbankverbindung sowie einen Experimentnamen gespeichert. Mit Hilfe dieser Information baut der *MetricRecorder* eine Verbindung mit der *MongoDB* auf und erstellt zwei Kollektionen mit den Namen "experiment\_name.metrics" und "experiment\_name.trainings".

Die Kollektion "experiment\_name.trainings" einthält pro Trainingsgang einen Eintrage in welchem die gesamte Konfiguration des Netzwerks und den Trainings Hyperparameter abgespeichert wird. Diese kann durch die Methode "record\_training\_info" gesetzt werden.

Der *MetricRecorder* startet bei der Instanziierung automatisch einen Timer. Wird nun einen neuen Messpunkt durch das Aufrufen der Methode *record* erfasst, werden die Trainings- und Validierungskosten, die vergangene Zeit sowie die aktuelle Iteration und Epoche in die Kollektion "experiment_name.metrics" geschrieben.

Nach einem Trainingsgang beinhaltet die Kollektion "experiment\_name.trainings" mit einer eineindeutigen "job\_id" unter welchen mehrere Messpunkte in der Kollektion "experiment_name.metrics" bestehen.

**Vorteil der automatischen Aufzeichnung**

Dadurch, dass der *MetricRecorder* automatisch eine Identifikation generiert und die Konfiguration des Netzwerks sowie die Trainingsverläufe aufzeichnet, können beliebige Netze Trainiert werden, ohne dass bei jedem neuen Training darauf geachtet werden muss, dass die Trainingsläufe und Parameterkonfigurationen aufgeschrieben werden. Dies ist extrem Hilfreich, damit keine Daten verloren gehen.

**Kollektionen**

Aufzeichnung der Struktur.

### MetricPlayer \label{head:metric-player}

Die Klasse *MetricPlayer* dient zur komfortablen Auswertung der aufgezeichneten Trainingsgänge.

Bei der Instanziierung wird dieselbe *JSON*-Datei wie beim *MetricRecorder* und *Speramint* mitgegeben.

Mit der Methode *get_records* können die Trainingsdaten in Form eines *Pandas DataFrame*  zurückgegeben werden.

Des weiteren kann mit der Methode *plot* direkt ein Trainingsgang visualisiert werden. Mit der Methode *compair_plot* können mehrere job Identifikatoren mitgegeben werden, welche gemeinsam in einem Plot visualisiert ausgegeben werden.

Die Plots im Kapitel \ref{head:evaluation} wurden alle mit dem *MetricPlayer* generiert.

**Vorteil beim Auswerten**

Dadurch, dass die Trainingsverläufe in einer *MongoDB* aufgezeichnet werden, kann auf dem eigenen Laptop mit Hilfe des *ipython notebooks* und der Klasse *MetricPlayer* die Trainingsverläufe bereits während dem Training visualisiert werden, obwohl das Training auf dem Server *deepgreen02* an der HTW-Berlin durchgeführt wird.

## Cleaner \label{head:cleaner}

Im Modul *cleaner* wird der Bereinigungsprozess Implementiert. Dafür werden die Module *network* und *preprocessor* benötigt.

### Klasse Cleaner

Der Klasse *Cleaner* muss bei der Instanziieren den Pfad zu einer durch *cPickle* serialisierten Instanz der Klasse *Network* mitgegeben werden.

So wird eine Instanz der Klasse *Network* mit dem gespeichertem Status generiert. Von der geladenen *Network* Instanz wird die Metainformation über die Eingangsgröße des kNN ausgelesen. Diese ist wichtig, damit später die zu bereinigenden Bilder in korrekt große Subbilder unterteilt werden können.

Die Klasse *Cleaner* stellt die Methoden *clean*, *clean_and_show*, *clean_and_save* sowie *to_submission_format* zur Verfügung.

Wird *clean* mit dem Pfad zum verunreinigten Bild aufgerufen, wird das Bild mit einer *Processor*-Instanz geladen und in die Subbilder unterteilt. Die Subbilder befinden sich in sortierter Reihenfolge. Diese Subbilder werden nun durch die Methode *predict* der *Netwerk*-Instanz, in gleicher Reihenfolge, in bereinigte Pixel umgewandelt. In diesem Vorgang wird aus jedem Subbild ein Pixel des bereinigten Bildes. Diese Pixel, werden nun wieder zu einem vollständigen Bild zusammengefügt und ausgegeben.

Die Methoden *clean_and_save* sowie *clean_and_show* verwenden beide die Methode *clean* und erweitern diese zum einfachen anzeigen oder abspeichern des bereinigten Bildes.

### BatchCleaner \label{head:batch-cleaner}

Die Klasse *BatchCleaner* verwendet die Klasse *Cleaner* um damit Bilder eines Ordners bereinigen zu können.


