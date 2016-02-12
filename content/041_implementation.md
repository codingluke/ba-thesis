# Implementierung

## Kontextdiagramm

![Kontextdiagramm \label{fig:kontextdiagramm}](images/kontextdiagramm.png)

Im Kontextdiagramm der Abbildung \ref{fig:kontextdiagramm} ist die Klassenstruktur und deren Abhängigkeiten abgebildet. Das Projekt ist in vier Verantwortungsbereiche Unterteilt, welche direkt in *Python* Modulen abgebildet werden.

Der *preprocessor* dient als Adapter der Originalbilder zum Netzwerk.

Das Modul *metric* beinhaltet Klassen um Trainingsvorgänge in eine *MongoDB* aufzunehmen.

Das Modul *network* räpresentiert das eigentliche kNN. Es beinhaltet die Klasse *Network* welche Instanzen der Klassen *FullyConnectedLayer* und *AutoencoderLayer* zu einem kNN verbindet und diese trainieren kann. Es macht Gebrauch vom Preprocessor und auch optional vom *MetricRecorder*. Ein trainiertes kNN kann als *Artefakt* persisten serialisiert werden.

Das Modul *cleaner* kann ein als Datei serialisiertes kNN laden, und damit beliebige Bilder bereinigen. Mit dem *BatchCleaner* können direkt Bilder eines ganze Ordners bereinigt werden. Es besteht ebenfalls die Möglichkeit eine komprimierte Datei im von *Kaggle* vorgegebenen Dateiformat zu erstellen. Mit dieser kann das kNN direkt auf *Kaggle* Bewertet werden.

## Preprocessor

### Klasse ImgPreprocessor

Das Generieren der Subbilder wird mit der *Python* Klasse *ImgPreprocessor* Implementiert. Der Klasse wird der Pfad zum verrauschten Bild, der Pfad zum bereinigten Bild sowie Parameter für die Anzahl der berücksichtigten Nachbarn übergeben.

Nun können die Subbilder als *tupel* mit den jeweiligen Zielpixel ausgegeben werden. Es wurden drei verschiedene Algorithmen implementiert. Dabei ist der erste in reinem *Python*-Code realisiert. Der zweite Algorithmus verwendet *numpy*-Matrix-Operationen. Diese beiden Algorithmen unterscheiden sich erheblich in der Performance was im Kapitel \ref{head:sliding_window} beschrieben ist. Es werden bei beiden alle Subbilder in sortierter Form als *Array* ausgegeben.

Die dritte Version unterscheidet sich insofern, dass sie zufällig nur ein Subbild und das dazugehörige Zielpixel ausgibt.

### Klasse BatchImgPreprocessor

Durch das Generieren von Subbilder für jedes Pixel aller Bilder, wird eine Menge an Daten generiert welche nicht gesamt im Arbeitsspeicher Platz findet. Somit muss einen Weg gefunden werden, die Daten in kleinere Gruppen zu unterteilen, um iterativ das *kNN* trainieren zu können.

Um dies zu realisieren, wurde eine *Python* Iterator-Klasse *BatchImgPreprocessor* implementiert. Diese verwendet die Klasse *ImgPreprocessor* für die Subbildgenerierung und ermöglicht das Generieren von Subbilder von Bilder eines gesamten Ordners.

Der Klasse *BatchImgPreprocessor* werden die Ordner der verunreinigten sowie der bereinigten Bilder angegeben. Es wird für jedes Bild eine *ImgPreprocessor*-Instanz angelegt. Zusätzlich wird die Größe der Gruppe von Subbilder, welche pro Iteration generiert werden sollen angegeben (Batchsize). Der *BatchImgPreprocessor* berechnet darauf, wie viele Gruppen für die Subbilder aller Bilder in den angegebenen Ordner anhand der Gruppengröße benötigt werden.

Wird der *BatchImgPreprocessor* als Iterator in einer For-Schleife verwendet, werden bei jeder Iteration immer genau so viele Subbilder wie gewünscht zurückgegeben.

**Zufälligkeit**

Der *Backpropagation-Algorithmus* verlangt, dass die Trainingsdaten nach jedem Durchlauf zufällig durchmischt werden. Da nicht alle Trainingsdaten im Arbeitsspeicher Platz haben, werden immer nur die Subbilder einer Gruppe zusammen durchmischt. Um die Gruppen pro Iteration neu aufzustellen, werden die Bilder, repräsentiert durch *ImgPreprocessor*-Instanzen, nach jeder Iteration neu gemischt. Wie im Kapitel \ref{head:evaluation} zu sehen ist, ist die Durchmischung dadurch nicht optimal.

Es wird auf Grund dieser Tatsache eine weitere Zufällige Variante implementiert. Sie wird in der Arbeit *Totaler Zufall* genannt. Beim *Totalen Zufall* wählt der *BatchImgPreprocessor* zufällig ein *ImgPreprocessor* aus und lässt sich ein Subbild inklusive Zielpixel ausgeben. Diesen Prozess wird so oft wiederholt, bis die gewünschte Gruppengröße erreicht ist.

Dadurch werden die Subbilder bereits bei ihrer Generierung zufällig angeordnet, was zum Einen schneller ist, da die Durchmischung nicht nachträglich gemacht werden muss, und zum Anderen eine zufällige Durchmischung über alle Subbilder aller Bilder ermöglicht. In der Abbildung \ref{fig:batch_vs_fully} wird Trainingsverlauf der Unterschied sichtbar.

## Netzwerk \label{head:network}

### Persistente Speicherung

Allen Klassen im Modul *Network* werden die Methoden *__setstate__* und *__getstate__* überschrieben. Dies ist die von *Python* vorgegebene Art festzulegen welche Attribute, wie Serialisiert werden sollen.

Es ist nicht unerlässlich diese Methoden zu überschreiben, wird dies jedoch nicht getan, kann nicht garantiert werden, dass bei einer Weiterentwicklung der Klasse, alte Serialisierte Objekte korrekt geladen werden können. Diese Methoden dienen vor allem als Adapter zur Garantie der Kompatibilität auf Zeit.

### Die Klasse Network

Der Kern des kNN, bildet die Klasse *Network*. Beim Instanziieren wird ihr eine Liste von Schicht-Klassen mitgegeben. Dadurch werden die Schichten miteinander so verknüpft, dass die Ausgangsneuronen der vorstehenden Schicht zu den Eingangsneuronen der darauffolgenden Schicht werden. Die Klasse *Network* ist damit das Bindeglied für Modular zusammenstellbaren Schicht-Klassen.

Ebenfalls ist die Klasse *Network* der Trainingsalgorithmus implementiert. Dafür ist die Methode *train* zuständig. Der Methode *train* werden die Trainings- und Validierungsdaten in Form von *BatchImgPreprocessor*-Instanzen sowie die Hyperparameter für die Lernrate, L2-Regularisation und auch die Art des Gradientenabstiegsverfahrens mitgegeben. Optional, kann auch eine Instanz der Klasse *MetricRecorder* mitgegeben werden. Ist dies der Fall, werden nach jeder Validierung die Zwischenergebnisse aufgezeichnet.

Werden bei der Instanziierung Schichten vom Typ *AutoencoderLayer* mitgegeben, können diese mit der Methode *pretrain_autoencoders* vorausgehend trainiert werden. Die Methode erkennt automatisch alle *AutoencoderLayer* und ruft bei diesen Schrittweise deren Methode *train* auf. Sind mehrere *AutoencoderLayer* vorhanden handelt es sich um einen *Stacked Autoencoder*. Hier werden der nächsten Schicht alle vorhergehenden Schichten mitgegeben, damit die Trainingsdaten zuerst von den vorgehenden Schichten verarbeitet werden können. Die *Netzwerk* Klasse übernimmt hier wieder das Bindeglied.

### Abstrakte Klasse Layer

Die abstrakte Klasse *Layer* definiert das Interface für mögliche Schichten. Die wichtigste Methode spielt dabei die Methode *set_inpt*. Diese wird von der Klasse *Network* dazu verwendet, die Schichten miteinander zu verbinden.

Beim Trainieren greift die Klasse *Network* über das Attribut *self.params* direkt auf die Gewichte und Biase der Layerklassen zu. Da in *Python* Instanz Attribute direkt in der Konstruktor-Methode *__init__* definiert werden, gibt es keine Möglichkeit abstrakte Instanz Attribute zu definieren.

Die Abstrakte Klasse dient vor allem zur Übersicht, nicht aber als ein von statischen Sprachen bekanntes Interface.

### FullyConnectedLayer \label{head:fully-connected}

Der *FullyConnectedLayer* bildet die Standardschicht. Sie repräsentiert eine Schicht bei der alle Ausgangsneuronen der vorgehenden Schicht allen eigenen Eingangsneuronen zugeordnet werden. Dem FullyConnectedLayer kann beim Instanziieren die Aktivierungsfunktion (activation\_fn) sowie eine *Dropout* Prozentzahl (p\_dropout) mitgegeben werden.

Ist der *Dropout* Prozentsatz größer als $0.0$ gesetzt, wird beim Trainieren der unsichtbaren Schicht wie im Kapitel \ref{head:dropout} beschrieben zufällig diverse Neuronen deaktiviert.

Mit dem Parameter activation\_fn kann die Art der Neuronen definiert werden.

### AutoencoderLayer \label{head:autoencoder-layer}

Der *AutoencoderLayer* baut auf dem *FullyConnectedLayer* auf. Es wird jedoch keine Vererbung eingesetzt.

Die wichtigste Eigenschaft vom *AutoencoderLayer* ist, dass dieser zwei *Gesichter* besitzt. Wird der *AutoencoderLayer* in einem Netzwerk verwendet, ist unsichtbare Schicht gleichzeitig die Ausgangsschicht.

Wird der *AutoencoderLayer* durch die eigene Methode *train* trainiert, wird intern der unsichtbaren Schicht eine neue Ausgangsschicht angefügt. Diese besitzt die gleiche Anzahl Neuronen wie die Eingangsschicht. Beim Trainieren werden die Gewichte und der Bias der unsichtbaren Schicht angepasst.

Dadurch ist der *AutoencoderLayer* ein *FullyConnectedLayer*, welcher unabhängig vom Netzwerk, in welchem er sich befinden, im voraus trainiert werden kann. Bei dem vorhergehende Trainieren handelt es sich um unbeaufsichtigtes Trainieren zur besseren Initialisierung der Gewichte und Biase.

## Metrik

Das Modul *metric* dient zur automatischer Aufzeichnung und komfortablen, nachträglicher Analyse der Trainings- sowie Validierungsverläufe.

Zur Konfiguration der Datenbankverbindung und Identifikation der Trainingsvorgänge dient eine JSON Datei. Die Struktur der Datei wurde vom Projekt *Spearmint* übernommen um Kompatibilität herzustellen. *Spearmint* verwendet ebenfalls eine *MongoDB* um Zwischenergebnisse zu speichern. Da *Spearmint* nur die Endresultate, nicht aber den Trainingsverlauf aufzeichnet, wurde die Klasse *MetricRecorder* implementiert, mit welche zusätzlich der Trainingsverlauf in eine eigene *MongoDB* Kollektion eingetragen werden.

### MetricRecorder \label{head:metric-recorder}

Dem *MetricRecorder* wird beim Instanziieren eine Pfad zu einer *JSON* Konfigurationsdatei mitgegeben. Darin wird die Datenbankverbindung sowie einen Experimentnamen gespeichert. Mit Hilfe dieser Information baut der *MetricRecorder* eine Verbindung mit der *MongoDB* auf und erstellt zwei Kollektionen mit den Namen "experiment\_name.metrics" und "experiment\_name.trainings".

Die Kollektion "experiment\_name.trainings" einthält pro Trainingsgang einen Eintrage in welchem die gesamte Konfiguration des Netzwerks und den Trainings Hyperparameter abgespeichert wird. Diese kann durch die Methode "record\_training\_info" gesetzt werden.

Der *MetricRecorder* startet bei der Instanziierung automatisch einen Timer. Wird nun einen neuen Messpunkt durch das Aufrufen der Methode *record* erfasst, werden die Trainings- und Validierungskosten, die vergangene Zeit sowie die aktuelle Iteration und Epoche in die Kollektion "experiment_name.metrics" geschrieben.

Nach einem Trainingsgang beinhaltet die Kollektion "experiment\_name.trainings" mit einer eineindeutigen "job\_id" unter welchen mehrere Messpunkte in der Kollektion "experiment_name.metrics" bestehen.

**Vorteil der automatischen Aufzeichnung**

Dadurch, dass der *MetricRecorder* automatisch eine Identifikation generiert und die Konfiguration des Netzwerks sowie die Trainingsverläufe aufzeichnet, können beliebige Netze Trainiert werden, ohne dass bei jedem neuen Training darauf geachtet werden muss, dass die Trainingsläufe und Parameterkonfigurationen aufgeschrieben werden. Dies ist extrem Hilfreich, damit keine Daten verloren gehen.

### MetricPlayer \label{head:metric-player}

Die Klasse *MetricPlayer* dient zur komfortablen Auswertung der augezeichneten Trainings- und Validierungskosten.

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

Wird *clean* mit dem Pfad zum verunreinigten Bild aufgerufen, wird das Bild mit einer *ImgPreprocessor*-Instanz geladen und in die Subbilder unterteilt. Die Subbilder befinden sich in sortierter Reihenfolge. Diese Subbilder werden nun durch die Methode *predict* der *Netwerk*-Instanz, in gleicher Reihenfolge, in bereinigte Pixel umgewandelt. In diesem Vorgang wird aus jedem Subbild ein Pixel des bereinigten Bildes. Diese Pixel, werden nun wieder zu einem vollständigen Bild zusammengefügt und ausgegeben.

Die Methoden *clean_and_save* sowie *clean_and_show* verwenden beide die Methode *clean* und erweitern diese zum einfachen anzeigen oder abspeichern des bereinigten Bildes.

### BatchCleaner \label{head:batch-cleaner}

Die Klasse *BatchCleaner* verwendet die Klasse *Cleaner* um damit Bilder eines Ordners bereinigen zu können.

## Unittests \label{head:unittests}

Für alle Klassen werden Unittests zur Sicherstellung der Funktionalität erstellt. Dabei wird nach dem *TDD* Prinzip, *Test Driven Development*, vorgegangen. Die Tests werden nicht am Ende hinzugefügt, sondern direkt wärend der Implementation erstellt. Anstatt in der *Python* Konsole einzelne Funktionalitäten manuell zu testen, geschieht dies direkt in Form von Unittests. Dieses Vorgehen spart Zeit, dient als Dokumentation und vereinfacht die spätere Weiterentwicklung und das Refactoring.

