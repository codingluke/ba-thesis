# Test Driven Development \label{head:tdd}

Das Testen wurde nach dem *TDD*-Prinzip, *Test Driven Development*, umgesetzt. Die Tests werden beim *TDD* nicht am Ende hinzugefügt, sondern direkt während der Implementation erstellt. Anstatt in der *Python*-Konsole einzelne Funktionalitäten durch manuelles Eintippen zu testen, und sich dadurch häufig zu wiederholen, geschieht dies in Form von *Unittests*. Dieses Vorgehen spart Zeit, dient als Dokumentation und vereinfacht die spätere Weiterentwicklung sowie das Refactoring. Alle in diesem Kapitel verwendeten Pfadangaben sind als relativ zum Projektordner zu betrachten.

## Python Unittests und Struktur

Die Tests werden mit Hilfe der *Python*-Bibliothek *unittest* umgesetzt. Im Hauptverzeichnis befindet sich die Datei */tests.py*, welche alle vorhandenen Tests lädt und ausführt. Die Tests können somit durch den Befehl *python tests.py* ausgeführt werden.

Für jede Moduldatei im Ordner */src/* befindet sich eine zugehörige Testdatei im Ordner */tests/*. Als Namenskonvention besteht der Name der Testdatei aus dem Namen der Moduldatei, welchem der Präfix *test\_* voranstellt wird. Als Beispiel existiert für die Moduldatei */src/network.py* die zugehörige Testdatei */tests/test_network.py*.

Jeder in der Moduldatei definierten Klasse, ist eine Test-Klasse in der Testdatei zugewiesen. Die Namensgebung der Test-Klasse basiert auf ähnlicher Logik wie bei der Test-Datei. Dem Test-Klassennamen wird der Präfix *Test* vorangestellt. Der Name der Test-Klasse für die Klasse *MetricRecorder* lautet demnach *TestMetricRecorder* welche sich in der Datei */tests/test_mertric.py* befindet.

### Konfigurationsmöglichkeit

Standardmäßig werden immer alle Tests ausgeführt. Es besteht jedoch die Möglichkeit, mit der Datei */tests/config.py* das Ausführungsverhalten zu verändern. So können langsame Tests und Tests, welche eine Datenbankanfrage stellen, deaktiviert werden, um während der Entwicklung spezifischer Funktionalitäten Zeit zu sparen.

### Daten

Eigene Dateien, welche von den Tests verwendet werden, befinden sich im Ordner */test/data/*. Die Wettbewerb-Daten dürfen nicht weitergegeben werden und befinden sich deshalb nicht in diesem Ordner, sowie auf der CD zu dieser Arbeit. Diese müssen selbst von der Webseite zum Wettbewerb heruntergeladen werden.

Um Wettbewerb-Daten in den Tests zu verwenden, wird die Variable *data_dir_path* in der Konfigurationsdatei */tests/config.py* zur Verfügung gestellt. Dieser muss den relativen Pfad von der Datei */tests.py* zum Datenordner zugewiesen werden. In den Tests wird ausschließlich über die Variabel *data_dir_path* auf diese Daten zugegriffen.

## Preprocessor

Die meisten Tests bestehen für die Klassen *Processor* und *BatchProcessor*. Es wird vor allem überprüft, ob die generierten Subbilder und die zu prüfenden Zielpixel auch übereinstimmen. Zudem wird geprüft, ob die Zufälligkeit gegeben ist.

Beim *BatchProcessor* wird getestet, ob die Anzahl der Iterationen korrekt ist, sodass garantiert über die gesamten Daten iteriert werden kann.

Es besteht ebenfalls einen Benchmark-Test, welcher die drei umgesetzten Algorithmen zur Subbildgenerierung auf Performance vergleicht. Der Test kann in der Konfigurationsdatei deaktiviert werden, da er sehr viel Zeit in Anspruch nimmt. Dafür muss die Variable *slow* auf *False* gesetzt werden. Das Resultat kann in Kapitel \ref{head:evaluierung} entnommen werden.

## Metrik

Beim Modul *metric.py* wird überprüft, ob die Verbindung zur Datenbank erstellt werden kann und ob die Einträge korrekt geschrieben und gelesen werden können. Diese Tests können in der Konfigurationsdatei, durch das Setzen der Variable *mongodb* auf den Wert *False*, deaktiviert werden, sofern gerade keine Datenbank zur Verfügung steht.

## Network

Zum Testen des Moduls *network.py*, wird ein *kNN* der Klasse *Network* instanziiert, welches über die Schichten *FullyConnectedLayer* und *AutoencoderLayer*, verfügt und mit einem sehr kleine Datensatz trainiert wird. Die *AutoencoderLayer* werden ebenfalls im Voraus trainiert.

Damit handelt es sich um Integrations- und nicht um *Unittests*. Die Tests dienen zur Sicherstellung, dass alle Schichten korrekt implementiert sind und ebenfalls als Vorlage für die Verwendung des Moduls *network.py*. Um sicher zu stellen, dass das *kNN* trainiert, müssen die Validierungskosten kleiner als 1 sein.

**Die Tests garantieren nicht, dass die Algorithmen korrekt implementiert wurden**, aber, dass das Netz mit Trainingsdaten trainiert werden kann, ohne dass ein Fehler auftaucht.

Um die Algorithmen testen zu können, bräuchte es vorgegebene Test- und Validierungsdaten, welchen für jeden Algorithmus genaue Endresultate zugewiesen wurden. Da dies nicht der Fall ist, können die Algorithmen auch nicht auf Korrektheit getestet werden. Gewissermaßen werden im Kapitel \ref{head:evaluierung} die Algorithmen, durch das erfolgreiche Bereinigen der Bilder, getestet.

## Cleaner

Das Modul *cleaner.py* wird getestet, indem ein trainiertes und gespeichertes *kNN* geladen und damit ein Testbild bereinigt und angezeigt wird. So ist vom Auge aus ersichtlich, ob das Bereinigen funktioniert.

Zudem wird beim *BatchCleaner* überprüft, ob die generierte Datei zur Einreichung auch in dem vom Wettbewerb vorgegebenen Format vorliegt. Damit wird sichergestellt, dass die generierte Einreichung auch korrekt formatiert ist.

Auch diese Tests dienen zusätzlich als Vorlage zur Verwendung der Klassen *Cleaner* und *BatchCleaner*.
