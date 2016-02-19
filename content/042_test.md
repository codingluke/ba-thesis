# Test Driven Development

Das Testen wird nach dem *TDD* Prinzip, *Test Driven Development*, umgesetzt. Die Tests werden in *TDD* nicht am Ende hinzugefügt, sondern direkt während der Implementation erstellt. Anstatt in der *Python* Konsole einzelne Funktionalitäten manuell zu testen, und dadurch häufig dieselben "Snippets" aus zu führen, geschieht dies direkt in Form von Unittests. Dieses Vorgehen spart Zeit, dient als Dokumentation und vereinfacht die spätere Weiterentwicklung und das Refactoring.

## Python Unittests und Struktur

Die Tests werden mit Hilfe der *Python* Bibliothek *unittest* umgesetzt. Im Hauptverzeichnis befindet sich die Datei *tests.py*, welche alle vorhandenen Tests ladet und ausführt.

Für jede Moduldatei im Ordner "/src" befindet sich eine zugehörige Testdatei im Ordner "/tests". Als Namenskonvention gilt, dass die Testdatei denselben Namen der Moduldatei besitzt und dieser "test\_" vorstellt. Als Beispiel besteht für die Moduldatei "src/network.py" die Testdatei "tests/test_network.py"

Jeder Klasse der Moduldatei wird eine TestKlasse in der Testdatei zugewiesen. Der Name der Testklasse wird aus den Wörter "Test" und dem Namen der zu Testenden Klasse zusammengesetzt. Der Name der Testklasse für die Klasse "MetricRecorder" lautet demnach "TestMetricRecorder".

### Konfigurationsmöglichkeit

Standardmäßig werden immer alle Tests geprüft. Es besteht die Möglichkeit, in einer Konfigurationsdatei das Ausführungsverhalten zu kontrollieren. So können langsame Tests und Tests welche eine Datenbankanfrage stellen ausgestellt werden um während der Entwicklung spezifischer Funktionalitäten zeit zu sparen.

### Daten

Die Dateien welche von den Tests verwendet werden befinden sich im Ordner "/test/data". Die Kaggle Test und Validierungsdaten dürfen nicht weitergegeben werden und befinden sich deshalb nicht in diesem Ordner. Die Kaggel Daten müssen selbst von Kaggle.com heruntergeladen werden.

Um Kaggle Daten in den Tests zu verwenden. Z.B zum Testen der Klassen *Network*und *Clener*, wird die Konfigurationsvariable "data_dir_path" in der Datei "/tests/config.py" zur Verfügung gestellt. Diese muss den relativen Pfad von der Datei "tests.py" zum Ordner mit den Kaggle Daten beinhalten. Diese Variable wird in den Tests verwendet, um auf Kaggle Daten zuzugreifen.

## Preprocessor

Am meisten Tests bestehen für die Klassen *ImgPreprocessor* und *BatchImgPreprocessor*. Es wird vor allem überprüft, ob die generierten Subbilder und die zu Prüfenden Zielpixel auch übereinstimmen. Zudem wird geprüft ob die Zufälligkeit gegeben ist.

Beim *BatchImgPreprocessor* wird getestet ob die Länge der Iteration korrekt ist, so dass auch wirklich über die gesamten Daten iteriert wird.

Es besteht ebenfalls einen Benchmark Test, welcher die drei verschiedenen Implementierungsartren zur Subbildgenerierung auf Performance vergleicht. Dier Test wird deaktiviert, wenn in der Konfigurationsdatei "/tests/config.py" die Variable "slow" auf "False" gesetzt wird, da er sehr viel Zeit in Anspruch nimmt. Das Resultat kann im Kapitel \ref{head:evaluation} entnommen werden.

## Metric

Beim Modul *Metric* wird überprüft ob die Verbindung zur Datenbank erstellt werden kann und ob die Einträge korrekt geschrieben und gelesen werden können. Diese Tests können in der Konfigurationsdatei "tests/config.py" deaktiviert werden, sofern gerade keine Datenbank zur Verfügung steht.

## Network

Zum Testen des Moduls *Network*, wird ein Netzwerk konfiguriert, welches über alle verfügbaren *Layer* verfügt und mit sehr wenig Daten trainiert. Die *AutoencoderLayer* werden ebenfalls voraus trainiert.

Damit handelt es sich um Integrations- und nicht um Unittests. Die Tests dienen zur Sicherstellung, dass alle Schichten korrekt implementiert sind und ebenfalls als Vorlage für die Verwendung des Moduls *Network*. Um sicher zu stellen, dass das Netzwerk trainiert müssen die Validierungskosten kleiner als 1 sein.

Die Tests garantieren nicht, dass die Algorithmen korrekt implementiert wurden, nur dass das Netz mit Trainingsdaten trainiert werden kann.

## Cleaner

Das Modul *Cleaner* wird getestet, indem ein gespeichertes, trainiertes Netzwerk geladen und ein Testbild der Kaggle Daten bereinigt und angezeigt wird. So ist vom Auge aus ersichtlich ob das Bereinigen funktioniert.

Zudem wird beim *BatchCleaner* überprüft, ob die generierte Datei zur Einreichung auch dem von Kaggle definierten Format vorliegt. Damit wird sichergestellt, dass die generierte Einreichung auch korrekt formatiert ist.

Auch diese Tests dienen zusätzlich als Vorlage zur Verwendung des *Cleaner* und *BatchCleaner*.
