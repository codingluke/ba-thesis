## Vor- und Nachverarbeitung der Bilder \label{head:vor-nachverarbeitung}

### Subbilder generieren

Die Trainings- und Testdaten bestehen aus verschieden großen Bilder. Um ein Bild als Ganzes von einem *kNN* bereinigen zu lassen, müsste die Eingangsschicht aus den Graustrufenwerte der einzelnen Pixel des Bildes bestehen. Dadurch definiert die Bildgröße auch die Eingangsgröße des *kNN*.

Der Aufbau eines *kNN* kann nachträglich nicht verändert werden. Dies bedeutet, dass in diesem Falle für jedes Bild mit einer anderen Bildgröße ein eigenes *kNN* zugeschnitten werden muss.

Um dieses Problem zu lösen, wird einen Vorverarbeitungsschritt eingeführt. Die Bilder werden nicht als Ganzes bereinigt, sondern werden zuerst in gleich große Subbilder anhand der Abbildung \ref{fig:sliding-window} unterteilt. Dieses Verfahren wurde in der Arbeit [@cleaning-handwritten-data2005] für die Handschriftbereinigung vorgestellt.

![Bereinigung eines verrauschten Pixel, mit Hilfe der Nachbarpixel \label{fig:sliding-window}](images/sliding-window.png)

In dem Verfahren wird für jedes Pixel vom verrauschten Bilde ein Subbild generiert. Dieses Subbild beinhaltet eine Anzahl Nachbarpixel, welche zur Bereinigung hilfreich sind. Auf die Frage, wie viele Nachbarpixel verwendet werden, wird im Kapitel \ref{evaluierung} eingegangen.

Um ebenfalls die Randpixel bereinigen zu können, wird das Bild vor dem Prozess mit einem schwarzen Rand erweitert. Der Rand hat dieselbe Größe der Anzahl Nachbarn.

Es werden somit nur die lokalen Information der Nachbarpixel verwendet, um ein bereinigter Pixelwert vorauszusagen. Dies macht dadurch sind, da Informationen außerhalb eines Wortes, geschweige denn am Ende des Bildes, keine Relevanz haben, im Bezug zu einem Pixel eines Buchstaben. Dies schlägt die Brücke zur Theorie hinter den *convolutional neural networks*, wenn auch diese nicht direkt in der Arbeit Verwendung finden. Dieses Verfahren wird in der Arbeit "Gradient-based learning applied to document recognition" [@Lecun98gradient-basedlearning] im Kapitel "Convolutional neural network for isolated character recognition" von Le Cun beschrieben.

### Generieren der Subbilder

Das Generieren der Subbilder wird mit der *Python* Klasse *ImgPreprocessor* Implementiert. Der Klasse wird der Pfad zum verrauschten Bild, der Pfad zum bereinigten Bild sowie Parameter für die Anzahl der berücksichtigten Nachbarn übergeben.

Nun können die Subbilder als *tupel* mit den jeweiligen Zielpixel ausgegeben werden. Es wurden drei verschiedene Algorithmen implementiert. Dabei ist der erste mit reinem *Python* code realisiert. Der zweite Algorithmus verwendet *numpy*Matrix-Index-Operationen. Diese beiden Algorithmen unterscheiden sich erheblich in der Performance was im Kapitel \ref{head:sliding_window} beschrieben ist. Es werden jedoch bei beiden alle Subbilder in sortierter Form als *Array* ausgegeben.

Die dritte Version unterscheidet sich insofern, dass sie zufällig nur ein Subbild und das dazugehörige Zielpixel ausgibt.

### Verwenden der Subbilder

Durch das Generieren von Subbilder für jedes Pixel aller Bilder, wird eine Menge an Daten generiert welche nicht gesamt im Arbeitsspeicher Platz findet. Somit muss einen Weg gefunden werden, die Daten in kleinere Gruppen zu unterteilen, um danach iterativ das *kNN* trainieren zu können.

Um dies zu realisieren, wurde eine *Python* Iterator-Klasse *BatchImgPreprocessor*, welche die Klasse *ImgPreprocessor* verwendet, implementiert.

Der Klasse *BatchImgPreprocessor* werden die Ordner der verunreinigten sowie der bereinigten Bilder angegeben. Dadurch erstellt er für jedes Bild einen *ImgPreprocessor*. Zusätzlich wird die Größe der Gruppe von Subbilder, welche pro Iteration generiert werden sollen angegeben. Der *BatchImgPreprocessor* berechnet darauf, wie viele Gruppen für die Subbilder aller Bilder in den angegebenen Ordner anhand der Gruppengröße benötigt werden.

Wird der *BatchImgPreprocessor* nun als Iterator in einer Schleife verwendet, werden bei jeder Iteration immer genau so viele Subbilder wie gewünscht zurückgegeben. Dem *kNN* werden als Datensätze zwei *BatchImgPreprocessor*-Instanzen mitgegeben. Einen für die Trainingsdaten und einer für die Validierungsdaten.

**Zufälligkeit**

Der *Backpropagation-Algorithmus* verlangt, dass die Trainingsdaten nach jedem Durchlauf wieder zufällig Angeordnet werden. Da nicht alle Trainingsdaten im Arbeitsspeicher Platz haben, werden immer nur die Subbilder in einer Gruppe auf zusammen gemischt. Um die Gruppen pro Iteration neu aufzustellen, werden die Bilder, repräsentiert durch *ImgPreprocessor*-Instanzen nach jeder Iteration neu gemischt. Wie im Kapitel \ref{head:evaluation} zu sehen ist, ist die Durchmischung dadurch nicht optimal.

Es wird auf Grund dieser Tatsache eine weitere Zufällige Variante implementiert. Sie wird in der Arbeit *Totaler Zufall* genannt. Beim *Totalen Zufall* wählt der *BatchImgPreprocessor* zufällig ein *ImgPreprocessor* aus und lässt sich ein Subbild inklusive Zielpixel ausgeben. Diesen Prozess wird so oft wiederholt, bis die gewünschte Gruppengröße erreicht ist.

Dadurch werden die Subbilder bereits bei der Generierung zufällig angeordnet, was zum einen schneller ist, da die Durchmischung nicht nachträglich gemacht werden muss, und zum anderen eine zufällige Druchmischung über alle Subbilder aller Bilder ermöglicht. In der Abbildung \ref{fig:batch_vs_fully} wird Trainingsverlauf der Unterschied sichtbar.

### Entrauschtes Bild zusammensetzen

### Ähnlichkeit zu convolutional Netzen

### Online vs. Offline verfahren

## Trainings- und Validationsdaten Unterteilung

Die wesentlichen Eigenschaften, welche vom kNN gelernt werden sollen sind die Schriftbilder der verwendeten Schriftarten und Stiele. Die von Kaggle zur Verfügung gestellten Testdaten verwenden exakt die selben Schriften und Stiele wie die Trainingsdaten. Die Testdaten unterscheiden sich durch verschiedene Hintergründe, Sättigung der Schrift, sowie einem anderen Text.

Es muss also darauf geachtet werden, dass die Test- und Validierungsdaten gleiche Schriftbilder enthalten. Das kNN soll neuen Schmutz erkennen und beseitigen, nicht aber neue Schriftbilder ableiten können.

Beim Analysieren der Trainingsdaten ist aufgefallen, dass zwei verschiedene Texte Verwendung finden. Diese Texte existieren beide exakt gleich oft in allen Schriftvariationen. Begünstigt wird dies dadurch dass die Hintergründe ebenfalls leicht abweichen.

Somit werden die Trainingsdaten in Trainings- und Validierungsdaten 50:50, anhand der verschiedenen Texte aufgeteilt. Dies soll eine möglichst reale Präzision beim Validieren ermöglichen.

Um bei Kaggel trotz der Hälfte der Trainingsdaten gut abzuschneiden, wird das Beste eigene Modell erneut mit den selben Hyperparameter, Schichtkombinationen und Epochenanzahl auf allen Trainingsdaten trainiert. ^[Alternativ könnte ein neues Modell gleicher Konfiguration auf den Validierungsdaten trainiert werden und mit dem Bestehenden zu einem Ensemble kombiniert werden.]

**Kleinere Datenbasis für schnellere Hyperparametersuche**

Anhand der oben beschriebenen Methodik wird ebenfalls eine noch kleinere Datenbasis zusammengestellt. Darauf können schneller und effizienter verschiedene Kombinationen von Hyperparameter trainiert und validiert werden. Dahinter verbirgt sich die von Aristoteles eingeführte Induktionslogik, mit welche vom Speziellen (kleinen) auf das Allgemeine (Große) geschlossen wird.

### Dekalration der Datenbasis

**Mittlere Datenbasis**

- Trainingsdaten
  - 72 Bilder
  - 24 Bilder (540x258)
  - 48 Bilder (540x420)
  - 8 Hintergründe
  - 1 Text
  - Alle Schriften und Stiele

- Validierungsdaten
  - 72 Bilder
  - 24 Bilder (540x258)
  - 48 Bilder (540x420)
  - 6 Unterschiedliche Hintergründe
  - 2 Auch in den Trainingsdaten vorhandene Hintergründe
  - 1 Text unterschiedlich zu den Trainingsdaten
  - Alle Schriften und Stiele gleich

**Kleine Datenbasis**

