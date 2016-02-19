# Analyse und Systementwurf

Im Kapitel Analyse, werden die nötigen Prozesse zur Umsetzung der Anforderungen analysiert und beschrieben. Es bestehen zwei Hauptprozesse, welche sich teilweise überschneiden. Diese wären der Bereinigungsprozess eines Bildes sowie der Trainingsprozess des kNN. Darüber hinaus werden unterstützende Vorgehensweisen zur Hyperparemetersuche und zur Konfiguration der Architektur des kNN beschrieben.

## Bereinigungsprozess

Die Trainings- und Testdaten bestehen aus verschieden großen Bildern. Um ein Bild als Ganzes von einem *kNN* bereinigen zu lassen, müsste die Eingangsschicht aus den Graustrufenwerten der einzelnen Pixel des Bildes bestehen. Dadurch definiert die Bildgröße auch die Eingangsgröße des *kNN*.

Der Aufbau eines *kNN* kann nachträglich nicht verändert werden. Dies bedeutet, dass in diesem Falle für jedes Bild mit einer anderen Bildgröße ein eigenes *kNN* zugeschnitten werden muss.

Um dieses Problem zu lösen, wird ein Vorverarbeitungsschritt eingeführt. Die Bilder werden nicht als Ganzes bereinigt, sondern werden zuerst in gleich große Subbilder (wie in Abbildung \ref{fig:sliding-window}) unterteilt. Dieses Verfahren wird in der Arbeit "Enhancement and Cleaning of Handwritten Data by Using Neural Networks" [@cleaning-handwritten-data2005] für die Handschriftbereinigung vorgeschlagen.

![Bereinigung eines verrauschten Bildes, durch pixelweises Bereinigen mit Hilfe der Nachbarpixel und eines kNN \label{fig:sliding-window}](images/prozess.png)

Um ebenfalls die Randpixel bereinigen zu können, wird das Bild vor dem Prozess mit einem schwarzen Rand erweitert. Der Rand besitzt dieselbe Größe wie die Anzahl der berücksichtigten Nachbarn. Am Ende muss das bereinigte Bild aus den einzelnen, bereinigten Pixeln wiederhergestellt werden. Deswegen ist es wichtig, das die Subbilder bei der Bereinigung sortiert vorliegen.

Es werden nur die lokalen Informationen der Nachbarpixel verwendet, um einen bereinigten Pixelwert vorauszusagen. Dies ist sinnvoll, da Informationen außerhalb eines Wortes, geschweige denn am Ende des Bildes, in Bezug auf die Pixel eines Buchstabens, keine Relevanz haben. Das Lernen des gesamten Textes ist auch nicht sinnvoll, da die Texte sich ändern. Es müssen also die einzelnen Buchstaben gelernt werden.

Dies schlägt die Brücke zur Theorie der *convolutional neural networks*, wenn auch diese nicht direkt in der Arbeit Verwendung finden. Dieses Verfahren wird in der Arbeit "Gradient-based learning applied to document recognition" [@Lecun98gradient-basedlearning] im Kapitel "Convolutional neural network for isolated character recognition" von Le Cun beschrieben.

### Alternative

Als Alternative könnten direkt mehrere benachbarte Pixel bereinigt werden. Dabei werden ebenfalls deren Nachbarn als Hilfspixel verwendet. Die Zielpixel wären dann nicht einzelne sondern mehrere Pixel. Dies könnte das Verfahren beschleunigen, da weniger Subbilder generiert und trainiert werden müssten. Die vorliegende Arbeit wird dieses Verfahren mangels Zeit nicht aufgreifen.

## Trainingsprozess

Das Trainieren des *kNN* Teilt die wesentlichen Schritte mit dem in Abbildung \ref{fig:sliding-window} dargestellten Bereinigungsprozess. Die Bilder werden in gleicher Weise vorverarbeitet. Der *Backpropagation-Algorithmus* verlangt jedoch, dass die Trainingsdaten in durchmischter Form trainiert werden. Um dies zu gewährleisten müssen die generierten Subbilder, vor jeder neuen Trainings-Epoche, durchmischt werden.

Auf die Implementation der Trainingsalgorithmen wird im Kapitel \ref{head:network} eingegangen.

### Aufzeichnung des Trainingsprozesses

Während dem Trainieren wird der Trainingsverlauf in einer *MongoDB* aufgezeichnet. Das Trainieren findet auf dem Server *deepgreen02* statt. Durch die Aufzeichnung kann der Trainingsverlauf auf einem Laptop, welcher sich über VPN im HTW-Berlin Netzwerk befindet, visualisiert und analysiert werden.

![Kontextdiagram der Trainingsumgebung \label{fig:training_kontext}](images/training_kontext.png)

\FloatBarrier

## Heuristische Hyperparametersuche

Ist das Netzwerk mit allen benötigten Eigenschaften implementiert, liegt die Schwierigkeit darin eine geeignete Konfiguration zu finden. Konfigurierbar ist die Anatomie des Netzes, sowie auch die Hpyerparameter während dem Training.

Für diese Suche wird häufig der *Grid-Search-Algorithmus* verwendet. Dieser besteht aus einer einfachen Schleife, welche naiv alle Kombinationen von Werten einer Hyperparametertabelle überprüft und die beste Kombination zurückgibt.

In dieser Arbeit wird eine heuristische Art des *Grid-Search-Algorithmus* eingesetzt. Es handelt sich um eine Bayes'sche-Optimierung, welche die bereits geprüften Hyperparameter-Kombinationen für die Wahl der nächsten Hyperparameter miteinbezieht. Dies soll das Finden der besten Kombination insofern beschleunigen, da nicht alle möglichen Kombinationen berücksichtigt werden.

Für die Suche wird die *Python* Bibliothek *Spearmint* verwendet. *Spearmint* wurde zur Arbeit "Practical Bayesian Optimization of Machine Learning Algorithms" [@spearmint] entwickelt und darf ausschließlich für nicht-kommerzielle Zwecke verwendet werden.

### Fine-tuning

Die heuristische Hyperparemetersuche wird als grobe Suche verwendet, um möglichst viele Kombinationen zu testen. Ist diese abgeschlossen, werden die Trainingsläufe analysiert und einzelne Parametereigenschaften zusätzlich analysiert.

So wird zusätzlich analysiert, welche Rolle die Lernrate, die unsichtbare Schicht und auch die Größe des *Mini-Batch* spielen. Um gezielte Trainingsläufe zu starten, wird die heuristische Suche umgangen.

## Konfiguration des kNN

Die Konfiguration des kNN wird unter anderem über die heuristische Hyperparametersuche gesucht. So wird die Neuronenanzahl der unsichtbaren Schichten auch als Hyperparameter übergeben.

Die Anzahl und Art der Schichten wird nicht automatisch gesucht. Es wird immer ein Netzwerk aus verschiedenen Schichten zusammengestellt und darin durch die heuristische Hyperparemetersuche die optimale Konfiguration gesucht.

Arten von kNN welche untersucht werden:

- Einschichtige kNN
- Einschichtiges Autoencoder kNN
- Einschichtiges Denoising Autoencoder kNN
- Mehrschichtiges kNN mit Sigmoid Aktivierungsfunktion
- Mehrschichtiges kNN mit ReLU Aktivierungsfunktion
- Mehrschichtiges Denoising Autoencoer kNN

## Trainings- und Validationsdaten Unterteilung

Die wesentlichen Eigenschaften, welche vom kNN gelernt werden sollen, sind die Schriftbilder der verwendeten Schriftarten und Stile. Die von Kaggle zur Verfügung gestellten Testdaten verwenden exakt dieselben Schriften und Stile wie die Trainingsdaten. Die Testdaten unterscheiden sich durch verschiedene Hintergründe, Sättigung der Schrift, sowie einem anderen Text.

Es muss also darauf geachtet werden, dass die Test- und Validierungsdaten gleiche Schriftbilder enthalten. Das kNN soll neuen Schmutz erkennen und beseitigen, nicht aber neue Schriftbilder ableiten können.

Beim Analysieren der Trainingsdaten ist aufgefallen, dass zwei verschiedene Texte Verwendung finden. Diese Texte existieren beide exakt gleich oft in allen Schriftvariationen. Begünstigt wird dies dadurch dass die Hintergründe ebenfalls leicht abweichen.

Somit werden die Trainingsdaten in Trainings- und Validierungsdaten 50:50, anhand der verschiedenen Texte aufgeteilt. Dies soll eine möglichst reale Präzision beim Validieren ermöglichen.

Um bei Kaggel trotz der Hälfte der Trainingsdaten gut abzuschneiden, wird das Beste eigene Modell erneut mit den selben Hyperparameter, Schichtkombinationen und Epochenanzahl auf allen Trainingsdaten trainiert.

### Kleine Datenbasis für effiziente Hyperparametersuche

Anhand der oben beschriebenen Methodik wird ebenfalls eine noch kleinere Datenbasis zusammengestellt. Darauf können schneller und effizienter verschiedene Kombinationen von Hyperparametern trainiert und validiert werden.

Dies muss nicht zwingend auch zum besten Modell für die gesamten Trainingsmenge führen, da auch die Menge der Trainingsdaten Einfluss auf das *kNN* haben (siehe Regularisation). Aus Zeitgründen, wird diese Methode angewendet.

### Deklaration der Datenbasis

**Mittlere Datenbasis**

Trainingsdaten

>72 Bilder, 24 Bilder (540x258), 48 Bilder (540x420), 8 Hintergründe, 1 Text, alle Schriften und Stile.

Validationsdaten

>72 Bilder; 24 Bilder (540x258); 48 Bilder (540x420); 6 neue Hintergründe; 2 Hintergründe identisch den Trainingsdaten; 1 neuer Text; alle Schriften und Stile gleich.

**Kleine Datenbasis**

Trainingsdaten

>20 Bilder, 12 Bilder (540x258), 8 Bilder (540x420), 4 Hintergründe, 1 Text, 5 Schriften in 5 Stile.

Validationsdaten

>20 Bilder, 12 Bilder (540x258), 8 Bilder (540x420), 3 neue Hintergründe, 1 Hintergrund identisch zu den Trainingsdaten, 1 neuer Text, 5 gleiche Schriften in 5 Stile.
