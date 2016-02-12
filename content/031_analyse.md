# Entwurf

Im Kapitel Entwurf, werden die nötigen Prozesse zur Umsetzung der Anforderungen analysiert und beschrieben. Es bestehen zwei Hauptprozesse, welche sich teilweise überschneiden. Diese wären der Bereinigungsprozess eines Bildes sowie der Trainingsprozess des kNN. Darüber hinaus werden unterstützende Vorgehensweisen zur Hyperparemetersuche und zur Konfiguration der Architektur des kNN beschrieben.

## Bereinigungsprozess

Die Trainings- und Testdaten bestehen aus verschieden großen Bilder. Um ein Bild als Ganzes von einem *kNN* bereinigen zu lassen, müsste die Eingangsschicht aus den Graustrufenwerte der einzelnen Pixel des Bildes bestehen. Dadurch definiert die Bildgröße auch die Eingangsgröße des *kNN*.

Der Aufbau eines *kNN* kann nachträglich nicht verändert werden. Dies bedeutet, dass in diesem Falle für jedes Bild mit einer anderen Bildgröße ein eigenes *kNN* zugeschnitten werden muss.

Um dieses Problem zu lösen, wird einen Vorverarbeitungsschritt eingeführt. Die Bilder werden nicht als Ganzes bereinigt, sondern werden zuerst in gleich große Subbilder anhand der Abbildung \ref{fig:sliding-window} unterteilt. Dieses Verfahren wurde in der Arbeit [@cleaning-handwritten-data2005] für die Handschriftbereinigung vorgestellt.

![Bereinigung eines verrauschten Bildes, durch pixelweises bereinigen mit Hilfe der Nachbarpixel und eines kNN \label{fig:sliding-window}](images/prozess.png)

Um ebenfalls die Randpixel bereinigen zu können, wird das Bild vor dem Prozess mit einem schwarzen Rand erweitert. Der Rand besitzt dieselbe Größe wie die Anzahl der berücksichtigten Nachbarn. Am Ende muss das bereinigte Bild aus den einzelnen, bereinigten Pixeln wiederhergestellt werden. Deswegen ist es wichtig, das die Subbilder bei der Bereinigung sortiert vorliegen.

Es werden nur die lokalen Informationen der Nachbarpixel verwendet, um einen bereinigten Pixelwert vorauszusagen. Dies ist sinnvoll, da Informationen außerhalb eines Wortes, geschweige denn am Ende des Bildes, keine Relevanz haben, in Bezug auf Pixel eines Buchstabens. Das Lernen des gesamten Textes ist auch nicht sinnvoll, da die Texte sich ändern. Es müssen also die einzelnen Buchstaben gelernt werden.

Dies schlägt die Brücke zur Theorie der *convolutional neural networks*, wenn auch diese nicht direkt in der Arbeit Verwendung finden. Dieses Verfahren wird in der Arbeit "Gradient-based learning applied to document recognition" [@Lecun98gradient-basedlearning] im Kapitel "Convolutional neural network for isolated character recognition" von Le Cun beschrieben.

## Trainingsprozess

Das Trainieren des *kNN* Teilt die wesentliche Schritte mit dem in Abbildung \ref{fig:sliding-window} dargestellten Bereinigungsprozess. Die Bilder werden in gleicher Weise vorverarbeitet. Der *Backpropagation-Algorithmus* verlangt jedoch, dass die Trainingsdaten in durchmischter Form trainiert werden. Um dies zu gewährleisten müssen die generierten Subbilder, vor jeder neuen Epoche, durchmischt werden.

Auf die Implementation der Trainingsalgorithmen wird im Kapitel \ref{head:network} Eingegangen.

### Aufzeichnung des Trainingsprozesses

Während dem Trainieren wird der Trainingsverlauf in einer *MongoDB* aufgezeichnet. Das Trainieren findet auf dem Server *deepgreen02* statt. Durch die Aufzeichnung kann der Trainingsverlauf auf einem Laptop, welcher sich über VPN im HTW-Berlin Netzwerk befindet, visualisiert und analysiert werden.

![Kontextdiagram der Trainingsumgebung \label{fig:training_kontext}](images/training_kontext.png)

\FloatBarrier

## Heuristische Hyperparametersuche

Ist das Netzwerk mit allen benötigten Eigenschaften implementiert, liegt die Schwierigkeit darin eine geeignete Konfiguration zu finden. Konfigurierbar ist die Anatomie des Netzes, als auch um die Hpyerparameter während dem Training.

Für diese Suche wird häufig der *Grid-Search-Algorithmus* verwendet. Dieser besteht aus einer einfachen Schleife, welche naiv alle Kombinationen von Werte einer Hyperparametertabelle überprüft und die beste Kombination zurück gibt.

In dieser Arbeit, wird eine heuristische Art des *Grid-Search-Algorithmus* eingesetzt. Es handelt sich um eine Bayes'sche Optimierung, welche die bereits geprüften Hyperparameter Kombinationen für die Wahl der nächsten Hyperparameter miteinbezieht. Dies soll das Finden der besten Kombination in sofern beschleunigen, da nicht alle möglichen Kombinationen berücksichtigt werden.

Für die Suche wird die *Python* Bibliothek *Spearmint* verwendet. *Spearmint* wurde zur Arbeit "Practical Bayesian Optimization of Machine Learning Algorithms" [@spearmint] entwickelt und darf ausschliesslich für nicht-kommerzielle Zwecke verwendet werden.

### Finetuning

Die heuristische Hyperparemetersuche wird als grobe Suche verwendet um möglichst viele Kombinationen zu testen. Ist diese Abgeschlossen, werden die Trainingsläufe analysiert und einzelne Parametereigenschaften zusätzlich untersucht.

So wird zusätzlich untersucht welche Rolle die Lernrate, die unsichtbare Schicht und auch die Größe des *Mini-Batch* spielen. Um gezielte Trainingsläufte zu starten, wird die heuristische Suche umgangen.

## Konfiguration des kNN

Die Konfiguration des kNN wird unter anderem über die heuristische Hyperparametersuche gesucht. So wird die Neuronenanzahl der unsichtbaren Schichten auch als Hyperparameter übergeben.

Die Anzahl und Art der Schichten wird nicht automatisch gesucht. Es wird immer ein Netzwerk aus verschiedenen Schichten zusammengestellt und darin durch die heuristische Hyperparemetersuche die optimale Konfituration gesucht.

Arten von kNN welche untersucht werden:

- Einschichtige kNN
- Einschichtiges Autoencoder kNN
- Einschichtiges Denoising Autoencoder kNN
- Mehrschichtiges kNN mit Sigmoid Aktivierungsfunktion
- Mehrschichtiges kNN mit ReLU Aktivierungsfunktion
- Mehrschichtiges Denoising Autoencoer kNN

## Trainings- und Validationsdaten Unterteilung

Die wesentlichen Eigenschaften, welche vom kNN gelernt werden sollen sind die Schriftbilder der verwendeten Schriftarten und Stiele. Die von Kaggle zur Verfügung gestellten Testdaten verwenden exakt die selben Schriften und Stiele wie die Trainingsdaten. Die Testdaten unterscheiden sich durch verschiedene Hintergründe, Sättigung der Schrift, sowie einem anderen Text.

Es muss also darauf geachtet werden, dass die Test- und Validierungsdaten gleiche Schriftbilder enthalten. Das kNN soll neuen Schmutz erkennen und beseitigen, nicht aber neue Schriftbilder ableiten können.

Beim Analysieren der Trainingsdaten ist aufgefallen, dass zwei verschiedene Texte Verwendung finden. Diese Texte existieren beide exakt gleich oft in allen Schriftvariationen. Begünstigt wird dies dadurch dass die Hintergründe ebenfalls leicht abweichen.

Somit werden die Trainingsdaten in Trainings- und Validierungsdaten 50:50, anhand der verschiedenen Texte aufgeteilt. Dies soll eine möglichst reale Präzision beim Validieren ermöglichen.

Um bei Kaggel trotz der Hälfte der Trainingsdaten gut abzuschneiden, wird das Beste eigene Modell erneut mit den selben Hyperparameter, Schichtkombinationen und Epochenanzahl auf allen Trainingsdaten trainiert. ^[Alternativ könnte ein neues Modell gleicher Konfiguration auf den Validierungsdaten trainiert werden und mit dem Bestehenden zu einem Ensemble kombiniert werden.]

### Kleine Datenbasis für effiziente Hyperparametersuche

Anhand der oben beschriebenen Methodik wird ebenfalls eine noch kleinere Datenbasis zusammengestellt. Darauf können schneller und effizienter verschiedene Kombinationen von Hyperparameter trainiert und validiert werden. Dahinter verbirgt sich die von Aristoteles eingeführte Induktionslogik, mit welche vom Speziellen (kleinen) auf das Allgemeine (Große) geschlossen wird.

### Deklaration der Datenbasis

**Mittlere Datenbasis**

Trainingsdaten

>72 Bilder, 24 Bilder (540x258), 48 Bilder (540x420), 8 Hintergründe, 1 Text, Alle Schriften und Stiele

Validationsdaten

>72 Bilder; 24 Bilder (540x258); 48 Bilder (540x420); 6 neue Hintergründe; 2 Hintergründe gleich zu den Trainingsdaten; 1 neuer Text; Alle Schriften und Stiele gleich.

**Kleine Datenbasis**

Trainingsdaten

>20 Bilder, 12 Bilder (540x258), 8 Bilder (540x420), 4 Hintergründe, 1 Text, 5 Schriften in 5 Stiele

Validationsdaten

>20 Bilder, 12 Bilder (540x258), 8 Bilder (540x420), 3 neue Hintergründe, 1 gleicher Hintergrund wie in den Trainingsdaten, 1 neuer Text, 5 gleiche Schriften in 5 Stiele
