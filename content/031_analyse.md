# Analyse und Systementwurf \label{head:analyse}

In diesem Kapitel werden die nötigen Prozesse zur Umsetzung der Anforderungen analysiert und beschrieben. Es bestehen zwei Hauptprozesse, welche sich teilweise überschneiden. Diese wären der Bereinigungsprozess eines Bildes sowie der Trainingsprozess des *kNN*. Darüber hinaus werden unterstützende Vorgehensweisen zur Hyperparemetersuche und zur Konfiguration der Architektur des *kNN* beschrieben.

## Bereinigungsprozess \label{head:bereinigungsprozess}

Die Trainings- und Testdaten bestehen aus verschieden großen Bildern. Um ein Bild als Ganzes von einem *kNN* bereinigen zu lassen, müsste die Eingangsschicht aus den Graustrufenwerten der einzelnen Pixel des Bildes bestehen. Dadurch definiert die Bildgröße sogleich die Eingangsgröße des *kNN*. Der Aufbau eines *kNN* kann nachträglich nicht verändert werden. Dies bedeutet, dass in diesem Falle für jedes Bild mit einer anderen Bildgröße ein eigenes *kNN* zugeschnitten werden muss.

Um dieses Problem zu lösen, wird ein Vorverarbeitungsschritt eingeführt. Die Bilder werden nicht als Ganzes bereinigt, sondern werden zuerst in gleich große Subbilder unterteilt (siehe Abbildung \ref{fig:sliding-window}). Dieses Verfahren wird in der Arbeit "Enhancement and Cleaning of Handwritten Data by Using Neural Networks" [@cleaning-handwritten-data2005] für die Handschriftbereinigung vorgeschlagen.

![Bereinigung eines verrauschten Bildes, durch pixelweises Bereinigen mit Hilfe der Nachbarpixel und eines kNN [@hodel] \label{fig:sliding-window}](images/Bereinigungsprozess.pdf)

Um ebenfalls die Randpixel bereinigen zu können, wird das Bild vor dem Prozess mit einem schwarzen Rand erweitert. Der Rand besitzt dabei dieselbe Größe wie die Anzahl der berücksichtigten Nachbarn. Am Ende muss das bereinigte Bild aus den einzelnen, bereinigten Pixeln wiederhergestellt werden. Deswegen ist es wichtig, das die Subbilder bei der Bereinigung sortiert vorliegen.

Es werden nur die lokalen Informationen der Nachbarpixel verwendet, um einen bereinigten Pixelwert vorauszusagen. Dies ist sinnvoll, da Informationen außerhalb eines Wortes, geschweige denn am Ende des Bildes, in Bezug auf die Pixel eines Buchstabens, nicht Relevant sind. Das Lernen des gesamten Textes ist ebenfalls nicht sinnvoll, da die Texte sich ändern. Es müssen also die einzelnen Buchstaben gelernt werden.

Dies schlägt die Brücke zur Theorie der *convolutional neural networks*, wenn auch diese nicht direkt in der Arbeit Verwendung finden. Dieses Verfahren wird in der Arbeit "Gradient-based learning applied to document recognition" [@Lecun98gradient-basedlearning] im Kapitel "Convolutional neural network for isolated character recognition" von Le Cun beschrieben.

### Alternative

Als Alternative könnten direkt mehrere benachbarte Pixel bereinigt werden. Dabei werden ebenfalls deren Nachbarn als Hilfspixel verwendet. Die Zielpixel wären dann nicht einzelne sondern mehrere Pixel. Dies könnte das Verfahren beschleunigen, da weniger Subbilder generiert und trainiert werden müssten. Die vorliegende Arbeit wird dieses Verfahren mangels Zeit nicht aufgreifen.

## Trainingsprozess

Das Trainieren des *kNN* Teilt die wesentlichen Schritte mit dem in Abbildung \ref{fig:sliding-window} dargestellten Bereinigungsprozess. Die Bilder werden in gleicher Weise vorverarbeitet. Der *Backpropagation-Algorithmus* verlangt hingegen, dass die Trainingsdaten in durchmischter Form trainiert werden. Um dies zu gewährleisten müssen die generierten Subbilder, vor jeder neuen Trainings-Epoche, durchmischt werden.

Auf die Implementation der Trainingsalgorithmen wird im Kapitel \ref{head:network} eingegangen.

### Aufzeichnung des Trainingsprozesses \label{head:aufzeichung-training-prozess}

Während dem Trainieren wird der Trainingsverlauf in einer *MongoDB* aufgezeichnet. Das Trainieren findet auf dem Server *deepgreen02* statt. Durch die Aufzeichnung kann der Trainingsverlauf auf einem Laptop, welcher sich über *VPN* im HTW-Berlin Netzwerk befindet, visualisiert und analysiert werden.

![Kontextdiagramm der Trainingsumgebung [@hodel] \label{fig:training-kontext}](images/VPN-Umgebung.pdf)

\FloatBarrier

## Heuristische Hyperparametersuche

Ist das Netzwerk mit allen benötigten Eigenschaften implementiert, liegt die Schwierigkeit darin eine geeignete Konfiguration zu finden. Konfigurierbar ist die Anatomie des Netzes, sowie auch die Hpyerparameter während dem Training.

Für diese Suche wird häufig der *Grid-Search-Algorithmus* verwendet. Dieser besteht aus einer einfachen Schleife, welche naiv alle Kombinationen von Werten einer Hyperparametertabelle überprüft und die beste Kombination zurückgibt.

In dieser Arbeit wird eine heuristische Art des *Grid-Search-Algorithmus* eingesetzt. Es handelt sich um eine Bayes'sche-Optimierung, welche die bereits geprüften Hyperparameter-Kombinationen für die Wahl der nächsten Hyperparameter miteinbezieht. Dies soll das Finden der besten Kombination insofern beschleunigen, da nicht alle möglichen Kombinationen berücksichtigt werden.

Für diese Suche wird die *Python*-Bibliothek *Spearmint* verwendet. *Spearmint* wurde zur Arbeit "Practical Bayesian Optimization of Machine Learning Algorithms" [@spearmint] entwickelt und darf ausschließlich für nicht-kommerzielle Zwecke verwendet werden.

### Fine-tuning

Die heuristische Hyperparemetersuche wird als grobe Suche verwendet, um möglichst viele Kombinationen zu testen. Ist diese abgeschlossen, werden die Trainingsläufe und zusätzlich einzelne Parametereigenschaften analysiert.

So wird ebenfalls erforscht, welche Rolle die Lernrate, die unsichtbare Schicht und auch die Größe des *Minibatch* spielen. Um gezielte Trainingsläufe zu starten, wird die heuristische Suche umgangen.

## Konfiguration des kNN

Die Konfiguration des *kNN* wird unter anderem über die heuristische Hyperparametersuche gesucht. Dazu wird die Neuronenanzahl der unsichtbaren Schichten ebenfalls als Hyperparameter übergeben.

Die Anzahl und Art der Schichten wird nicht automatisch gesucht. Es wird immer ein Netzwerk aus verschiedenen Schichten zusammengestellt und darin durch die heuristische Hyperparemetersuche die optimale Konfiguration gesucht.

Arten von *kNN* welche untersucht werden:

- Einschichtige *kNN*
- Einschichtiges Autoencoder *kNN*
- Einschichtiges Denoising-Autoencoder *kNN*, *dA*
- Mehrschichtiges *kNN*, *MLP*, mit *Sigmoid*-Aktivierungsfunktion
- Mehrschichtiges *kNN*, *MLP*, mit *ReLU*-Aktivierungsfunktion
- *Stacked-denoising-Autoencoer*, *SdA*

## Datenunterteilung

Da für die *Testdaten* keine bereinigten Zieldaten (y) existieren, das Prüfen erfolgt durch ein Formular auf der Wettbewerb-Webseite, müssen die *Wettbewerb-Trainingsdaten* nochmals in eigene Trainings- und Validierungsdaten unterteilt werden (siehe Abbildung \ref{fig:train-test-split}). Dabei muss darauf geachtet werden, dass die abgeleiteten Trainings- und Validierungsdaten gleiche Schriftbilder enthalten. Das *kNN* soll neuen Schmutz erkennen und beseitigen, nicht aber neue Schriftbilder ableiten können.

![Aufteilung der Daten in Test-, Trainings- und Validierungsdaten mehrerer Größen. Dabei besitzen die Wettbewerb-Trainingsdaten verunreinigte (X) und bereinigte (y) Daten. Die Testdaten jedoch nur verunreinigte. [@hodel] \label{fig:train-test-split}](images/Datenunterteilung.pdf)

Beim Analysieren der *Wettbewerb-Trainingsdaten* ist aufgefallen, dass zwei verschiedene Texte Verwendung finden. Diese Texte existieren beide exakt gleich oft in allen Schriftvariationen. Begünstigt wird dies dadurch, dass die Hintergründe ebenfalls leicht abweichen. Deswegen werden die *Wettbewerb-Trainingsdaten*, anhand der verschiedenen Texte in Trainings- und Validierungsdaten aufgeteilt. Dies soll eine möglichst reale Präzision beim Validieren ermöglichen. Mit Hilfe dieser abgeleiteten Trainings- und Validationsdaten wird nach dem besten Modellen gesucht.

Um beim Wettbewerb so gut wie möglich abzuschneiden, werden die gefundenen Modelle erneut mit denselben Hyperparameter und Schichtkombinationen, jedoch diesmal mit Hilfe aller *Wettbewerb-Trainingsdaten*, trainiert. Da in diesem Fall für die Validationsdaten, die *Testdaten*, keine bereinigten Zielbilder existieren, ist hierfür das Aufzeichnen des Validierungsverlaufs nicht möglich. Das Netz wird also "blind" trainiert und anschließend durch den Wettbewerb validiert.

### Kleine Datenbasis für effiziente Hyperparametersuche

Anhand der oben beschriebenen Methodik wird ebenfalls eine noch kleinere, die *Kleine-Datenbasis*, zusammengestellt (siehe Abbildung \ref{fig:train-test-split}). Darauf können schneller und effizienter verschiedene Kombinationen von Hyperparametern trainiert und validiert werden.

Dies muss nicht zwingend auch zum besten Modell für die *Große-Datenbasis* führen, da auch die Menge der Trainingsdaten Einfluss auf das *kNN* haben (siehe Regularisation in Kapitel \ref{head:kNN}). Aus Zeitgründen, wird diese Methode angewendet, da angenommen wird, dass Tendenzen auch auf der *Kleinen-Datenbasis* sichtbar werden.

### Eigener Datensatz für das Trainieren der Denoising-Autoencoder

Die *Pretraindaten* für das vorausgehende, schichtweise Training der *Denoising-Autoencoder*, beinhalten ausschließlich bereinigte Zielbilder (y) der Trainingsdaten. Dabei handelt es sich genauer, um die Teilmenge welche nur heterogene Bilder enthaltet (siehe Abbildung \ref{fig:pretrain}).

![Die Pretraindaten sind eine Teilmenge der Zieldaten (y) \label{fig:pretrain} [@hodel]](images/Trainingsdaten.pdf)

Unter den Zieldaten existieren tatsächlich diverse Bilder, unter verschiedenen Namen, mehrfach. Diese Redundanz kommt zu Stande, da verrauschte Bilder mit demselben Schriftbild, jedoch anderem Hintergrund, identische Zielbilder besitzen und deren Zuweisung durch eine Namenskonvention und nicht durch eine Tabelle besteht.

Der Grund, wieso die bereinigte Zielbilder verwendet werden, liegt darin, dass der *Denoising-Autoencoder* die eingehenden Subbilder selbst Verunreinigt. Durch diesen automatisch hinzugefügten *Schmutz*, wird erhofft, dass die dadurch entstehenden Hintergrundbilder das *Unsupervised-Feature-Learning* darin unterstützen besser zu Generalisieren.

#### Alternative

Anstatt automatisch die Eingabedaten zu mutieren, könnte dem *Denoising-Autoencoder* die bereits verunreinigten, sowie dessen bereinigten Zielbilder mitgegeben werden. Diese Variante wurde aus Zeitgründen nicht weiter verfolgt (sollte allerdings nach den Resultaten im Kapitel \ref{head:evaluierung} in Betracht gezogen werden).

### Deklaration der Datenbasen

#### Große-Datenbasis

Trainingsdaten

  ~ 72 verunreinigte Bilder (X), davon 24 Bilder (540x258) und 48 Bilder (540x420). Diese beinhalten 8 Hintergründe, 1 Text, alle in den *Wettbewerb-Daten* vorhandenen Schriftarten und Stile.
  ~ 72 bereinigte Zielbilder (y).

Validationsdaten

  ~ 72 verunreinigte Bilder (X), davon 24 Bilder (540x258) und 48 Bilder (540x420). Diese beinhalten 6 neue Hintergründe, 2 Hintergründe identisch den Trainingsdaten, 1 neuer Text, alle Schriftarten und Stile sind identisch zu den Trainingsdaten
  ~ 72 bereinigte Zielbilder (y).

Pretraindaten

  ~ Alle heterogenen Zielbilder (y) der Trainingsdaten.

#### Kleine-Datenbasis

Trainingsdaten

  ~ 20 verunreinigte Bilder, davon 12 Bilder (540x258) und 8 Bilder (540x420). Diese beinhalten 4 Hintergründe, 1 Text, 5 Schriften in 5 Stile.
  ~ 20 bereinigte Zielbilder (y).

Validationsdaten

  ~ 20 verunreinigte Bilder, davon 12 Bilder (540x258) und 8 Bilder (540x420). Diese beinhalten 3 neue Hintergründe, 1 Hintergrund identisch zu den Trainingsdaten, 1 neuer Text, 5 Schriftarten und 5 Stile identisch zu den Trainingsdaten.
  ~ 20 bereinigte Zielbilder (y).

Pretraindaten

  ~ Alle heterogenen Zielbilder (y) der Trainingsdaten.

#### Wettbewerb-Datenbasis

Trainingsdaten

  ~ Alle vom Wettbewerb zur Verfügung gestellten Trainingsdaten.

Testdaten

  ~ Alle vom Wettbewerb zur Verfügung gestellten Testdaten.
  ~ Keine Zieldaten (y) vorhanden.

Pretraindaten

  ~ Alle heterogenen Zielbilder (y) der Trainingsdaten.
