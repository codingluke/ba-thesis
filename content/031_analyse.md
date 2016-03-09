# Analyse und Systementwurf \label{head:analyse}

In diesem Kapitel werden die nötigen Prozesse zur Umsetzung der Anforderungen analysiert und beschrieben. Es bestehen zwei Hauptprozesse, welche sich teilweise überschneiden. Diese wären der Bereinigungsprozess eines Bildes sowie der Trainingsprozess des *kNN*. Darüber hinaus werden unterstützende Vorgehensweisen zur Hyperparemetersuche und zur Konfiguration der Architektur des *kNN* beschrieben.

## Bereinigungsprozess \label{head:bereinigungsprozess}

Die Trainings- und Testdaten bestehen aus verschieden großen Bildern. Um ein Bild als Ganzes von einem *kNN* bereinigen zu lassen, müsste die Eingangsschicht aus den Graustrufenwerten der einzelnen Pixel des Bildes bestehen. Dadurch definiert die Bildgröße sogleich die Eingangsgröße des *kNN*. Der Aufbau eines *kNN* kann nachträglich nicht verändert werden. Dies bedeutet, dass in diesem Falle für jedes Bild mit einer anderen Bildgröße ein eigenes *kNN* zugeschnitten werden muss.

Um dieses Problem zu lösen, wird ein Vorverarbeitungsschritt eingeführt. Die Bilder werden nicht als Ganzes bereinigt, sondern werden zuerst in gleich große Subbilder unterteilt (siehe Abbildung \ref{fig:sliding-window}). Dieses Verfahren wird in der Arbeit "Enhancement and Cleaning of Handwritten Data by Using Neural Networks" [@cleaning-handwritten-data2005] für die Handschriftbereinigung vorgeschlagen.

![Bereinigung eines verrauschten Bildes, durch pixelweises Bereinigen mit Hilfe der Nachbarpixel und eines kNN [Hodel] \label{fig:sliding-window}](images/Bereinigungsprozess.pdf)

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

![Kontextdiagramm der Trainingsumgebung [Hodel] \label{fig:training-kontext}](images/VPN-Umgebung.pdf)

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
- *Stacked-Denoising-Autoencoer*, *SdA*

## Trainings- und Validationsdaten Unterteilung

Die wesentlichen Eigenschaften, welche vom *kNN* gelernt werden sollen, sind die Schriftbilder der verwendeten Schriftarten und Stile. Die von *Kaggle* zur Verfügung gestellten Testdaten verwenden die exakt selben Schriften und Stile wie die Trainingsdaten. Die Testdaten unterscheiden sich durch verschiedene Hintergründe, Sättigung der Schrift, sowie einem anderen Text.

Es muss also darauf geachtet werden, dass die daraus Abgeleiteten Test- und Validierungsdaten ebenfalls gleiche Schriftbilder enthalten. Das *kNN* soll neuen Schmutz erkennen und beseitigen, nicht aber neue Schriftbilder ableiten können.

Beim Analysieren der Trainingsdaten ist aufgefallen, dass zwei verschiedene Texte Verwendung finden. Diese Texte existieren beide exakt gleich oft in allen Schriftvariationen. Begünstigt wird dies dadurch dass die Hintergründe ebenfalls leicht abweichen.

Deswegen werden die Trainingsdaten in Trainings- und Validierungsdaten, 50:50, anhand der verschiedenen Texte aufgeteilt. Dies soll eine möglichst reale Präzision beim Validieren ermöglichen.

Um beim Wettbewerb [@kaggleDDD] so gut wie möglich abzuschneiden, werden die Modell erneut mit denselben Hyperparameter, Schichtkombinationen und Epochenanzahl, jedoch diesmal mit Hilfe aller verfügbaren Trainingsdaten, trainiert. Da in diesem Fall keine Validationsdaten vorhanden sind, sondern nur die Testdaten von Kaggle, für welche keine bereinigten Zielbilder existieren, ist dafür das Aufzeichnen des Validierungsverlauf nicht möglich. Das Netz wird somit "blind" trainiert und anschließend durch das Hochladen direkt auf *Kaggle* validiert.

### Kleine Datenbasis für effiziente Hyperparametersuche

Anhand der oben beschriebenen Methodik wird ebenfalls eine noch kleinere Datenbasis zusammengestellt. Darauf können schneller und effizienter verschiedene Kombinationen von Hyperparametern trainiert und validiert werden.

Dies muss nicht zwingend auch zum besten Modell für die große Trainingsmenge führen, da auch die Menge der Trainingsdaten Einfluss auf das *kNN* haben (siehe Regularisation in Kapitel \ref{head:kNN}). Aus Zeitgründen, wird diese Methode angewendet, da angenommen wird, dass Tendenzen auch auf der kleinen Datenbasis sichtbar werden.

### Eigener Datensatz für das Trainieren der Denoising-Autoencoder

Der Datensatz für das schichtweise Training der *Denoising-Autoencoder* beinhaltet ausschließlich bereits bereinigte Schriftbilder. Dabei werden pro Trainingsdatensatz nur Bilder mit Schriften gewählt, welche auch in den Trainings- und Validierungsdaten vorkommen.

Der Grund, wieso bereinigte Bilder verwendet werden liegt darin, da der *Denoising-Autoencoder* die eingehenden Subbilder selbst Verunreinigt. Durch diesen automatisch generierten *Schmutz* wird erhofft, dass zusätzlich zu den bestehenden Hintergrundbilder neue generiert werden und somit das *Unsupervised-Feature-Learning* in sofern unterstützen, dass es besser generalisiert.

#### Alternative

Anstatt automatisch die Eingabedaten zu mutieren, könnte dem *Denoising-Autoencoder* die bereits verunreinigten sowie dessen bereinigten Zielbilder mitgegeben werden. Diese Variante wurde aus Zeitgründen nicht weiter verfolgt, sollte allerdings nach den Resultaten im Kapitel \ref{head:evaluierung} in Betracht gezogen werden.

### Deklaration der Datenbasen

**Große Datenbasis**

Trainingsdaten

>72 Bilder, 24 Bilder (540x258), 48 Bilder (540x420), 8 Hintergründe, 1 Text, alle Schriften und Stile.

Validationsdaten

>72 Bilder; 24 Bilder (540x258); 48 Bilder (540x420); 6 neue Hintergründe; 2 Hintergründe identisch den Trainingsdaten; 1 neuer Text; alle Schriften und Stile gleich.

Autoencoder-Trainingsdaten

>Alle bereinigten Trainingsbilder, wobei doppelte ausgeschlossen wurden (Bilder mit verschiedenen Hintergründe, jedoch dem selben Text besitzen gleiche Zielbilder)

**Kleine Datenbasis**

Trainingsdaten

>20 Bilder, 12 Bilder (540x258), 8 Bilder (540x420), 4 Hintergründe, 1 Text, 5 Schriften in 5 Stile.

Validationsdaten

>20 Bilder, 12 Bilder (540x258), 8 Bilder (540x420), 3 neue Hintergründe, 1 Hintergrund identisch zu den Trainingsdaten, 1 neuer Text, 5 gleiche Schriften in 5 Stile.

Autoencoder-Trainingsdaten

>Alle bereinigten Trainingsbilder, wobei doppelte ausgeschlossen wurden (Bilder mit verschiedenen Hintergründe, jedoch dem selben Text besitzen gleiche Zielbilder)

**Kaggle Datenbasis**

Trainingsdaten

> Alle von Kaggle zur Verfügung gestellten Trainingsdaten.

Validierungsdaten

> Alle von Kaggle zur Verfügung gestellten Testdaten. Da für diese Testdaten keine Zielbilder vorhanden sind, wird die Validierung auf der Kaggle-Webseite vorgenommen.

Autoencoder-Trainingsdaten

> Alle bereinigten Trainingsbilder.
