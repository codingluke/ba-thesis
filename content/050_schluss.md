# Fazit

## Rückblick

### Erreichte Resultate

In der Arbeit wurde erfolgreich ein modularer Ansatz zur Konfiguration von mehrschichtigen *kNN* und dessen Training von Grund auf in der Programmiersprache *Python* implementiert. Die Implementation wurde sogleich erfolgreich für das bereinigen von verrauschter, eingescannter Bilder eingesetzt. Für den speziellen Fall des bereinigen verrauschter Schriftbilder mit variablen Größe, wurde ein vorausgehenden Verarbeitungsschritt eingeführt.

Die Resultate haben ergeben, dass eine zweite unsichtbare Schicht einen signifikante Verbesserung nach sich zieht. Weitere unsichtbare Schichten, verbessern das Resultat nur geringfügig. In der vierten Schicht nimmt das Resultat sogar wieder ab. Als weiteres konnte aufgezeigt werden, dass im voraus trainierte *Stacked-Denoising-Autoencoder* in den lokalen Trainings- und Validierungsdaten, im Gegensatz zu normalen mehrschichtigen *kNN*, zu besseren Resultate führen und auch schneller Konvergieren. Normale mehrschichtige *kNN* mit der Aktivierungsfunkion *ReLU* relativieren die Nützlichkeit der *Stacked-Denoising-Autoencoder*, da sie nicht am Problem des *Gradientenschwunds* leiden. Dies wurde bereits in der Arbeit "On rectified linear units for speech processing" [@Zeiler_onrectified] beschrieben und konnte rekonstruiert werden.

Umso überraschender war das sehr gute Abschneiden des normalen dreischichtigen *Sigmoid-kNN* im *Kaggle-Wettbewerb*. Dies hat sehr eindrücklich aufgezeigt, dass die Trainings- und Valdidierungsdaten das Fundament beim Trainieren von *kNN* bildet. Das gleichzeitig schlechte Abschneiden der *ReLU*-Neuronen, lässt die Vermutung offen, dass diese auf die Trainingsdaten zu gut passen *Overfitting*.

Regularisierungs-Methoden wie *Dropout* und *L2-Regularisation* haben mit den verfügbaren Validierungsdaten nicht zu einem besseren Resultat geführt. Als möglicher Erklärung, wird angenommen, dass die Validierungsdaten sowie die *Kaggle*-Testdaten den Trainingsdaten zu ähnlich sind und somit *Overfitting* zu besseren Resultaten führt.

Mit der direkten, automatischen Aufzeichnung der Trainingsläufe, wurde ein Prozess erarbeitet und umgesetzt, welcher das spätere Analysieren, Vergleichen und Nachvollziehen einzelner Trainingsläufe enorm vereinfacht und dadurch zusätzlich Übersicht schafft.

Das Grundlagenwissen über künstliche neuronale Netze, sowie deren möglichen Implementationsweisen wurde autodidaktisch im Selbststudium erarbeitet und waren nicht Gegenstand belegter Module. Mit der Implementation eines einfachen *Frameworks* wurde das angeeignete Wissen erfolgreich angewendet.

### Schwierigkeiten

Die Hauptschwierigkeit lag an der permanenten Unsicherheit, ob das Netzwerk und deren Schichten korrekt implementiert sind. So sind Anfangs viele irritierende Ergebnisse entstanden. In der Mitte der Arbeit wurde beispielsweise erst klar, dass die bisher verwendete Validierungskostenfunktion nicht korrekt implementiert ist. Dies führte dazu, dass viele Trainingsvorgänge nochmal wiederholt werden mussten. Das Trainieren nimmt sehr viel Zeit in Anspruch. Wenn nun ein grundlegender Fehler gefunden wird, sind alte Tests nicht mehr repräsentativ. Dies wird jedoch häufig erst während dem Trainieren sichtbar. Somit ist ständig die Angst vorhanden am Ende in Zeitnot zu gelangen. Das inkonsistenten Resultat auf *Kaggle* hat diese Unsicherheit zusätzlich verstärkt.

Es kann gut möglich sein, dass bei der Implementation der Regularisierungs-Methoden Fehler unterlaufen sind, und diese deshalb das Resultat nicht positiv beeinflussen. Durch das allgemein gute Abschneiden auf *Kaggle* ist jedoch ziemlich sicher, dass die Trainingsalgorithmen korrekt implementiert sind.

Hier wird auch direkt auf die nächste Schwierigkeit verwiesen. Grundsätzlich können ständig neue Tests gemacht werden. Um jedoch ebenfalls an der Arbeit zu schreiben, muss an einem Zeitpunkt einen Schlussstrich gezogen werden. Während dem Training hat es sich als Schwierig herausgestellt mit klarem Kopf zu schreiben. Ständig war die Versuchung im Raum, das Training zu verfolgen und neue Hypothesen aufzustellen.

Des Weiteren hat sich das *Debugen* von komplexen *Theano*-Funktionen, als schwierig herausgestellt. Solche Funktionen können nicht wie gewohnt durch *Beakponts* lokal analysiert werden. Fehler in *Theano*-Funktionen waren ausschließlich auf fehlerhafte Datentypen zurückzuführen. Im allgemeinen war das Anwenden von symbolischen Mathematik eine Neuigkeit, welche viel Zeit zum Verstehen in Anspruch nahm.

## Ausblick

Um das *kNN* generalisierter trainieren zu können, werden vor allem bessere Trainingsdaten vorausgesetzt. Hier könnte versucht werden automatisiert Daten zu generieren. Es können Algorithmen zum Erstellen verrauschter Hintergründe sowie verzerren, verschmieren und weiteren Modifizieren der Schrift vorgenommen werden.

Im professionellen Einsatz sollte allenfalls die Verwendung bestehender *Frameworks* wie *Keras* in Betracht gezogen werden. Ebenfalls gibt es mit den aufeinander aufbauenden *Frameworks* *Blocks* und *Fuel* neue Ansätze, *Theano* elementar zu verwenden, ohne die vielen Standardprozesse selbst implementieren zu müssen. Interessanterweise verwendet *Blocks* und *Fuel* einen ähnlichen Ansatz wie in dieser Arbeit implementierte Klassen *BatchProcessor* und *Network* mit den Modularen Schichtklassen. Nur werden noch viel mehr Kombinationen unterstützt. Zusätzlich bewahrt dies vor der Ungewissheit, ob die fundamentalen Algorithmen auch korrekt implementiert sind.

Zusätzlich kann überlegt werden, wie die Vorverarbeitung weiter angepasst werden kann. Als Beispiel wurde bereits in Kapitel \ref{head:bereinigungsprozess} darauf eingegangen, dass direkt mehrere Pixel auf einmal bereinigt werden könnten und nicht, wie in dieser Arbeit, jedes Einzelne. Dies könnte durch ein Erweitern der Klasse *preprocessor.Processor* erreicht werden. Es wird vor allem angenommen, dass die in Abbildung \ref{fig:schicht-vergleich-22} anzutreffende Lücken in Buchstaben so möglicherweise geschlossen werden können.
