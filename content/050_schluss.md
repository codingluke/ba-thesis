# Fazit \label{head:fazit}

## Rückblick

### Erreichte Resultate

In der Arbeit wurde erfolgreich ein modularer Ansatz zur Konfiguration von mehrschichtigen *kNN* und dessen Training von Grund auf in der Programmiersprache *Python* implementiert. Die Implementation wurde sogleich erfolgreich für das Bereinigen verrauschter, eingescannter Bilder eingesetzt. Für den speziellen Fall des Bereinigens verrauschter Schriftbilder mit variablen Größen wurde ein vorausgehender Verarbeitungsschritt eingeführt.

Die Resultate mit den eigens abgeleiteten Datensätze haben ergeben, dass eine zweite unsichtbare Schicht eine signifikante Verbesserung nach sich zieht. Eine dritte unsichtbare Schicht verbessert das Resultat nur noch geringfügig und nimmt mit vier Schichten sogar wieder ab. Des Weiteren konnte aufgezeigt werden, dass im Voraus trainierte *Stacked-denoising-Autoencoder*, im Gegensatz zu normalen *MLP*, zu besseren Resultaten führen und auch schneller konvergieren. Ebenfalls konnte das Resultat der Arbeit "On rectified linear units for speech processing" [@Zeiler_onrectified] rekonstruiert werden, wonach *ReLU*-Neuronen der *Sigmoid*-Neuronen beim *Deep-Learning* im Vorteil sind.

Umso überraschender waren die Resultate im offiziellen Wettbewerb. Hier überragte das dreischichtige normale *MLP* mit *Sigmoid*-Neuronen und holte den 11. Platz. Dies hat äußerst eindrücklich aufgezeigt, dass die Trainings- und Validierungsdaten zusammen das Fundament beim Trainieren im maschinellen Lernen bilden. Das gleichzeitig schlechte Abschneiden der *ReLU*-Neuronen, lässt die Vermutung offen, dass diese auf die Trainingsdaten zu gut passen, also *Overfitten*. Die *SdA* schneiden bei zwei und vier Schichten besser ab als das normale *MLP*. Ausgerechnet beim siegenden dreischichtigen *MLP* ist der *SdA* dem *MLP*, im Gegensatz zu den vorhergehenden Tests, leicht unterlegen.

Regularisierungs-Methoden wie *Dropout* und *L2-Regularisation* haben mit den verfügbaren Validierungsdaten nicht zu einem besseren Resultat geführt. Als mögliche Erklärung wird angenommen, dass die Testdaten den Trainingsdaten zu ähnlich sind und somit *Overfitting* zu besseren Resultaten führt.

Mit der direkten, automatischen Aufzeichnung der Trainingsläufe, wurde ein Prozess erarbeitet und umgesetzt, welcher das spätere Analysieren, Vergleichen und Nachvollziehen einzelner Trainingsläufe enorm vereinfacht und dadurch zusätzlich Übersicht schafft.

Das Grundlagenwissen über künstliche neuronale Netze, sowie deren mögliche Implementationsweisen, wurde im Selbststudium erarbeitet und war nicht Gegenstand belegter Module. Mit der Implementation eines einfachen *Frameworks* konnte das angeeignete Wissen erfolgreich angewendet werden.

### Schwierigkeiten

Die Hauptschwierigkeit lag an der permanenten Unsicherheit, ob das Netzwerk und deren Schichten korrekt implementiert sind. So sind anfangs viele irritierende Ergebnisse entstanden. In der Mitte der Arbeit wurde beispielsweise erst klar, dass die bisher verwendete Validierungskostenfunktion nicht korrekt implementiert ist. Dies führte dazu, dass viele Trainingsvorgänge nochmal wiederholt werden mussten. Das Trainieren selbst stellt die zeitintensivste Komponente dar. Wenn während dessen ein grundlegender Fehler gefunden wird, sind alte Tests nicht mehr repräsentativ. Dies wird allerdings häufig erst während dem Trainieren sichtbar. Somit ist ständig das Risiko vorhanden am Ende in Zeitnot zu geraten. Das inkonsistente Resultat im Wettbewerb hat diese Unsicherheit zusätzlich bestärkt.

Es kann gut möglich sein, dass bei der Implementation der Regularisierungs-Methoden Fehler unterlaufen sind und diese deshalb das Resultat nicht positiv beeinflussen. Durch das allgemein gute Abschneiden im Wettbewerb kann jedoch davon ausgegangen werden, dass die Trainingsalgorithmen korrekt implementiert sind. Auch sollte das Trainieren des dreischichtigen *SdA* wiederholt werden um sicher zu sein, ob hier nicht ein Fehler unterlaufen ist.

Dadurch wird direkt auf die nächste Schwierigkeit verwiesen. Grundsätzlich können ständig neue Tests gemacht werden. Um jedoch mit der eigentlicher Niederschrift der Arbeit beginnen zu können, musste zu einem gewissen Zeitpunkt mit den Testgängen abgeschlossen werden. Während dem Training bestand also die größte Schwierigkeit darin, sich vollständig auf das Schreiben zu konzentrieren und nicht der "Versuchung" nachzugeben, das Training zu verfolgen und neue Hypothesen aufzustellen.

Des Weiteren hat sich das *Debugen* von komplexen *Theano*-Funktionen als schwierig herausgestellt. Solche Funktionen können nicht wie gewohnt durch *Beakponts* lokal analysiert werden. Fehler in *Theano*-Funktionen waren ausschließlich auf fehlerhafte Datentypen zurückzuführen. Im Allgemeinen war das Anwenden von symbolischer Mathematik ein Novum, welches viel Zeit zum Verstehen in Anspruch nahm.

## Ausblick

Um das *kNN* generalisierter trainieren zu können, werden vor allem diversere Trainingsdaten vorausgesetzt. Hier könnte versucht werden, automatisiert Daten zu generieren. Es könnten Algorithmen zum Erstellen verrauschter Hintergründe sowie Verzerrungen, Verschmierungen und weiteres Modifizieren der Schrift vorgenommen werden.

Im professionellen Einsatz sollte allenfalls die Verwendung bestehender *Frameworks*, wie *Keras*, in Betracht gezogen werden. Ebenfalls gibt es mit den aufeinander aufbauenden *Frameworks* *Blocks* und *Fuel* neue Ansätze, *Theano* elementar zu verwenden, ohne die vielen Standardprozesse selbst implementieren zu müssen. Interessanterweise verwendet *Blocks* und *Fuel* einen ähnlichen Ansatz wie die in dieser Arbeit implementierten Klassen *BatchProcessor* und *Network*, inklusive ihren dazugehörigen, modularen Schichtklassen. Nur werden noch mehr Konfigurationen gegeben. Zusätzlich bewahrt dies vor der Ungewissheit, ob die fundamentalen Algorithmen auch korrekt implementiert sind.

Abschließend kann überlegt werden, wie die Vorverarbeitung weiter angepasst werden könnte. Als Beispiel wurde bereits in Kapitel \ref{head:bereinigungsprozess} darauf eingegangen, dass mehrere Pixel auf einmal bereinigt werden könnten und nicht, wie in dieser Arbeit, jedes für sich. Dies könnte durch ein Erweitern der Klasse *preprocessor.Processor* erreicht werden. Es wird vor allem erhofft, dass möglicherweise damit die weißen Ritzen in Buchstaben, welche in Abbildung \ref{fig:schicht-vergleich-22} anzutreffenden sind, besser geschlossen werden können.

