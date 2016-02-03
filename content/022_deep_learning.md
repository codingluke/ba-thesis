## Deep-Learning

### Definition

Sobald ein kNN über mehr als eine versteckte Schicht verfügt, handelt es sich um ein tiefes kNN. Kommen solche tiefe kNN zum Einsatz wird generell von *Deep-Learning* gesprochen.

Das Verwenden von mehren Schichten, kann sich dadurch Positiv auf das Resultat auswirken, dass jede weitere Schicht die gelernten Eigenschaften der vorhergehenden wiederverwenden und verfeinern kann. Dies führt zu höheren, abstrakteren Eigenschaften pro zusätzlichen Schicht.

Vor allem in der Bild- und Sprachverarbeitung wurde durch tiefe kNN neue Standards gesetzt. Google Now, Microsoft's Cortana und auch Apple's Siri verwenden alle unter anderem tiefe kNN.

### Gradientenschwund \label{gradietenschwund}

Das Trainieren von tiefen kNN mit mehr als drei Schichten, war lange Zeit durch rechenkapazität beschränkt. Ein großes Problem stellte vor allem der *vanishing-gradient*, zu Detusch *Gradientenschwund* dar.

Der *Gradientenschwund* findet durch den *Backpropagation Algorithmus* zwischen den Schichten statt. Beim *Backpropagation Algorithmus* wird der Fehler von der Ausgangsschicht zur Eingangsschicht zurück geleitet. Nun verliert der Gradient der partiell abgeleiteten *Aktivierungsfunktion* jeder vorhergehenden Schicht an Größe, bis er fast nicht mehr existent ist. Dies hat zur Folge, dass die ersten Schichten in einem tiefen kNN sehr langsam bis gar nicht lernen.

### Unsupervised Feature Learning \label{unsubervised-feature-learning}

Ein weiteres Thema von *Deep-Learning* ist das *Unsupervised-Feature-Learning*, nicht beaufsichtigtes Lernen von Eigenschaften. Es wird versucht ein kNN dazu veranlassen Eigenschaften in den Eingabedaten selber zu erkennen und lernen.

Das *Unsupervised-Feature-Learning* ist mit dem Gruppieren unbekannter Daten anhand ähnlichen Eigenschaften (Clustering) zu vergleichen. Es werden Zusammenhänge neuer Daten ohne Zieldaten gefunden, wobei beim *Supervised-Feature-Learning* das Resultat mit erlesenen Zieldaten validiert wird.

### Autoencoder

Das Konzept des *Autoencoder* wird von Bengio Yoshua im Artikel "Learning Deep Architectures for AI" [@Bengio09] beschrieben und ist ein Ansatz die Probleme der vorhergehenden Kapitel \ref{gradietenschwund} und \ref{unsubervised-feature-learning} zu lösen.

Der *Autoencoder* ist ein kNN mit der Form $X-h-X$, wobei die Eingangs- und Ausgangsschicht (X) dieselbe Anzahl Neuronen besitzen und die unsichtbare Schicht (h) eine beliebige Größe haben kann.

![Autoencoder: Wiederverwendung der unsichtbaren Schicht \label{fig:autoencoder}](images/autoencoder.png)

Es wird versucht die zu lernende Daten so in der unsichtbare Schicht abzubilden (encode), damit sie wieder möglichst gut auf die Ausgangsschicht reproduziert (decode) werden können. Der *Autoencoder* versucht somit die *Identitätsfunktion* zu lernen.

Die gelernten Gewichte aus der unsichtbaren Schicht, bilden die Trainingsdaten in einer abstrakteren Ebene ab und können, wie in Abbildung \ref{fig:autoencoder}, für weitere Netze als Initialgewichte verwendet werden. Gelernt wurden diese Gewichte ohne zusätzliche Zieldaten. Somit handelt es sich um *Unsupervised-Feature-scoreLearning*.

### Denoising Autoencoder

In der Arbeit "Extracting and Composing Robust Features with Denoising Autoencoders" [@VincentPLarochelleH2008] wird bewiesen, dass die Identitätsfunktion noch abstrakter gelernt werden kann, wenn die Trainingsdaten bei der Eingabe zufällig verrauscht werden.

Ein *Denoising Autoencoder* ist somit ein *Autoencoder* welche die Eingangsdaten zuerst zufällig verrauscht, auf die unsichtbare Schicht abbildet und schlussendlich versucht diese bereinigt an der Ausgangsschicht auszugeben.

Das Rauschen veranlasst den *Autoencoder* dazu, die wesentlichen Eigenschaften der Daten zu finden. Ansonsten ist es nicht möglich diese zu bereinigen. Ist das Training erfolgreich, wurde das "Wesen" der Daten erkannt und die Gewichte und Biase der unsichtbaren Schicht können als Initialgewichte verwendet werden.

### Stacked Denoising Autoencoder

Der *Stacked Denoising Autoencoder*, ist die Kombination von mehreren Autoencodern zu einem "Stapel" und soll dem *Gradientenschwund* beisteuern. Es werden mehrere Autoencoder so aneinander gekoppelt, dass die unsichtbare Schicht des Vorgehenden zur Eingangsschicht des Nachfolgenden wird.

![Stacked Autoencoder mit zwei unsichtbaren Schichten](images/stacked_autoencoder.png)

Die Autoencoder werden einzeln, beim vordersten beginnend, autonom Trainiert. So werden die unsichtbaren Schichten nicht alle miteinander, sondern nacheinander Trainiert und umgehen den *Gradientenschwund*.

Sind alle Schichten fertig Trainiert, wird das gesamte Netz nochmals *supervied* Nachtrainiert (fine-tunig).

### ReLU / Rampenfunktion

Beim *Fine-Tuning* des Netzes ist das Problem vom *Gradientenschwund* immer noch vorhanden. Die *Aktivierungsfunktion* *ReLU* hat sich im *Deep-Learning* dadurch durchgesetzt, dass bei ihr der *Gradientenschwund* nicht so ausgeprägt ist wie bei der *Sigmoid-Funktion*. [@GlorotBB11]

