# Anwendung

Die drei implementierten Module network, metric und preprocessor können wie in Abblidung \ref{fig:sequenz_training} angewendet werden. Es werden je einen Preprocessor für die Test- und Vidierungsdaten erstellt sowie eine *Recorder* Instanz, mit Hilfe einer Konfigurationsdatei.

![Sequenzdiagram: Konfiguration und training. \label{fig:sequenz_training}](images/ablauf_training.png)

Danach werden beliebig viele Layer mit beliebiger Anzahl Ein und Ausgänge Instanziiert. Dabei muss beachtet werden, dass die Anzahl Ausgänge des Vorgängers mit der Anzahl Eingäge der Volgeschicht übereinstimmen müssen. Diese Schichtinstanzen werden dann bei der Instanziierung des *Network* als Liste mitgegeben.

Wenn das Training durch die Methode *train* der *Network*-Instanz *net* gestartet wird, werden die Trainings- und Validierungsdaten, sowie der Recorder und die Hyperparameter mit übergeben.

Es wird nun über alle Trainingsdaten iteriert und Batchweise die das Netzwerk trainiert. Nach einer Epoche wird nun über alle Validierungsdaten iteriert und die Validierungskosten berechnet. Die Trainings und Validierungskosten werden danach vom *Recorder* aufgezeichnet. Sind die Validierungskosten kleiner als in einer vorhergehenden Validierung, wird das Netzwerk serialisiert und als Datei abgespeichert.

Ist das Training zu Ende, werden die Validierungskosten der besten Validierung ausgegeben. Das als Datei abgespeicherte beste Modell, kann später von dem Modul *Cleaner* geladen und verwendet werden.

