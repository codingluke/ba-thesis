# Einleitung \label{head:einleitung}

## Motivation \label{head:motivation}

Es existieren immer noch immense Wissensbestände, welche die Menschheit seit der Erfindung der Schrift niedergeschrieben hat, in reiner Papierform. Diese Art der Wissenskonservierung hat sich über Jahrtausende bewährt. Ein großer Nachteil dieses Mediums liegt jedoch im effizienten Durchsuchen.

Neue Arten der Wissensspeicherung in Form von digitaler Schrift, bieten den Vorteil, das Wissen durch Volltext-Indexierung effizient durchsuchbar zu machen. Um diese Errungenschaft auch auf alte Schriften anwenden zu können müssen diese digitalisiert werden.

Durch die jahrelange Archivierung sind die Schriften oft nicht mehr gut leserlich. Dazu kann der Staub, Luftfeuchtigkeit, die Sonne oder auch ungeschickter Umgang mit Flüssigkeiten geführt haben. Diese Verunreinigung kann dazu führen, dass beim Digitalisieren diverse Buchstaben und Wörter nicht korrekt erkannt werden.

An diesem Punkt möchte diese Arbeit ansetzen. Mit Hilfe von künstlichen neuronalen Netze, soll eine Möglichkeit erarbeitet werden, wie die eingescannten Schriften, vor der eigentlichen Schrifterkennung, bereinigt werden, damit der Schrifterkennung weniger Fehler unterlaufen.

## Zielstellung \label{head:zielstellung}

Das Hauptziel dieser Bachelorarbeit ist, ein künstliches neuronales Netzwerk, kNN, zu entwerfen, implementieren und
trainieren, welches möglichst gut im Kaggle Wettbewerb “Denoising Dirty Documents” abschneidet. [@kaggleDDD]

Im Wettbewerb geht es darum verrauschte Bilder automatisch so zu bereinigen, dass das darin enthaltene Schriftbild optimal hervorgehoben wird.

Es wird auf verschiedene Architekturmodelle von kNN zurückgegriffen, welche untereinander verglichen werden. Somit ist ein weiteres Ziel der Arbeit die Stärken und Schwächen unterschiedlicher kNN Architekturen aufzuzeigen. Vor allem wird der Unterschied von einschichtigen kNN zu mehrschichtigen, tiefen kNN untersucht. Bei den mehrschichtigen kNN wird ein besonderes Augenmerk auf die Methode der geschichteten Autoencoder gelegt.

Eine weitere Zielsetzung ist es, eine Architektur zur komfortablen Hyperparametersuche und Analyse des Lernvorgans herauszuarbeiten. Diese Architektur wird direkt für das Lernen und Analysieren der verschiedenen kNN dieser Bachelorarbeit angewendet.

## Aufbau der Arbeit \label{head:aufbau}

Die Arbeit ist in sieben aufeinander aufbauenden Hauptkategorien unterteilt. Angefangen mit der Einleitung folgt ein Kapitel über die Grundlagen der in der Arbeit verwendeten Techniken.

Nachdem die Grundlagen erörtert wurden, wird in der Aufgabenstellung darauf eingegangen, was genau das Ziel und die Randbedingungen der Arbeit ist und welche alternativen Ansätze existieren.

Darauf folgt das Kapitel Entwurf, in welchem die benötigten Prozesse und Umsetzungsvarianten analysiert und aufgezeigt werden. Die gefundenen Prozesse werden dann im Kapitel Implementation in der Programmiersprache *Python* umgesetzt.

Mit Hilfe des zuvor implementieren Programm wird im Kapitel Evaluierung versucht ein möglichst gutes kNN zu Konfigurieren und Trainieren. Es werden verschiedene Konfigurationen und Trainingsvarianten gegenübergestellt.

Schlussendlich wird im Schlussteil eine kritische Retrospektive der Arbeit niedergeschrieben und einen Ausblick auf weitere Möglichkeiten gegeben.


