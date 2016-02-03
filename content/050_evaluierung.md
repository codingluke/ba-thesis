# Evaluierung

## Vorverarbeitung

### Subbild generierung \label{head:sliding_window}

Im Gesamten müssen für die Trainingsdaten laut der Berechnung \ref{lst:patches_calc} 28460160 Subbilder, wie im Kapitel \ref{head:vor-nachverarbeitung} beschrieben, generiert werden. Diese Subbilder haben je nach Größe der gewählten Überlappung eine unterschiedliche Bildgröße. Bei einer Überlappung von 2 Pixel, besitzen sie die Bildgröße (5x5) und bestehen somit aus 25 Pixeln.

```{caption="Kalkulation der zu generierenden Unterbilder" frame=single label=lst:patches_calc .java}
48 Bilder * 540p * 258p + 96 Bilder * 540p * 420p = 28460160 Unterbilder
28560160 Bilder * 25 Pixel = 714004000 Pixel
714004000 * (float32 -> 4 Bytes) -> 2856016000 Bytes -> 2.856 GB
```

In einem Benchmark wurden die drei implementierten Generierungsverfahren einander gegenübergestellt und die Resultate in der Tabelle \ref{table:sliding-window} aufgelistet. Hiebei ist ersichtlich, dass die nicht durchmischte Generierung anhand der *numpy*-Matrix-Index-Operationen mit 62 Sekunden am schnellsten ist.

| Verfahren          | Sekunden | Minuten | Maschine  |
|:-------------------|:--------:|:-------:|----------:|
| Schleifen          | 189      | 3,15    |deepgreen02|
| Schleifen, Zufall  | 319      | 5,2     |deepgreen02|
| Matrix-Op.         | 62       | 1       |deepgreen02|
| Matrix-Op, Zufall  | 173      | 2,8     |deepgreen02|
| Kompletter Zufall  | 100      | 1,6     |deepgreen02|

  : Vergleich der Sliding Window Algorithmen. \label{table:sliding-window}

Wenn nun die Zufälligkeit mit einberechnet wird, werden die Vorteile der *Totalen-Zufall* Methode sichtbar. Diese benötigt mit 100 Sekunden erheblich länger als die sortierte *numpy* Optimierte variante, dafür müssen die Daten jedoch nicht mehr durchmischt werden.

Es hat sich ergeben, dass das Durchmischen mehr Zeit in Anspruch nimmt als die Generierung der Subbilder. Dadurch ist die *Totalen-Zufall* Methode die schnellste Zufallsvariante.

Zu beachte ist, dass nicht garantiert ist, dass jedes Subbild berücksichtigt wird. Wiederum können andere Subbilder doppelt vorkommen.

![Vergleich: Batchweise oder Komplette Zufälligkeit \label{fig:batch_vs_fully}](images/fully_vs_batch_random.png)

In der Abblidung \ref{fig:batch_vs_fully} ist zu sehen, dass die komplette Zufallsvariante zwar bei den Trainingskosten nicht besser abschneidet als die Batchweise zufälligkeit, bei den Validierungskosten erreicht sie jedoch ein besseres Resultat. Ebenfalls im Verlauf Trainingskosten ist eine verbesserte Regelmäßigkeit abzulesen.

Für weitere Trainingsdruchläufe, wird die Komplette Zufälligkeit verwendet.

## Einfaches kNN

### Minibatch Size

Die Größe des *Minibatch* definiert wie viele Datensätze parallel trainiert werden. Diese Größe hat zwei wesentliche Einflüsse auf das Trainingsverhalten.

Als Vorteil benötigt das Trainieren einer Epoche bei einem größeren *Minibatch* weniger Zeit, da durch Vektorberechnungen parallel mehrere Datensätze direkt auf der *GPU* berechnet werden. Zusätzlich werden durch größere Gruppen weniger Kopiervorgänge zwischen dem CPU und GPU Arbeitsspeicher getätigt, was die Performance zusätzlich steigert.

Als Nachteil kann ein zu großen *Minibatch* den Trainingsfortschritt pro Epoche verringern, da nur am Ende jedes *Minibatch* Durchlauf die Gewichte angepasst werden. Dadurch werden die Gewichte bei einem kleineren *Minibatch* öfter angepasst.

![Trainingsverlauf  \label{fig:mb-vergleich}](images/training_minibatch_vergleich.png)

In der Abblidung \ref{fig:mb-vergleich} ist zu sehen, wie der Trainingsfortschritt pro Epoche sich verringert, um so größer der *Mini-Batch* gewählt wird. Gleichzeitig verringert sich aber auch die Lerndauer. Um möglichst viele Trainingsvariationen prüfen zu können, wird eine *Mini-Batch-Größe* von $500$ gewählt. Für das beste Resultat wird für die Kaggle Kompetition mit einer kleineren von $200$ Trainiert.

### Fixe gegenüber dynamischer Lernrate

In der Abbildung \ref{fig:fix_dyn_eta} ist der Trainingsablauf einer Fixen Lernrate mit einer dynamisch abnehmender Lernrate gegenübergestellt. Dabei ist vor allem bei den Validierungskosten den Unterschied sichtbar.

![Unterschiedliche Trainingsverhalten bei fixer oder dynamischer Lernrate \label{fig:fix_dyn_eta}](images/lernrate_fix_vs_decrease.png)

Anfangs ist bei beiden Validierungskurven eine sprunghafte Verbesserung sichtbar. Ab der zweiten Epoche ist das Minimum bereits fast erreicht.
Nun werden größere Sprünge bei die Kurve mit dem fixen Lernrate sichtbar, wobei die Kurve mit der dynamische Lernrate konstant bleibt. Dies, da die dynamische Lernrate pro Epoche verkleinert wird, womit der Lernvorgang verlangsamt rsp. verfeinert wird.

### Größe der unsichtbaren Schicht




