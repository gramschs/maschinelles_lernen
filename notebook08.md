---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 8. ML-Workflow: Datenvorverarbeitung

## 8.1 Fehlende Daten

Realistische Datensätze sind oft unvollständig. In einer Umfrage hat eine Person
mit einer Frage nichts anfangen können und daher nichts angekreuzt. Ein
Messsensor an der Produktionsanlage ist abends ausgefallen, was erst am nächsten
Morgen bemerkt wurde. Die Mitarbeitenden einer Arztpraxis sind im Urlaub und
lassen die Meldung der verabreichten Impfungen noch bis nach dem Urlaub liegen.
Es gibt viele Gründe, warum Datensätze unvollständig sind. In diesem Abschnitt
beschäftigen eir uns damit, fehlende Daten aufzuspüren und lernen einfache
Methoden kennen, damit umzugehen.

### Lernziele Kapitel 8.1

* Sie können in einem Datensatz mit **isnull()** fehlende Daten aufspüren und
  analysieren.
* Sie kennen die beiden grundlegenen Strategien, mit fehlenden Daten umzugehen:
  * **Elimination** (Löschen) und
  * **Imputation** (Vervollständigen).
* Sie können Daten gezielt mit **drop()** löschen.
* Sie können fehlende Daten mit **fillna()** vervollständigen.

### Fehlende Daten aufspüren mit isnull()

Wir arbeiten im Folgenden mit einem echten Datensatz der Verkaufsplattform
[Autoscout24.de](https://www.autoscout24.de), der Verkaufsdaten zu 1000 Autos
enthält. Sie können die csv-Datei hier herunterladen:
<https://gramschs.github.io/book_ml4ing/data/autoscout24_fehlende_daten.csv>
und in das Jupyter Notebook importieren. Alternativ können Sie die csv-Datei
auch über die URL importieren, wie es in der folgenden Code-Zelle gemacht wird.

```{code-cell}
import pandas as pd

url = 'https://gramschs.github.io/book_ml4ing/data/autoscout24_fehlende_daten.csv'
daten = pd.read_csv(url)

daten.info()
```

Wir hatten bereits festgestellt, dass die Anzahl der `non-null`-Einträge für die
verschiedenen Merkmale unterschiedlich ist. Offensichtlich ist nur bei 963 Autos
eine »Farbe« eingetragen und die »Leistung (PS)« ist nur bei 987 Autos gültig.
Am wenigsten gültige Einträge hat das Merkmal »Verbrauch (l/100 km)«, wohingegen
bei der Eigenschaft »Kilometerstand (km)« nur ein ungültiger Eintrag auftaucht.
Welche Einträge ungültig sind, können wir mit der Methode `isnull()` bestimmen.
Die Methode liefert ein Pandas DataFrame zurück, das True/False-Werte enthält.
True steht dabei dafür, dass ein Wert fehlt bzw. mit dem Eintrag `NaN`
gekennzeichnet ist (= not a number). Weitere Details finden Sie in der
[Pandas-Dokumentation →
isnull()](https://pandas.pydata.org/docs/reference/api/pandas.isnull.html).

```{code-cell}
daten.isnull()
```

Bereits in der zweiten Zeile befindet sich ein Auto, bei dem das Merkmal
»Verbrauch (l/100 km)« nicht gültig ist (ggf. müssen Sie weiter nach rechts
scrollen), den dort steht `True`. Wir betrachten uns diesen Eintrag:

```{code-cell}
daten.loc[1,:]
```

Bei dem Auto handelt es sich um einen Hybrid, vielleicht wurde deshalb der
»Verbrauch (l/100 km)« nicht angegeben. Ist das vielleicht auch bei den anderen
Autos der Grund? Wir speichern zunächst die isnull()-Datenstruktur in einer
eigenen Variable ab und ermitteln zunächst, wie viele Autos keinen gültigen
Eintrag bei diesem Merkmal haben. Dazu nutzen wir aus, dass der boolesche Wert
`False` bei Rechnungen als 0 interpretiert wird und der boolesche Wert `True`
als 1. Die Methode `.sum()` summiert pro Spalte alle Werte, so dass sie direkt
die Anzahl der ungültigen Werte pro Spalte liefert.

```{code-cell}
fehlende_daten = daten.isnull()

fehlende_daten.sum()
```

Jetzt lassen wir uns diese 109 Autos anzeigen, bei denen ungültige Werte beim
»Verbrauch (l/100 km)« angegeben wurden. Dazu nutzen wir die True-Werte in der
Spalte `Verbrauch (l/100 km)` als Filter für den ursprünglichen Datensatz.
Zumindest die ersten 20 Autos lassen wir uns dann mit der `.head(20)`-Methode
anzeigen.

```{code-cell}
autos_mit_fehlendem_verbrauch_pro_100km = daten[ fehlende_daten['Verbrauch (l/100 km)'] == True ]
autos_mit_fehlendem_verbrauch_pro_100km.head(20)
```

Bemerkung: Der Vergleich `== True` ist redundant und kann auch weggelassen werden.

Beim Kraftstoff werden alle möglichen Angaben gemacht: Hybrid, Benzin, Diesel
und Elektro. Wir müssten jetzt systematisch den fehlenden Angaben nachgehen. Für
Elektrofahrzeuge und ggf. Hybridautos ist die Angabe »Verbrauch (l/ 100 km)«
unsinnig. Aber das zweite Auto mit der Nr. 5 wird mit Benzin betrieben, da
scheint Nachlässigkeit beim Ausfüllen der Merkmale vorzuliegen. Beim fünften
Auto mit der Nr. 77 ist zwar der »Verbrauch (l/100 km)« nicht angegeben, aber
dafür der »Verbrauch (g/km)«. Daraus könnten wir den »Verbrauch (l/100 km)«
abschätzen und den fehlenden Wert ergänzen. Es gibt verschiedene Strategien, mit
fehlenden Daten umzugehen. Die beiden wichtigsten Verfahren zum Umgang mit
fehlenden Daten sind

1. Löschen (Elimination) und
2. Vervollständigung (Imputation).

Bei Elimination werden Datenpunkte (Autos) und/oder Merkmale gelöscht. Bei
Imputation (Vervollständigung) werden die fehlenden Werte ergänzt. Beide
Verfahren werden wir nun etwas detaillierter betrachten.

### Löschen (Elimination) mit drop()

Bei der Elimination (Löschen) können wir filigran vorgehen oder die
Holzhammer-Methode verwenden. Beispielsweise könnten wir entscheiden, das
Merkmal »Verbrauch (l/100 km)« komplett zu löschen und einfach nur den
»Verbrauch (g/km)« zu berücksichtigen. Aber ein kurzer Blick auf die Daten hatte
ja bereits gezeigt, dass diese Werte auch nur unzuverlässig gefüllt waren, auch
wenn sie technisch gültig sind. Wir löschen beide Merkmale. Dazu benutzen wir
die Methode `drop()` mit dem zusätzlichen Argument `columns=['Verbrauch (l/
100 km)', 'Verbrauch (g/km)']`. Da wir gleich zwei Spalten aufeinmal eliminieren
möchten, müssen wir die Spalten (Columns) als Liste übergeben. Danach überprüfen
wir mit der Methode `.info()`, ob das Löschen geklappt hat.

```{code-cell}
daten.drop(columns=['Verbrauch (l/100 km)', 'Verbrauch (g/km)'])
daten.info()
```

Leider hat der Befehl `drop()` nicht funktioniert! Was ist da los? Python und
Pandas verfolgen das Programmierparadigma »Explizit ist besser als implizit!«
Daher werden zwar werden durch den `drop()`-Befehl die beiden Spalten gelöscht,
aber der Datensatz `daten` selbst bleibt aus Sicherheitsgründen unverändert.
Möchten wir den Datensatz mit den gelöschten Merkmalen weiter verwenden, müssen
wir ihn in einer neuen Variable speichern oder die alte Variable `daten` damit
überschreiben. Wir nehmen eine neue Variable namens `daten_ohne_verbrauch`.

```{code-cell}
daten_ohne_verbrauch = daten.drop(columns=['Verbrauch (l/100 km)', 'Verbrauch (g/km)'])
daten_ohne_verbrauch.info()
```

Ein weiterer Datenpunkt weist einen ungültigen Eintrag für den »Kilometerstand
(km)« auf. Schauen wir zunächst nach, um welches Auto es sich handelt.

```{code-cell}
daten_ohne_verbrauch[ daten_ohne_verbrauch['Kilometerstand (km)'].isnull() ]
```

Bei den Einträgen des Autos sind noch mehr Probleme ersichtlich. Die
Erstzulassung war sicherlich nicht bei 37.500 km und das Jahr ist nicht 12/2020.
Wir können jetzt diesen Datenpunkt löschen oder den Datenpunkt reparieren.
Zunächst einmal der Code zum Löschen des Datenpunktes. Standardmäßig löscht die
`drop()`-Methode ohnehin Zeilen, also Datenpunkte, so dass wir ohne weitere
Optionen den Index der zu löschenden Datenpunkte angeben. Diesmal verwenden wir
die alte Variable um den reduzierten Datensatz zu speichern.

```{code-cell}
daten_ohne_verbrauch = daten_ohne_verbrauch.drop(708)
daten_ohne_verbrauch.info()
```

Wie Sie in der [Dokumentation Scikit-Learn →
drop()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html)
nachlesen können, gibt es zum expliziten Überschreiben der alten Variable auch
die Alternative, die Option `inplace=True` zu setzen. Welche Option Sie nutzen,
ist Geschmackssache.

Ob alle Angaben plausibel sind, ist nicht gesagt. Bei dem Peugeot mit dem Index
708 hatten wir ja gesehen, dass bei der Erstzulassung eine Kilometerangabe
stand. Tatsächlich gab es bereits erste Hinweise darauf, dass manche Werte
technisch gültig, aber nicht plausibel sind. Die Spalte mit dem Jahr
beispielsweise wurde beim Import als Datentyp Object klassifiziert. Zu erwarten
wäre jedoch der Datentyp Integer gewesen. Schauen wir noch einmal in den
ursprünglichen Datensatz hinein.

```{code-cell}
daten['Jahr'].unique()
```

Da bei dem Peugeot mit dem Index 708 das Jahr fälschlicherweise mit `12/2020`
angegeben wurde, hat dieser eine Text-Eintrag dazu geführt, dass die komplette
Spalte als Object klassifiziert wurde und nicht als Integer. Daher müssen stets
weitere Plausibilitätsprüfungen durchgeführt werden, bevor die Daten genutzt
werden, um statistische Aussagen zu treffen oder ein ML-Modell zu trainieren.

### Vervollständigung (Imputation) mit fillna()

Auch bei den Angaben zur Farbe fehlen Einträge. Zum Beispiel die Zeile mit
dem Index 2 ist unvollständig.

```{code-cell}
daten_ohne_verbrauch.loc[2, :]
```

Diesmal entscheiden wir uns dazu, diese Eigenschaft nicht wegzulassen.
ML-Verfahren brauchen aber immer einen gültigen Wert und nicht `NaN`. Wir müssen
daher den fehlenden Wert ersetzen. Eine Möglichkeit ist, eine Farbe zu erfinden,
z.B. 'bunt', oder die fehlenden Werte explizit durch einen Eintrag 'keine
Angabe' zu vervollständigen. Dazu benutzen wir die Methode `fillna()` (siehe
[Pandas-Dokumentation →
fillna()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)).
Die Vervollständigung soll nur die NaN-Werte der Spalte »Farbe« füllen. Daher
filtern wir zuerst diese Spalte und wenden darauf die `fillna()`-Methode an. Das
erste Argument der `fillna()`-Methode ist der Wert, durch den die NaN-Werte
ersetzt werden sollen (hier `'keine Angabe'`). Damit die Verwollständigung
explizit gespeichert wird, überschreiben wir die Spalte.

```{code-cell}
daten_ohne_verbrauch['Farbe'] = daten_ohne_verbrauch['Farbe'].fillna('keine Angabe')

# Kontrolle der Vervollständigung
daten_ohne_verbrauch.isnull().sum()
```

Wenn wir uns jetzt noch einmal die dritte Zeile ansehen, sehen wir, dass
`fillna()` funktioniert hat.

```{code-cell} ipython3
daten_ohne_verbrauch.loc[2,:]
```

Bei den PS-Zahlen haben wir ebenfalls nicht vollständige Daten vorliegen.
Diesmal haben wir nicht kategoriale Daten wie die Farben, sondern numerische
Werte. Daher bietet es sich hier eine zweite Methode der Ersetzung (Imputation)
an. Wenn wir überall da, wo keine PS-Zahlen vorliegen, den Mittelwert der
vorhandenen PS-Zahlen einsetzen, machen wir zumindest den Mittelwert des
gesamten Datensatzes nicht kaputt. Wir berechnen daher zuerst den Mittelwert mit
der Methode `.mean()` und nutzen dann die `fillna()`-Methode.

```{code-cell}
mittelwert = daten_ohne_verbrauch['Leistung (PS)'].mean()
print(f'Der Mittelwert der vorhandenen Einträge »Leistung (PS)« ist: {mittelwert:.2f}')

daten_ohne_verbrauch['Leistung (PS)'] = daten_ohne_verbrauch['Leistung (PS)'].fillna(mittelwert)
```

Noch einmal die Kontrolle, ob jetzt alle NaN-Werte eliminiert oder
vervollständigt wurden:

```{code-cell}
daten_ohne_verbrauch.isnull().sum()
```

Der Mittelwert der »Leistung (PS)« ist sehr hoch. Vielleicht haben wir doch den
Datensatz eher verschlechtert, indem wir fehlende Werte durch den Mittelwert
ersetzt haben. Beispielsweise könnte der Median eine bessere Alternative sein.
Auch könnten wir zunächst die Autos mit fehlenden PS-Zahlen weglassen, für die
übrigen Autos ein lineares Regressionsmodell oder einen Entscheidungsbaum
trainieren und damit die fehlenden PS-Zahlen abschätzen. Bei diesem Beispiel
wäre die beste Lösung zur Imputation der ungültigen Werte »Leistung (PS)« die
Umrechung der vorhandenen, gültigen Werte der Spalte »Leistung (kW)«.
Tatsächlich sind die beiden Merkmale redundant, da es sich um dasselbe Merkmal
in zwei verschiedenen Einheiten handelt, so dass wir die Spalte »Leistung (PS)«
auch entfernen könnten.

### Zusammenfassung und Ausblick Kapitel 8.1

Ein wichtiger Teil eines ML-Projektes beschäftigt sich mit der Aufbereitung der
Daten für die ML-Algorithmen. Dabei ist es nicht nur wichtig, in großen
Datensammlungen fehlende Einträge aufspüren zu können, sondern ein Gespür dafür
zu entwickeln, wie mit den fehlenden Daten umgegangen werden soll. Die
Strategien hängen dabei von der Anzahl der fehlenden Daten und ihrer Bedeutung
ab. Häufig werden unvollständige Daten aus der Datensammlung gelöscht
(Elimination) oder numerische Einträge durch den Mittelwert der vorhandenen
Daten ersetzt (Imputation). Wie kategoriale Daten für ML-Algorithmen aufbereitet
werden müssen, wird im nächsten Kapitel erklärt.

## 8.2 Trainings- und Testdaten

Bei den Entscheidungsbäumen und der linearen Regression haben wir mit der
`score()`-Methode bewertet, wie viele der Daten durch das Modell korrekt
prognostiziert wurden. Je näher der Score an 1 liegt, desto besser. Doch selbst
ein perfekter Score bedeutet nicht zwangsläufig, dass das Modell optimal ist. Es
könnte überangepasst (overfitted) sein und daher bei neuen, unbekannten Daten
schlechte Prognosen liefern. Im Folgenden beschäftigen wir uns mit der
Aufteilung von Daten in Trainings- und Testdaten.

### Lernziele Kapitel 8.2

* Sie verstehen, warum Daten in **Trainingsdaten** und **Testdaten** aufgeteilt
  werden.
* Sie können mit der Funktion **train_test_split()** Pandas-DataFrames in
  Trainings- und Testdaten aufteilen.
* Sie kennen das Konzept der **Kreuzvalidierung**.

### Auswendiglernen nützt nichts

Um die Herausforderungen bei der Modellauswahl zu verdeutlichen, betrachten wir
einen künstlich generierten Datensatz. Angenommen, wir hätten die folgenden 20
Messwerte erfasst und möchten ein Regressionsproblem lösen.

```{code-cell}
import pandas as pd 
import plotly.express as px

# Generierung Daten
daten = pd.DataFrame()
daten['Ursache'] = [1.8681193560547067, 0.18892899670288932, 1.8907374398595373, 0.8592639746974586, 0.7909152983890833, -1.1356420176784945, 1.905097819104967, -1.9750789791816405, -0.9880705504662242, -0.26083387038221684, 1.1175316871750098, -1.2092597015989877, 1.451972942396889, 1.933602708701251, -1.3446310343812051, 0.38933577573143685, -1.96405560932978, -0.45371486942548245, -1.8233597682740017, 1.8266118708569437]
daten['Wirkung'] = [18.06801933135814, 0.09048390063552635, 18.29951272892001, 4.02392603643671, 1.97091878521032, 6.799411114666941, 17.540101218695103, 21.051664199041685, 5.604758672240995, 0.38630710692300024, 5.261393705782588, 7.365977868421521, 10.701020062336028, 17.48514901635516, 11.263523310016517, 1.1522069460363902, 20.979929897937023, -0.08352624016486021, 18.258951764602635, 15.321589041941028]

# Visualisierung
fig = px.scatter(daten, x = 'Ursache', y = 'Wirkung', title= 'Künstlich generierte Messdaten')
fig.show()
```

Nun würden wir das folgende Modell implementieren. Der Name des Modells sagt
bereits alles!

```{code-cell}
from sklearn.metrics import r2_score

class AuswendigLerner:
    def __init__(self) -> None:
        self.X = None
        self.y = None

    def fit(self, X,y):
        self.X = X
        self.y = y

    def predict(self, X):
        return self.y
```

Wir trainieren unser Modell und lassen es dann bewerten. Um nicht selbst den
R²-Score implementieren zu müssen, verwenden wir die allgemeine Funktion aus
Scikit-Learn (siehe [Dokumentation Scikit-Learn →
r2_score()}(https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)).

```{code-cell}
# Adaption der Daten
X = daten[['Ursache']]
y = daten['Wirkung']

# Auswahl Modell und Training
mein_super_modell = AuswendigLerner()
mein_super_modell.fit(X, y)

# prediction
y_prognose = mein_super_modell.predict(X)

# check quality
score = r2_score(y,y_prognose)
print(f'Der R2-Score ist: {score:.2f}')
```

Ein R²-Score von 1, unser Modell scheint perfekt zu funktionieren! Doch wie
prognostiziert es neue Daten? Das Modell funktioniert zwar hervorragend für die
gegebenen Trainingsdaten, ist jedoch **nicht verallgemeinerbar**.

```{code-cell}
mein_super_modell.predict([[1.3]])
```

Anstatt für den x-Wert $1.3$ (Ursache) eine Prognose zu treffen, gibt das Modell
einfach die auswendig gelernten y-Werte (Wirkungen) aus.

### Daten für später aufheben

Bei der Modellauswahl und dem Training des Modells müssen wir zusätzlich
sicherstellen, dass das Modell verallgemeinerbar ist, das heißt, dass es auch
für neue, zukünftige Daten verlässliche Prognosen liefern kann. Da wir jedoch
sofort abschätzen wollen, wie gut das Modell auf neue Daten reagiert, und nicht
warten möchten, bis die nächsten Messungen vorliegen, legen wir jetzt schon
einen Teil der vorhandenen Daten zur Seite. Diese Daten nennen wir
**Testdaten**. Die verbleibenden Daten verwenden wir für das Training des
Modells ﹣ sie heißen **Trainingsdaten**. Später nutzen wir die Testdaten, um zu
überprüfen, wie gut das Modell bei Daten funktioniert, die nicht zum Training
verwendet wurden.

Für die Aufteilung in Trainings- und Testdaten verwenden wir eine dafür
vorgesehene Funktion von Scikit-Learn namens `train_test_split()` (siehe
[Dokumentation Scikit-Learn →
train_test_split()](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)).
Diese Funktion müssen wir aus dem Modul `sklearn.model_selection` importieren.
Dann übergeben wir `train_test_split()` die Daten, die aufgeteilt werden sollen,
und erhalten als Rückgabe zwei DataFrames: Der erste enthält die Trainingsdaten,
der zweite die Testdaten.

```{code-cell}
from sklearn.model_selection import train_test_split

daten_train, daten_test = train_test_split(daten)
```

Nun wollen wir sehen, welche Datenpunkte zu den Trainingsdaten und welche zu den
Testdaten gehören. Dazu fügen wir dem Datensatz ein neues Merkmal hinzu und
füllen es mit den Strings `'Trainingsdaten'` bzw. `'Testdaten'`. Anschließend
visualisieren wir die Datenpunkte wie oben, wobei die Punkte entsprechend ihrer
Zugehörigkeit (Trainings- oder Testdaten) eingefärbt werden.

```{code-cell}
# Anreicherung der Daten mit dem Splitstatus
daten.loc[daten_train.index,'Splitstatus'] = 'Traingsdaten'
daten.loc[daten_test.index, 'Splitstatus'] = 'Testdaten'

# Visualisierung
fig = px.scatter(daten, x = 'Ursache', y = 'Wirkung', color='Splitstatus', 
title='Künstlich generierte Messdaten')
fig.show()
```

Standardmäßig hält die Funktion `train_test_split()` 25 % der Daten als
Testdaten zurück. Ein schnelles Zählen der fünf Testdatenpunkte bestätigt dies.
Die Auswahl der Testdaten erfolgt zufällig, sodass jeder Durchlauf des Codes
eine andere Aufteilung der Daten erzeugt.

Die Funktion bietet aber auch Optionen, um die Aufteilung nach eigenen Wünschen
anzupassen:

* `test_size`: Mit der Option `test_size` kann ein anderer Anteil als 25 % für
  die Testdaten festgelegt werden. Möchte man zum Beispiel nur 10 % der Daten
  als Testdaten zurückhalten, kann man `test_size=0.1` einstellen. Der Anteil
  wird als Float zwischen 0.0 und 1.0 angegeben. Verwendet man stattdessen einen
  Integer, interpretiert Scikit-Learn diesen als Anzahl der Testdatenpunkte.
  `test_size=7` bedeutet also, dass sieben Datenpunkte als Testdaten verwendet
  werden.
* `random_state`: Die zufällige Auswahl der Testdaten erfolgt durch einen
  Zufallszahlengenerator, der bei jedem Durchlauf neu gestartet wird. Wenn wir
  zwar eine zufällige Auswahl wollen, aber den Neustart des
  Zufallszahlengenerators verhindern möchten, können wir den Ausgangszustand des
  Generators mit einem festen Wert (Integer) festlegen. Das ist vor allem für
  Präsentationen oder Lehrmaterialien nützlich.

```{code-cell}
daten_train, daten_test = train_test_split(daten, test_size=7, random_state=0)

# Aktualisierung des Splitstatus
daten.loc[daten_train.index,'Splitstatus'] = 'Traingsdaten'
daten.loc[daten_test.index, 'Splitstatus'] = 'Testdaten'

# Visualisierung
fig = px.scatter(daten, x = 'Ursache', y = 'Wirkung', color='Splitstatus', 
title='Künstlich generierte Messdaten')
fig.show()
```

### Idee der Kreuzvalidierung

Das Zurückhalten eines Teils der Daten als Testdaten hat den Nachteil, dass
weniger Daten für das Training zur Verfügung stehen. Besonders bei kleinen
Datensätzen kann dies dazu führen, dass das Modell ungenau oder schlecht
trainiert wird. Hier kommt die Kreuzvalidierung ins Spiel.

Die Idee der **Kreuzvalidierung** ist, die Daten in mehrere Teilmengen zu
unterteilen und das Modell mehrmals zu trainieren und zu testen, um die Leistung
besser beurteilen zu können. Schauen wir uns zunächst die zweifache
Kreuzvalidierung an:

Bei der zweifachen Kreuzvalidierung teilen wir die Daten in zwei Teilmengen, A
und B. Das Modell wird dann zweimal trainiert und getestet: einmal mit A als
Trainingsdaten und B als Testdaten, und einmal umgekehrt. Die endgültige
Modellbewertung ergibt sich aus dem Durchschnitt der beiden Testergebnisse.

Die dreifache Kreuzvalidierung funktioniert ähnlich, mit dem Unterschied, dass
die Daten in drei Teilmengen A, B und C aufgeteilt werden. In drei Durchläufen
wird jeweils mit zwei der Teilmengen trainiert und mit der dritten getestet:

* Im ersten Durchlauf wird mit A und B trainiert und mit C getestet.
* Im zweiten Durchlauf wird mit B und C trainiert und mit A getestet.
* Im dritten Durchlauf wird mit A und C trainiert und mit B getestet. Am Ende
wird der Durchschnitt der drei Testergebnisse als Maß für die Modellleistung
verwendet.

Dieses Verfahren lässt sich auf beliebig viele Teilmengen erweitern.
Scikit-Learn bietet dafür auch spezielle Funktionen zur effizienten Umsetzung
der Kreuzvalidierung. Eine detailliertere Betrachtung dieser Techniken erfolgt
jedoch in einem späteren Kapitel. An dieser Stelle soll lediglich das Konzept
der Kreuzvalidierung eingeführt werden.

### Zusammenfassung und Ausblick Kapitel 8.2

In diesem Abschnitt haben wir die Aufteilung von Daten in Trainings- und
Testdaten kennengelernt und die Funktion `train_test_split()` verwendet. Diese
Funktion wird uns in zukünftigen Kapiteln und Projekten begleiten. Zudem haben
wir eine erste Einführung in die Kreuzvalidierung erhalten, die wir später
ausführlicher behandeln werden.

## 8.3 Kodierung und Skalierung

ML-Algorithmen können nur Zahlen verarbeiten. In diesem Kapitel werden wir uns
zunächst damit beschäftigen, wie auch kategoriale Daten wie beispielsweise die
Farbe eines Autos verarbeitet werden können. Da viele ML-Modelle empfindlich
darauf reagieren, wenn die numerischen Werte in sehr unterschiedlichen
Größenordnungen liegen, beschäftigen wiruns auch mit der Sklaierung von
numerischen Daten.

### Lernziele Kapitel 8.3

* Sie können geordnete kategoriale (= ordinale) Daten mit Hilfe eines
  Dictionaries und der `replace()`-Methode als Zahlen kodieren.
* Sie können ungeordnete kategoriale (= nominame) Daten mit Hilfe der
  `get_dummies()`-Methode als Zahlen kodieren. Diese Methode nennt man
  **One-Hot-Kodierung**.
* Sie können numerische Daten skalieren, indem Sie
  * mit dem **MinMaxScaler** die Daten **normieren** oder
  * mit dem **StandardScaler** die Daten **standardisieren**.

### Kodierung von kategorialen Daten

Bei den Beispielen zur linearen Regression haben wir zur Prognose des
Verkaufspreises nur numerische Daten genutzt, wie beispielsweise den
Kilometerstand. Es gibt jedoch weitere Merkmale, die die Kaufentscheidung
beeinflussen, wie der Kraftstofftyp (Diesel oder Benzin) und die Marke des
Autos. Diese würden wir ebenfalls gerne in die Prognose des Preises einfließen
lassen. Dazu müssen die kategorialen Daten, die in der Regel durch den Datentyp
String gekennzeichnet sind, vorab in Integers oder Floats umgewandelt werden. Je
nachdem, ob die kategorialen Daten geordnet oder ungeordnet sind, gibt es
verschiedene Vorgehensweisen, wie wir uns im Folgenden anhand eines Beispiels
erarbeiten.

Wir laden einen Datensatz mit Verkaufsdaten der Plattform
[Autoscout24.de](https://www.autoscout24.de). Sie können die csv-Datei hier
herunterladen:
<https://gramschs.github.io/book_ml4ing/data/autoscout24_kodierung.csv> und in
das Jupyter Notebook importieren. Alternativ können Sie die csv-Datei auch über
die URL importieren, wie es in der folgenden Code-Zelle gemacht wird. Mit der
Methode `.info()`lassen wir uns anzeigen, welchen Datentyp die Merkmale haben.

```{code-cell}
import pandas as pd 

url = 'https://gramschs.github.io/book_ml4ing/data/autoscout24_kodierung.csv'
daten = pd.read_csv(url)

daten.info()
```

Wir sehen

* 8 Merkmale mit Datentyp `object`: Marke, Modell, Farbe, Erstzulassung,
  Getriebe, Kraftstoff, Bemerkungen, Zustand,
* 4 Merkmale mit Datentyp `int64`: Jahr, Preis (Euro), Leistung (PS), Leistung
  (kW)
* 2 Merkmale mit Datentyp `float64`: Verbrauch (l/100 km) und Kilometerstand
  (km).

Als erstes betrachten wir geordnete Daten.

#### Geordnete kategoriale Daten mit zwei Kategorien (binär ordinale Daten)

Als erstes betrachten wir das Merkmal »Getriebe«. Mit der Methode `.unique()`
ermitteln wir, wie viele verschiedene Kategorien es für dieses Merkmal gibt.

```{code-cell}
daten['Getriebe'].unique()
```

Es gibt nur zwei Kategorien: Automatik und Schaltgetriebe. Diese beiden Werte
wollen wir durch Integers ersetzen:

* Automatik --> 0 und
* Schaltgetriebe --> 1.

Pandas bietet dazu die Methode `replace()` an. Bei der Verwendung dieser Methode
darf sich der Datentyp nicht ändern (in Pandas Version 2 noch erlaubt, ab
Version 3 verboten). Daher kodieren wir zunächst die Strings `'Automatik'` und
`'Schaltgetriebe'` als die Strings `'0'` und `'1'`mit Hilfe eines Dictionaries:

```{code-cell}
getriebe_kodierung = {
  'Automatik': '0',
  'Schaltgetriebe': '1',
}
```

Dann verwenden wir `replace()`, um die Ersetzung vorzunehmen. Zuletzt wandeln
wir die Strings `'0'` und `'1'` noch mit der Methode `astype()` in Integers um:

```{code-cell}
daten['Getriebe'] = daten['Getriebe'].replace(getriebe_kodierung)
daten['Getriebe'] = daten['Getriebe'].astype('int')

# Kontrolle
daten['Getriebe'].unique()
```

#### Geordnete kategoriale Daten (ordinale Daten)

Für das Merkmal »Zustand« gibt es vier Kategorien.

```{code-cell}
daten['Zustand'].unique()
```

Die vier Zustände haben eine Ordnung, denn ein Neuwagen ist wertvoller als ein
Jahreswagen. Der Jahreswagen wiederum ist im Allgmeinen wertvoller als der junge
Gebrauchtwagen. Am wenigsten wertvoll ist der Gebrauchtwagen. Durch diese
Ordnung ist es sinnvoll, beim Kodieren der Zustände durch Integers die Ordnung
beizubehalten. Ob wir jetzt die 0 für den Neuwagen vergeben und die 3 für den
Gebrauchtwagen oder umgekehrt, ist Geschmackssache.

```{code-cell}
zustand_kodierung = {
  'Gebrauchtwagen': '0',
  'junger Gebrauchtwagen': '1', 
  'Jahreswagen': '2',
  'Neuwagen': '3'
}

daten['Zustand'] = daten['Zustand'].replace(zustand_kodierung)
daten['Zustand'] = daten['Zustand'].astype('int')

# Kontrolle
daten['Zustand'].unique()
```

#### Ungeordnete kategoriale Daten (nominale Daten): One-Hot-Kodierung

Anders verhät es sich bei den ungeordnetem kategorialen Daten wie beispielsweise
den Farben der Autos.

```{code-cell}
daten['Farbe'].unique()
```

14 verschiedene Farben haben die Autos in dem Datensatz. Es wäre jedoch falsch,
nun Integers von 0 bis 13 zu vergeben, denn das würde eine Ordnung der Farben
voraussetzen, die es nicht gibt. Wir verwenden daher das Verfahren der
**One-Hot-Kodierung**. Anstatt einer Spalte mir den Farben führen wir 14 neue
Spalten mit den Farben 'grau', 'grün', 'schwarz', 'blau', usw. ein. Wenn ein
Auto die Farbe 'grau' hat, notieren wir in der Spalte 'grau' in dieser Zeile
eine 1 und in den übrigen 13 Spalten mit den anderen Farben eine 0. So können
wir die Farben numerisch kodieren, ohne eine Ordnung der Farben einzuführen, die
es nicht gibt. Pandas bietet dafür die Methode `get_dummies()`an. Schauen wir
uns zunächst an, was diese Methode bewirkt.

```{code-cell}
pd.get_dummies(daten['Farbe'])
```

Damit haben wir die Spalte »Farbe« nun durch 14 Spalten kodiert. Wir könnten nun
im ursprünglichen Datensatz die Spalte »Farbe« löschen und die neuen 14 Spalten
hinzufügen. Tatsächlich erledigt das Pandas bereits für uns, wenn wir die
Methode etwas modifiziert aufrufen. Mit dem Argument `data=` übergeben wir nun
den kompletten Datensatz und mit dem Argument `columns=` spezifizieren wir die
Liste der ungeordneten kategorialen Daten, die One-Hot-kodiert werden sollen.

```{code-cell}
daten = pd.get_dummies(data=daten, columns=['Farbe'])
daten.head()
```

Die neuen Spaltennamen sind eine Kombination aus dem alten Spaltennamen »Farbe«
und den Kategorien.

### Skalierung von numerischen Daten

Nachdem wir uns intensiv mit den kategorialen Daten beschäftigt haben,
betrachten wir nun die numerischen Daten. Wir laden den Original-Datensatz und
entfernen die kategorialen Daten.

```{code-cell}
url = 'https://gramschs.github.io/book_ml4ing/data/autoscout24_kodierung.csv'
daten = pd.read_csv(url)

daten = daten.drop(columns=['Marke', 'Modell', 'Farbe', 'Erstzulassung', 
                            'Getriebe', 'Kraftstoff','Bemerkungen', 'Zustand'])
daten.info()
```

Ein erster Blick auf die Daten zeigt bereits, dass die Eigenschaftswerte in
unterschiedlichen Bereichen liegen.

```{code-cell}
daten.head()
```

Der Verbrauch gemessen in Litern pro 100 Kilometer liegt zwischen 5 und 10,
wohingegen der Kilometerstand die 100000 km übersteigt.Das zeigt auch die
Übersicht der statistischen Kennzahlen:

```{code-cell} ipython3
daten.describe()
```

Damit ist auch der Boxplot nur noch schwer lesbar:

```{code-cell} ipython3
import plotly.express as px 

fig = px.box(daten)
fig.show()
```

Das hat auch Auswirkungen auf das Training der ML-Modelle. Daher beschäftigen
wir uns nun mit der Skalierung von Daten.

Sind die Bereich der Daten von ihren Zahlenwerten sehr verschieden, sollten alle
numerischen Werte in dieselbe Größenordnung gebracht werden. Dieser Vorgang
heißt **Skalieren** der Daten. Gebräulich sind dabei zwei verschiedene Methoden:

* **Normierung** und
* **Standardisierung**.

#### Normierung

Bei der Normierung wird festgelegt, dass alle Zahlenwerte in einem festen
Intervall liegen. Besonders häufig wird das Intervall $[0,1]$ genommen. Die
Verbrauch (l/ 100 km), der zwischen 3.5 und 14.9 liegt, würde so transformiert
werden, dass das Minimum 3.5 der 0 entspricht und das Maximum 14.9 der 1.
Genauso würde mit den anderen Eigenschaften verfahren werden. Wir nutzen zur
praktischen Umsetzung Scikit-Learn.

Damit keine Informationen über die Testdaten in das Training des ML-Modells
sickern (Data Leakage), wird die Normierung an das Minimum und das Maximum der
Trainingsdaten angepasst und ggf. für die Testdaten angewendet. Damit können
einzelne Testdaten auch außerhalb des Intervalls $[0,1]$ liegen. Wir splitten
daher zunächst unsere Daten in Trainings- und Testdaten.

```{code-cell}
from sklearn.model_selection import train_test_split

daten_train, daten_test = train_test_split(daten, random_state=0)
```

Dann importieren wir die Klasse `MinMaxScaler` aus dem Untermodul
`sklearn.preprocessing` und erzeugen ein MinMaxScaler-Objekt:

```{code-cell}
from sklearn.preprocessing import MinMaxScaler

# Auswahl Skalierungsmethode: Normierung
normierung = MinMaxScaler()
```

Jetzt wird das Minimum/Maximum jeder Spalte bestimmt, also der MinMaxScaler an
die Trainingsdaten angepasst. Daher ist es nicht verwunderlich, dass die Methode
`fit()` genannt wurde. Dem MinMaxScaler werden also die Trainingsdaten
übergeben:

```{code-cell}
normierung.fit(daten_train)
```

Zuletzt erfolgt die Transformation der Daten mit der `transform()`-Methode. Dazu
werden einmal die Trainingsdaten und einmal die Testdaten dem angepassten
MinMaxScaler übergeben und die transformierten Daten in neuen Variablen
gespeichert.

```{code-cell}
# Transformation der Trainungs- und Testdaten
X_train_normiert = normierung.transform(daten_train)
X_test_normiert = normierung.transform(daten_test)
```

Wir schauen in 'X_train_normiert' hinein:

```{code-cell} ipython3
print(X_train_normiert)
```

Die Normierung der Daten scheint funktioniert zu haben. Alle Werte liegen
zwischen 0 und 1. Gleichzeitig haben wir aber die Pandas-DataFrame-Datenstruktur
verloren. Die Normierung ist nicht für uns Menschen gedacht, sondern für den
ML-Algorithmus. Daher nutzt Scikit-Learn die Transformation der Daten
gleichzeitig für die Umwandlung in das speichereffizientere NumPy-Array, das für
den ML-Algorithmus gebraucht wird.

#### Standardisierung

Oft sind Daten normalverteilt. Die Standardisierung berücksichtigt das und
transformiert nicht auf ein festes Intervall, sondern verschiebt den Mittelwert
auf 0 und die Varianz auf 1. Die normalverteilten Daten werden also
standardnormalverteilt. Auch das lassen wir Scikit-Learn erledigen:

```{code-cell} ipython3
from sklearn.preprocessing import StandardScaler

# Auswahl Skalierungsmethode: Standardisierung
standardisierung = StandardScaler()

# Analyse: jede Spalte wird auf ihr Minimum und ihre Maximum hin untersucht
# es werden immer die Trainingsdaten verwendet
standardisierung.fit(daten_train)

# Transformation der Trainungs- und Testdaten
X_train_standardisiert = standardisierung.transform(daten_train)
X_test_standardisiert = standardisierung.transform(daten_test)

print(X_train_standardisiert)
```

Auch hier geht die Pandas-DataFrame-Struktur verloren.

### Zusammenfassung und Ausblick

Kategoriale Daten müssen kodiert werden, damit sind in einem ML-Algorithmus
verarbeitet werden können. Geordnete kategoriale (ordinale) Daen können dabei
über ein Dictionary und die `replace()`-Methode kodiert werden. Für ungeordnete
kategoriale (nominale) Daten muss die One-Hot-Kodierung verwendet werden.

Auch numerische Daten müssen häufig für ML-Algorithmen aufbereitet werden, vor
allem, wenn die Daten in sehr unterschiedlichen Zahlenbereichen liegen. Bei den
bisher eingeführten ML-Modellen lineare Regression und Entscheidungsbäumen ist
die Skalierung der numerischen Daten nicht notwendig. Erst die nachfolgenden
ML-Modelle werden davon Gebrauch machen.

## Übungen

### Aufgabe 8.1

Eine Abalone oder ein Seeohr ist eine Schnecke mit Schale, die einer Ohrmuschel
ähnelt (siehe <https://de.wikipedia.org/wiki/Seeohren>). Laden Sie den Datensatz
'abalone_DE.csv'. Ziel dieser Aufgabe ist ein Modell zu trainieren, das aus den
Angaben zu Geschlecht, Größe und Gewicht die Anzahl der Ringe prognostiziert.
Die Anzahl der Ringe +1.5 gibt das Alter der Abalone an.

1. Führen Sie eine Datenexploration durch. Dazu gehören insbesondere
   * Übersicht
   * statistische Kennzahlen der Eigenschaften
   * Visualisierungen der Eigenschaften
   * Analyse bzgl. Ausreißer
2. Bereinigen Sie den Datensatz. Dazu gehört insbesondere die Entfernung von
   Ausreißern.  
3. Wählen Sie ein Modell.
4. Bereiten Sie die Daten für das Modell auf. Dazu gehört insbesondere auch der
   Split in Trainings- und Testdaten.
5. Validieren Sie das Modell. Erhöhen Sie die Modellkomplexität und beurteilen
   Sie, ob Over- oder Underfitting vorliegt.

```{code-cell}
#
```

### Aufgabe 8.2

Der Datensatz
'statistic_id226994_annual-average-unemployment-figures-for-germany-2005-2022.csv'
stammt von Statista. Die Daten beschreiben die Entwicklung der
Arbeitslosenzahlen (in Mio.) seit 1991. Im Original-Excel sind einige
Ungereimtheiten, die sich auch so im csv-File befinden.

1. Korrigieren Sie den Datensatz zuerst mit einem Texteditor.
2. Führen Sie dann eine explorative Datenanalyse durch (Übersicht, statistische
   Kennzahlen, Boxplot und Visualisierung der Arbeitslosenzahlen abhängig vom
   Jahr.)
3. Wählen Sie mehrere ML-Modelle aus. Adaptieren Sie die Daten für das Training
   und lassen Sie die gewählten ML-Modelle trainieren.
4. Validieren Sie Ihr Modell: ist es geeignet? Bewerten Sie die Modelle bzgl.
   Over- und Underfitting.
5. Visualisieren Sie eine Prognose von 1990 bis 2030.

```{code-cell}
#
```
