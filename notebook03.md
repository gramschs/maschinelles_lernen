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

# 3. Pandas und Plotly anstatt Excel

## 3.1 Pandas Series

Eine **Series** ist eine von zwei grundlegenden Datenstrukturen des
Pandas-Moduls. Die Series dient vor allem dazu, Daten zu verwalten und
statistisch zu erkunden. Bevor wir die neue Datenstruktur näher beleuchten,
machen wir uns aber zuerst mit dem **Modul Pandas** vertraut.

### Lernziele 3.1

* Sie können erklären, was ein **Modul** in Python ist.
* Sie kennen das Modul **Pandas** und können es mit seiner üblichen Abkürzung
  **pd** importieren.
* Sie kennen die Pandas-Datenstruktur **Series**.
* Sie wissen, was ein **Index** ist.
* Sie können aus Listen eine Series-Objekt erzeugen und mit einem Index
  versehen.
* Sie können mit Series-Objekten rechnen.
* Sie können die Elemente eines Series-Objektes mit **sort_values()**
  aufsteigend und absteigend sortieren lassen.

### Das Modul Pandas

Stellen Sie sich vor, Sie möchten Spaghetti mit Tomatensauce kochen. Am
einfachsten ist es, die Spaghetti-Nudeln und die Tomatensauce im Supermarkt zu
kaufen. Sie können aber auch nur die Nudeln kaufen und die Tomatensauce selbst
aus Tomaten und Basilikum zubereiten. Oder Sie gehen noch einen Schritt weiter
und machen sogar die Nudeln aus Mehl, Eiern, Wasser und Salz selbst.

In der Programmierung verhält es sich ähnlich. Sie können alle Funktionalitäten
(d.h. die Spagehtti oder die Tomatensauce) selbst programmieren. Oder sie
verwenden schon fertige Komponenten und setzen Sie so zusammen, wie Sie es zur
Lösung ihres Problems brauchen. Eine Sammlung von fertigen Python-Komponenten zu
einem bestimmten Thema wird **Modul** genannt. In anderen Programmiersprachen
oder allgemein in der Informatik nennt man eine solche Sammlung auch
**Bibliothek** oder verwendet den englischen Begriff **Library**.

**Was ist ... ein Modul?**
Ein Modul (oder eine Bibliothek oder eine Library) ist eine Sammlung von
Python-Code zu einem bestimmten Thema, der als Werkzeug für eigene Programme
eingesetzt werden kann.

Um ein Modul in Python benutzen zu können, muss es zunächst einmal installiert
sein. Um dann die Funktionen, Klassen, Datentypen oder Konstanten benutzen zu
können, die das Modul zur Verfügung stellt, wird es importiert. Wir werden in
dieser Vorlesung sehr intensiv das Modul **Pandas** verwenden. Pandas ist ein
Modul zur Verarbeitung und Analyse von Daten. Es ist üblich, das Modul `pandas`
mit der Abkürzung `pd` zu importieren, damit wir nicht immer `pandas` schreiben
müssen, wenn wir Code aus dem Pandas-Modul benutzen.

```{code-cell}
import pandas as pd
```

Sollten Sie jetzt eine Fehlermeldung erhalten haben, ist das Pandas-Modul nicht
installiert. Installieren Sie zunächst Pandas beispielsweise mit `!conda install
pandas` oder `!pip install pandas`. Mit der Funktion `dir()` werden alle
Funktionalitäten des Moduls aufgelistet.

```{code-cell}
dir(pd)
```

Eine sehr lange Liste.

### Die Datenstruktur Series

Einfache Listen reichen nicht aus, um größere Datenmengen oder Tabellen
effizient zu speichern. Dazu benutzen Data Scientists die Datenstrukturen
`Series` oder `DataFrame` aus dem Pandas-Modul. Dabei wird **Series** für
Datenreihen genommen. Damit sind Vektoren gemeint, wenn alle Elemente der
Datenreihe aus Zahlen bestehen, oder eindimensionale Arrays. Die Datenstruktur
**DataFrame** wiederum dient zum Speichern und Verarbeiten von tabellierten
Daten, also sozusagen Matrizen, wenn alle Elemente Zahlen sind, oder
zweidimensionale Arrays.

Wir starten mit der Datenstruktur Series. Als Beispiel betrachten wir die
Verkaufspreise (in Euro) von zehn Autos. Die Daten stammen von der
Internetplattform [Autoscout24](https://www.autoscout24.de). Die Preise kommen
zunächst in eine Liste (erkennbar an den eckigen Klammern), aus der dann ein
Series-Objekt erzeugt wird.

```{code-cell}
preisliste = [1999, 35990, 17850, 46830, 27443, 14240, 19950, 15950, 21990, 50000]
preise = pd.Series(preisliste)
print(preise)
```

Was ist aber jetzt der Vorteil von Pandas? Warum nicht einfach bei der Liste
bleiben? Der wichtigste Unterschied zwischen Liste und Series ist der **Index**.

Bei einer Liste ist der Index implizit definiert. Damit ist gemeint, dass bei
der Initialisierung einer Liste automatisch ein nummerierter Index 0, 1, 2, 3,
... angelegt wird. Wenn bei einer Liste auf das dritte Element zugegriffen
werden soll, dann verwenden wir den Index 2 (zur Erinnerung: Python zählt ab 0)
und schreiben

```{code-cell}
preis_drittes_auto = preisliste[2]
print(f'Preis des dritten Autos: {preis_drittes_auto} EUR')
```

Die Datenstruktur Series ermöglich es aber, einen *expliziten Index* zu setzen.
Über den optionalen Parameter `index=` speichern wir als Zusatzinformation noch
ab, von welchem Auto der Verkaufspreis erfasst wurde. Wir werden diesen
Datensatz in den folgenden Kapiteln noch weiter vertiefen. An dieser Stelle
halten wir fest, dass die ersten drei Autos von der Marke Audi sind, die
nächsten sind BMWs und die letzten fünf sind von der Marke Citroen.

```{code-cell}
autos = ['Audi Nr. 1', 'Audi Nr. 2', 'Audi Nr. 3', 'BMW Nr. 1', 'BMW Nr. 2', 'Citroen Nr. 1', 'Citroen Nr. 2', 'Citroen Nr. 3', 'Citroen Nr. 4', 'Citroen Nr. 5']
preise = pd.Series(preisliste, index = autos)
print(preise)
```

Jetzt ist auch klar, warum beim ersten Mal, als wir `print(preise)` ausgeführt
haben, die Zahlen 0, 1, 2, usw. ausgegeben wurden. Zu dem Zeitpunkt hatte das
Series-Objekt noch einen impliziten Index wie eine Liste. Den expliziten Index
nutzen wir jetzt, um auf den Verkaufspreis des dritten Autos zuzugreifen. Das
dritte Auto ist `Audi Nr. 3`. Wie bei Listen verwenden wir eckige Klammern:

```{code-cell}
preis_drittes_auto = preise['Audi Nr. 3']
print(f'Preis des dritten Autos: {preis_drittes_auto} EUR')
```

Die Datenstruktur Series hat gegenüber der Liste noch einen weiteren Vorteil. In
der Datenstruktur ist noch eine Zusatzinformation gespeichert, die Eigenschaft
`dtype`. Darin gespeichert ist der Datentyp der Elemente des Series-Objektes.
Auf diese Eigenschaft kann auch direkt mit dem sogenannten Punktoperator
zugegegriffen werden.

```{code-cell}
datentyp_preise = preise.dtype
print(f'Die einzelnen Elemente des Series-Objektes "preise" haben den Datentyp {datentyp_preise}, sind also Integer.')
```

Offensichtlich sind die gespeicherten Werte Integer.

**Mini-Übung**
Erzeugen Sie ein Series-Objekt mit den Wochentagen als Index und der Anzahl der
Vorlesungsstunden (SWS) an diesem Wochentag.

```{code-cell}
# Hier Ihr Code:
```

### Arbeiten mit Series-Objekten

Falls der Datentyp der einzelnen Elemente eines Series-Objektes ein numerischer
Typ ist (Integer oder Float), können wir mit den Einträgen auch rechnen. So
lassen sich beispielweise die Preise nicht in Euro, sondern als Preis pro
Tausend Euro angeben, wenn wir alle Preise durch 1000 teilen.

```{code-cell}
preise_pro_1000euro = preise / 1000
print(preise_pro_1000euro)
```

Oder Sie könnten auf die Idee kommen, das billigste Auto auf den Preis 0 zu
setzen und sich ausgeben lassen, um wie viel Euro die anderen Autos teuer sind.
Oder anders ausgedrückt, wir subtrahieren von jedem Preis den Wert 1999 EUR:

```{code-cell}
preise_differenz = preise - 1999
print(preise_differenz)
```

Bei zehn Autos war es relativ einfach, das billigste Auto zu ermitteln, indem
wir einfach die Preisliste durchgeschaut haben. Hilfreicher ist es, vorher die
Preise aufsteigend oder absteigend zu sortieren. Dazu nutzen wir die Methode
`.sort_values()`. Der Name lässt vermuten, dass die Methode die Elemente nach
ihrem Wert sortiert.

```{code-cell}
preise_aufsteigend = preise.sort_values()
print(preise_aufsteigend)
```

Jetzt zeigt sich auch der Vorteil des expliziten Index, denn auf die
ursprüngliche Reihenfolge kommt es nicht an. Der explizite Index ermöglicht uns,
jedes Auto auch in der nach Preisen aufsteigend sortierten Liste eindeutig
wiederzufinden. Zum Abschluss sortieren wir noch absteigend. Mit dem optionalen
Argument `ascending` wird gesteuert, ob aufsteigend sortiert werden soll oder
nicht. Fehlt das Argument, so nimmt der Python-Interpreter an, dass `ascending =
True` gewünscht wird, also dass `aufsteigend = wahr` sein soll. Wollen wir
absteigend sortieren, müssen wir `aufsteigend = falsch` setzen, also `ascending
 = False`.

```{code-cell}
preise_absteigend = preise.sort_values(ascending = False)
print(preise_absteigend)
```

Vielleicht ist Ihnen aufgefallen, dass wir das sortierte Series-Objekt gleich in
einer neuen Vaiable abgespeichert haben. Das ist notwendig, wenn die neue
Sortierung erhalten bleiben soll. Standardmäßig wirkt der Sortierungsbefehl
nämlich nur einmalig und ändert die eigentliche Reihenfolge im Original nicht.
Auch das könnte man durch weitere Parameter ändern (`inplace = True`), wie Sie
in der [Pandas-Dokumentation →
sort_values()](https://pandas.pydata.org/docs/reference/api/pandas.Series.sort_values.html)
nachlesen können.

**Mini-Übung**
Alice, Bob, Charlie und Dora sind 22, 20, 24 und 22 Jahre alt. Speichern Sie
diese Informationen in einem Series-Objekt und sortieren Sie von alt nach jung.

```{code-cell}
# Hier Ihr Code
```

## Zusammenfassung und Ausblick Kapitel 3.1

In diesem Kapitel haben wir Pandas und die sehr wichtige Datenstruktur Series
kennengelernt. Im nächsten Kapitel geht es darum, die wichtigsten statistischen
Kennzahlen der Daten zu ermitteln, die in dem Series-Objekt gespeichert sind.

## 3.2 Statistik mit Pandas

Pandas dient nicht nur dazu, Daten zu sammeln, sondern ermöglicht auch
statistische Analysen. Die deskriptive Statistik hat zum Ziel, Daten durch
einfache Kennzahlen und Diagramme zu beschreiben. In diesem Kapitel geht es
darum, die wichtigsten statistischen Kennzahlen mit Pandas zu ermitteln und zu
interpretieren.

### Lernziele 3.2

* Sie können sich mit **.describe()** eine Übersicht über statistische Kennzahlen
  verschaffen.
* Sie wissen, wie Sie die Anzahl der gültigen Einträge mit **.count()** ermitteln.
* Sie kennen die statistischen Kennzahlen Mittelwert und Standardabweichung und
  wissen, wie diese mit **.mean()** und **.std()** berechnet werden.
* Sie können das Minimum und das Maximum mit **.min()** und **.max()** bestimmen.
* Sie wissen wie ein Quantil interpretiert wird und wie es mit **.quantile()**
  berechnet wird.

### Schnelle Übersicht mit .describe()

Die Methode `.describe()` aus dem Pandas-Modul liefert eine schnelle Übersicht
über viele statistische Kennzahlen. Vor allem, wenn neue Daten geladen werden,
sollte diese Methode direkt am Anfang angewendet werden. Wir bleiben bei unserem
Beispiel mit den zehn Autos und deren Verkaufspreisen.

```{code-cell}
# Import des Pandas-Moduls 
import pandas as pd

# Erzeugung der Daten als Series-Objekt
preisliste = [1999, 35990, 17850, 46830, 27443, 14240, 19950, 15950, 21990, 12450]
autos = ['Audi Nr. 1', 'Audi Nr. 2', 'Audi Nr. 3', 'BMW Nr. 1', 'BMW Nr. 2', 'Citroen Nr. 1', 'Citroen Nr. 2', 'Citroen Nr. 3', 'Citroen Nr. 4', 'Citroen Nr. 5']
preise = pd.Series(preisliste, index = autos)
```

Die Anwendung der `.describe()`-Methode liefert folgende Ausgabe:

```{code-cell}
preise.describe()
```

Offensichtlich liefert die Methode `.describe()` acht statistische Kennzahlen,
deren Bedeutung in der
[Pandas-Dokumentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.describe.html)
erläutert wird. Wir gehen im Folgenden jede Kennzahl einzeln durch.

Aber was machen wir, wenn wir die statistischen Kennzahlen erst später verwenden
wollen, können wir sie zwischenspeichern? Probieren wir es aus.

```{code-cell}
statistische_kennzahlen = preise.describe()
```

Es kommt keine Fehlermeldung. Und was ist in der Variable
`statistische_kennzahlen` nun genau gespeichert, welcher Datentyp?

```{code-cell}
type(statistische_kennzahlen)
```

Offensichtlich wird durch das Anwenden der `.describe()`-Methode auf das
Series-Objekt `preise` ein neues Series-Objekt erzeugt, in dem wiederum die
statistischen Kennzahlen von `preise` gespeichert sind. Da wir im letzten
Kapitel schon gelernt haben, dass mit eckigen Klammern und dem Index auf einen
einzelnen Wert zugegriffen werden kann, können wir uns so den minimalen
Verkaufspreis ausgeben lassen:

```{code-cell}
minimaler_preis = statistische_kennzahlen['min']
print(f'Das billigste Auto wird für {minimaler_preis} EUR angeboten.')
```

**Mini-Übung**
Lassen Sie zuerst die Verkaufspreise aufsteigend sortieren und ausgeben. Lesen
Sie anhand der Ausgabe ab: welches Auto ist am teuersten und für wieviel Euro
wird es bei Autoscout24 angeboten? Lassen Sie dann das Maximum über die
statistischen Kennzahlen, d.h. mit .describe() ermitteln. Vergleichen Sie beide
Werte.

```{code-cell}
# Hier Ihr Code
```

Neben der Möglichkeit, die statistischen Kennzahlen über .describe() berechnen
zu lassen und dann mit dem expliziten Index darauf zuzugreifen, gibt es auch
Methoden, um die statistischen Kennzahlen direkt zu ermitteln.

### Anzahl count()

Mit `.count()` wird die Anzahl der Einträge bestimmt, die *nicht* 'NA' sind. Der
Begriff 'NA' ist ein Fachbegriff des maschinellen Lernens. Gemeint sind fehlende
Einträge, wobei die fehlenden Einträge verschiedene Ursachen haben können:

* NA = not available (der Messsensor hat versagt)
* NA = not applicable (es ist sinnlos bei einem Mann nachzufragen, ob er
  schwanger ist)
* NA = no answer (eine Person hat bei dem Umfrage nichts angegeben)

```{code-cell}
anzahl_gueltige_preise = preise.count()
print(f'Im Series-Objekt sind {anzahl_gueltige_preise} nicht NA-Werte, also gültige Datensätze gespeichert.')
```

### Mittelwert mean()

Der Mittelwert ist die Summe aller Elemente geteilt durch ihre Anzahl. Wie
praktisch, dass wir mit .count() schon die Anzahl der gültigen Werte geliefert
bekommen. Rechnen wir zuerst einmal "händisch" nach, was der durchschnittliche
Verkaufspreis der 10 Autos ist.

```{code-cell}
preisliste = [1999, 35990, 17850, 46830, 27443, 14240, 19950, 15950, 21990, 12450]
summe = 1999 + 35990 + 17850 + 46830 + 27443 + 14240 + 19950 + 15950 + 21990 + 12450
print(f'Die Summe ist {summe} EUR.')
mittelwert = summe / 10
print(f'Der durchschnittliche Verkaufspreis ist {mittelwert} EUR.')
```

Mittelwert heißt auf Englisch mean. Daher ist es nicht verwunderlich, dass die
Methode `.mean()` den Mittelwert der Einträge in jeder Spalte berechnet.

```{code-cell}
mittelwert = preise.mean()
print(f'Der mittlere Verkaufspreis beträgt {mittelwert} Euro.')
```

Falls Sie prinzipiell nochmal die Berechnung des Mittelwertes wiederholen
wollen, können Sie folgendes Video ansehen: <https://youtu.be/IKfsGPwACnU>.

### Standardabweichung std()

Der Mittelwert ist eine sehr wichtige statistische Kennzahl. Allerdings verrät
der Mittelwert nicht, wie sich die einzelnen Autopreise relativ zum Mittelwert
verhalten. Bei den 10 Autos sehen wir mit einem Blick, dass einzelne Autos sehr
stark vom Mittelwert abweichen. Audi Nr. 1 kostet nur 1999 EUR und damit nur
circa 10 % vom durchschnittlichen Verkaufspreis. Dafür ist BMW Nr. 1 mehr als
doppelt so teuer. Es ist daher wichtig, sich zusätzlich zum Mittelwert
anzusehen, wie die anderen Datenpunkte vom Mittelwert abweichen. In der
Statistik wird das als **Streuung** bezeichnet. Eine statistische Kennzahl, die
die Streuung von Daten um den Mittelwert angibt, ist die **Standardabweichung**.

Zur Berechnung der Standardabweichung werden zuerst die Abweichungen jedes
Datenpunktes zum Mittelwert berechnet.

```{code-cell}
mittelwert = preise.mean()
differenzen = preise - mittelwert
print(differenzen)
```

Die negativen Vorzeichen stören, wir wollen ja die Abweichung. Daher quadrieren
wir die Differenzen.

```{code-cell}
quadrate = differenzen * differenzen
print(quadrate)
```

Die durchschnittliche Abweichung beschreibt nun, wie weit "weg" die anderen
Verkaufspreise vom Mittelwert sind. Daher bilden wir nun von den Abweichungen
wiederum den Mittelwert. Da Quadrate ein Series-Objekt ist, machen wir das
diesmal nicht händisch, sondern nutzen die Methode `.mean()`.

```{code-cell}
durchschnittliche_abweichungen = quadrate.mean()
print(f'Die durchschnittliche Abweichung ist {durchschnittliche_abweichungen}.')
```

Wenn wir die durchschnittliche Abweichung wiederum als Verkaufspreis gemessen in
Euro interpretieren wollen, gibt es ein Problem. Offensichtlich ist diese Zahl
soviel größer als das teuerste Auto. Das ist nicht verwunderlich, denn wir haben
ja die quadrierten Differenzen genommen. Die Einheit der durchschnittlichen
Abweichung ist also EUR². Das ist aber unpraktisch. Also ziehen wir wieder die
Wurzel, damit wir ein Maß für die durchschnittliche Abweichung haben, das auch
direkt Verkaufspreise widerspiegelt. Das nennen wir dann Standardabweichung.

```{code-cell}
standardabweichung = quadrate.mean()**0.5
print(f'Die Standardabweichung ist {standardabweichung:.2f} EUR.')
```

Benutzen wir Pandas, so liefert die Methode `.std()` die Standardabweichung. Das
'st' in `.std()` für Standard steht, ist nachvollziehbar. Der dritte Buchstabe
'd' kommt von 'deviation', also Abweichung. Somit ist wiederum die Methode nach
dem englischen Fachbegriff 'standard deviation' benannt. Probieren wir die
Methode für die Autopreise aus.

```{code-cell}
standardabweichung = preise.std()
print(f'Die Standardabweichung bei den Verkaufspreisen beträgt {standardabweichung} Euro.')
```

Der Wert, den Pandas berechnet, unterscheidet sich von dem Wert, den wir
"händisch" berechnet haben. Der Unterschied kommt daher, dass es zwei Formeln
zur Berechnung der Standardabweichung gibt. Einmal wird der Durchschnitt über
die Quadrate gebildet, indem die Summe durch die Anzahl aller Elemente geteilt
wird, so wie wir es getan haben. Wir haben durch 10 geteilt. Bei der anderen
Formel wird die Summe der Quadrate durch 9 geteilt.

Was war eigentlich nochmal die Standardabweichung? Falls Sie dazu eine kurze
Wiederholung der Theorie benötigen, empfehle ich Ihnen dieses Video:
<https://youtu.be/QNNt7BvmUJM>.

### Minimum und Maximum mit min() und max()

Die Namen der Methoden `.min()` und `max()` sind fast schon wieder
selbsterklärend. Die Methode `.min()` liefert den kleinsten Wert zurück, der
gefunden wird. Umgekehrt liefert `.max()` den größten Eintrag. Wie häufig die
minimalen und maximalen Werte vorkommen, ist dabei egal. Es kann durchaus sein,
dass das Minimum oder das Maximum mehrfach vorkommt.

Schauen wir uns an, was der niedrigste Verkaufspreis ist. Und dann schauen wir
nach, welches Auto am teuersten ist.

```{code-cell}
preis_min = preise.min()
print(f'Das billigste oder die billigsten Autos werden zum Preis von {preis_min} EUR angeboten.')

preis_max = preise.max()
print(f'Das teuerste oder die teuersten Autos werden für {preis_max} EUR angeboten.')
```

### Quantil mit quantile()

Das Quantil $p \%$ ist der Wert, bei dem $p %$ der Einträge kleiner oder gleich
als diese Zahl sind und $100 \% - p \%$ sind größer. Meist werden nicht
Prozentzahlen verwendet, sondern p ist zwischen 0 und 1, wobei die 1 für 100 %
steht.

Angenommen, wir würden gerne das 0.5-Quantil (auch Median genannt) der Preise
wissen. Mit der Methode `.quantile()` können wir diesen Wert leicht aus den
Daten holen.

```{code-cell}
quantil50 =preise.quantile(0.5)
print(f'Der Median, d.h. das 50 % Quantil, liegt bei {quantil50} EUR.')
```

Das 50 % -Quantil liegt bei 18900 EUR. 50 % aller Autos werden zu einem Preis
angeboten, der kleiner oder gleich 18900 EUR ist. Und 50 % aller Autos werden
teuer angeboten. Wir schauen uns jetzt das 75 % Quantil an.

```{code-cell}
quantil75 = preise.quantile(0.75)
print(f'75 % aller Autos haben einen Preis kleiner gleich {quantil75} EUR.')
```

75 % aller Autos werden günstiger als 26079.75 EUR angeboten. Auch wenn Sie sich
natürlich für jede beliebigen Prozentsatz zwischen 0 % und 100 % das Quantil
ansehen können, interessieren wir uns noch für das 25 % Quantil.

```{code-cell}
quantil25 = preise.quantile(0.25)
print(f'25 % aller Autos haben einen Preis kleiner gleich {quantil25} EUR.')
```

### Zusammenfassung und Ausblick Kapitel 3.2

In diesem Abschnitt haben wir uns mit einfachen statistischen Kennzahlen
beschäftigt, die Pandas mit der Methode `.describe()` zusammenfasst, die aber
auch einzeln über

* `.count()`
* `.mean()`
* `.std()`
* `.min()` und `.max()`
* `.quantile()`

berechnet und ausgegeben werden können. Im nächsten Kapitel geht es darum, durch
Diagramme mehr über die Daten zu erfahren.

## 3.3 Boxplots mit Plotly

Die wichtigsten statistischen Kennzahlen lassen sich mit einem Diagramm
visualisieren, das Boxplot genannt wird. Selten wird auch der deutsche Begriff
Kastendiagramm dafür gebraucht. In diesem Kapitel visualisieren wir nur einen
Datensatz. Die große Stärke der Boxplots ist normalerweise, die statistischen
Kennzahlen von verschiedenen Datensätzen nebeneinander zu visualisieren, um so
leicht einen Vergleich der Datensätze zu ermöglichen.

## Lernziele 3.3

* Sie können **Plotly Express** mit der typischen Abkürzung **px** importieren.
* Sie können mit **px.box()** einen Boxplot eines Pandas-Series-Objektes
  visualisieren.
* Sie können die Beschriftung eines Boxplots verändern. Dazu gehört die die
  Beschriftung der Achsen und der Titel.
* Sie können die Datenpunkte neben einem Boxplot anzeigen lassen.
* Sie wissen, was ein **Ausreißer** ist und können Ausreißer im Boxplot anzeigen
  lassen.

### Plotly

Es gibt zahlreiche Python-Module, die zur Visualisierung von Daten geeignet
sind. In dieser Vorlesung verwenden wir **Plotly**. Plotly unterstützt sehr
viele verschiedene Diagrammtypen, wie auch das bekannteste Modul zur Erstellung
von Diagrammen, die sehr bekannte Python-Bibliothek **Matplotlib**. Im Gegensatz
zu Matplotlib ist Plotly jedoch interaktiv. Zusätzlich bietet Plotly das Module
**Plotly Express** an, das eine einfach zu bedienende Schnittstelle zur
Erstellung von Diagrammen zur Verfügung stellt.

Üblicherweise wird Plotly Express als `px` abgekürzt. Wir importieren das Modul
und schauen uns mit der `dir()`-Funktion an, welche Funktionalitäten Plotly
Express bietet.

```{code-cell}
import plotly.express as px

dir(px)
```

### Boxplots mit Plotly Express

Wir greifen erneut unser Autoscout24-Beispiel mit den 10 Autos auf.

```{code-cell}
import pandas as pd

preisliste = [1999, 35990, 17850, 46830, 27443, 14240, 19950, 15950, 21990, 12450]
preise = pd.Series(preisliste, index = ['Audi Nr. 1', 'Audi Nr. 2', 'Audi Nr. 3', 'BMW Nr. 1', 'BMW Nr. 2', 
    'Citroen Nr. 1', 'Citroen Nr. 2', 'Citroen Nr. 3', 'Citroen Nr. 4', 'Citroen Nr. 5'])

print(preise)
```

Um einen Boxplot zu erstellen, nutzen wir die Funktion `box()` von Plotly
Express. Wir speichern das Diagramm, das durch diese Funktion erstellt wird, in
der Variablen `diagramm`. Um es dann auch nach seiner Erzeugung tatsächlich
anzeigen zu lassen, verwenden wir die Methode `.show()`. Zusammen sieht der
Python-Code zur Erzeugung eines Boxplots folgendermaßen aus:

```{code-cell}
diagramm = px.box(preise)
diagramm.show()
```

Bewegen wir die Maus über dem Diagramm, so sehen wir die interaktiven
Möglichkeiten. Damit die Zahlen besser ablesbar sind, werden sie eingeblendet,
sobald wir mit dem Mauszeiger über der Box sind. Auch erscheinen rechts oben
weitere Einstellmöglichkeiten.

Die untere Antenne zeigt das Minimum an, die obere Antenne das Maximum der
Daten. Der Kasten, also die Box, wird durch das untere unteren Q1 und das obere
Quartil Q3 begrenzt. Oder anders formuliert liegen 50 % aller auftretenden
Elemente in der Box. Der Median wird durch die Linie in der Box dargestellt.

Das folgende Video erklärt, wie der Boxplot zu interpretieren ist:
<https://youtu.be/1I_ma7nvKQw>.

### Beschriftung des Boxplots verändern

Die Achsenbeschriftungen wurden automatisch gesetzt. Die x-Achse ist mit
'variable' und die y-Achse mit 'value' beschriftet. Darüber hinaus ist der Titel
der Box '0'. Das wird auch angezeigt, wenn die Maus sich über die Box bewegt.

Die 0 wird angezeigt, weil das Pandas-Series-Objekt 'preise' für den Boxplot als
Tabelle interpretiert wird und die erste Spalte den Index 0 hat. Wir können der
Spalte aber auch einen eigenen Namen geben. Am einfachsten klappt das direkt bei
der Erzeugung, indem der Parameter `name=` gesetzt wird.

```{code-cell}
preisliste = [1999, 35990, 17850, 46830, 27443, 14240, 19950, 15950, 21990, 12450]
preise_mit_name = pd.Series(preisliste, index = ['Audi Nr. 1', 'Audi Nr. 2', 'Audi Nr. 3', 'BMW Nr. 1', 'BMW Nr. 2', 
    'Citroen Nr. 1', 'Citroen Nr. 2', 'Citroen Nr. 3', 'Citroen Nr. 4', 'Citroen Nr. 5'],
    name='XXS-Liste von Autoscout24')

print(preise_mit_name)
```

Der neue Name 'XXS-Liste von Autoscout24' wird zusätzlich zur Information 'dtype' angezeigt.
Damit sieht der Boxplot folgendermaßen aus:

```{code-cell}
diagramm = px.box(preise_mit_name)
diagramm.show()
```

Sollen nun auch noch die Achsenbeschriftungen geändert werden, müssen wir die
automatisch gesetzten Beschriftungen durch neue Namen ersetzt werden.
Eingeleitet wird die Ersetzung durch das Schlüsselwort `labels=`. Danach steht
in geschweiften Klammern `{` und `}` der alten Name, dann folgt ein Doppelpunkt
und dann der neue Name.

```{code-cell}
diagramm = px.box(preise_mit_name, labels={'variable': 'Name des Datensatzes'})
diagramm.show()
```

Sollen gleich mehrere Beschriftungen ersetzt werden, werden alle Paare mit einem
Komma getrennt aufgelistet.

```{code-cell}
diagramm = px.box(preise_mit_name, labels={'variable': 'Name des Datensatzes', 'value': 'Verkaufspreis [EUR]'})
diagramm.show()
```

Fehlt noch eine Überschrift, ein Titel. Wie das englische Wort 'title' heißt
auch das entsprechende Schlüsselwort zum Erzeugen eines Titels, nämlich
`title=`.

```{code-cell}
diagramm = px.box(preise_mit_name, 
              labels={'variable': 'Name des Datensatzes', 'value': 'Verkaufspreis [EUR]'},
              title='Statistische Kennzahlen als Boxplot')
diagramm.show()
```

### Datenpunkte im Boxplot anzeigen

Oft ist es wünschenswert die Rohdaten zusammen mit dem Boxplot zu visualisieren.
Das ist mit dem `points=`-Parameter recht einfach, jedoch haben wir zwei mögliche
Optionen. Wir können mit `'all'` alle Punkte anzeigen lassen oder nur die
Ausreißer (`'outliers'`).

Lassen wir zuerst alle Punkte anzeigen und setzen also `points='all'`.

```{code-cell}
diagramm = px.box(preise_mit_name, 
              labels={'variable': 'Name des Datensatzes', 'value': 'Verkaufspreis [EUR]'},
              points='all')
diagramm.show()
```

Die Punkte werden links vom Boxplot platziert. Als nächstes lassen wir uns die
Ausreißer anzeigen.

```{code-cell}
diagramm = px.box(preise_mit_name, 
              labels={'variable': 'Name des Datensatzes', 'value': 'Verkaufspreis [EUR]'},
              points='outliers')
diagramm.show()
```

Es sind keine Punkte zu sehen, was ist falsch? Nun, um das zu klären, müssen wir
erst einmal definieren, was ein Ausreißer ist.

### Ausreißer berechnen und visualisieren

Die Box im Boxplot enthält 50 % aller Datenpunkte, denn sie ist durch das untere
Quartil Q1 und das obere Quartil Q3 begrenzt. Die Differenz zwischen Q1 und Q3
wird **Interquartilsabstand** (manchmal auch kurz Quartilsabstand) genannt und
mit **IQR** (englisch für Interquartile Range) abgekürzt. In der Statistik
werden Punkte, die kleiner als Q1 - 1.5 IQR oder Punkte, die größer als Q3 + 1.5
IQR sind, als Ausreißer angesehen. Im Beispiel des XXS-Datensatzes der
Autopreise kommen keine Ausreißer vor, weil Minimum und Maximum noch innerhalb
dieses Bereichs liegen. Wir fügen daher noch ein neues, teureres Auto ein. Jetzt
sehen wir einen Ausreißer.

```{code-cell}
preise_mit_name['BMW Nr. 3'] = 62999
diagramm = px.box(preise_mit_name, 
              labels={'variable': 'Name des Datensatzes', 'value': 'Verkaufspreis [EUR]'},
              points='outliers')
diagramm.show()
```

### Zusammenfassung und Ausblick Kapitel 3.3

Der Boxplot ermöglicht eine einfache Visualisierung der wichtigsten
statistischen Kennzahlen eines Datensatzes. Seine Stärke spielt er aus, sobald
mehrere Datensätze miteinander verglichen werden sollen. Daher werden wir im
nächsten Kapitel uns mit Tabellen beschäftigen.

## Übungen

Gegeben sind folgende Daten zu der Verteilung von Studierenden (männlich/weiblich) auf die Hochschularten Universität und Fachhochschulen (Hochschulen für angewandte Wissenschaften), Quelle: [https://www.statistischebibliothek.de/mir/receive/DESerie_mods_00007716]

```python
bundeslaender = ['Baden-Württemberg', 'Bayern', 'Berlin', 'Brandenburg', 
                 'Bremen', 'Hamburg', 'Hessen', 'Mecklenburg-Vorpommern', 
                 'Niedersachsen', 'Nordrhein-Westfalen', 'Rheinland-Pfalz', 'Saarland',
                 'Sachsen', 'Sachsen-Anhalt', 'Schleswig-Holstein', 'Thüringen']
studierende_universitaeten_maennlich = [85183, 118703, 58682, 15845,
                                        9291, 27444, 68753, 10349, 
                                        62192, 235564, 31487, 7806, 
                                        35826, 15847, 16548, 14350]
studierende_universitaeten_weiblich = [82635, 131158, 65587, 18742,
                                       10181, 28438, 75292, 12821,
                                       69866, 246467, 41755, 8391,
                                       37669, 17061, 22760, 17245]
studierende_fachhochschulen_maennlich = [83058, 81163, 34727, 7778,
                                         8299, 26818, 53998, 7120,
                                         33147, 132976, 21759, 7407,
                                         15497, 12023, 14167, 39330]
studierende_fachhochschulen_weiblich = [65332, 63198, 33333, 6323,
                                        8235, 33558, 47600, 6886,
                                        27157, 106755, 18042, 5767,
                                        11087, 11273, 7943, 63669]
```

### Übung 3.1

Laden Sie die Daten zu den Studentinnen an Fachhochschulen. Verschaffen Sie sich einen Überblick über die statistischen Kennzahlen. Lesen Sie dann ab: In welchem Bundesland studieren am wenigsten Studentinnen und im welchem Bundesland am meisten?

### Übung 3.2

Überprüfen Sie für die anderen drei Datensätze, ob auch dort die beiden gleichen Bundesländer herauskommen. Lassen Sie dazu zuerst das Minimum und das Maximum eines jeden Datensatzes direkt mit einer f-print-Anweisung ausgeben und kontrollieren Sie mit der Anzeige des kompletten Datensatzes, welches Bundesland zum Minimum oder Maximum gehört. Fügen Sie Ihre Antwort als Markdown-Zelle ein.

### Übung 3.3

Lassen Sie jeden der vier Datensätze durch einen Boxplot darstellen. Verwenden Sie dabei unterschiedliche Variablen zum Speichern des Boxplots (also beispielsweise fig1, fig2, fig3 und fig4). Gibt es Ausreißer?

### Übung 3.4

Es wäre schön, die Boxplots in einer Grafik nebeneinander zu stellen. Dazu benötigen wir das Untermodul `Graph Objects` von `Plotly`. Danach können die Grafiken wie folgt kombiniert werden.

```python
import plotly.graph_objects as go

fig = go.Figure(data = fig1.data + fig2.data + fig3.data + fig4.data)
fig.update_layout(title='Verteilung Studierende 2022')

fig.show()
```

Kopieren Sie den oben Code in eine Code-Zelle und führen Sie die Code-Zelle aus. Vergleichen Sie die vier Boxplots miteinander. Wo liegt der Median am ehesten in der Mitte des Interquartilabstandes?

### Übung 3.5

Recherchieren Sie im Internet (auch Large Language Models wie ChatGPT oder Bard sind erlaubt), wie die y-Achse auf das Intervall [0, 135000] begrenzt wird, damit der Vergleich leichter fällt und die Ausreißer "abgeschnitten" werden. Modifizieren Sie den gemeinsamen Plot der vier Boxplots entsprechend und beurteilen Sie erneut, wo der Median am ehesten in der Mitte des Interquartilabstandes liegt.
