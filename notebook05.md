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

# 5. Kategoriale Daten erkunden

## 5.1 Was sind kategoriale Daten?

In unserem bisherigen Beispiel zu den Autoverkaufspreisen haben wir bestimmte
Eigenschaften der Autos, wie die Marke oder die Farbe, nicht berücksichtigt.
Unsere statistischen Analysen und Visualisierungen konzentrierten sich
hauptsächlich auf numerische Werte wie den Kilometerstand. Dies liegt daran,
dass es für Daten wie Farben oder Automarken keine Rechenoperationen gibt. In
diesem Kapitel werden wir uns intensiver mit diesen nicht-numerischen Daten
auseinandersetzen.

## Lernziele Kapitel 5.1

* Sie wissen, was **numerische (metrische oder quantitative) Daten** sind.
* Sie wissen, was **kategoriale (qualitative) Daten** sind.
* Sie können die Methode **.unique()** benutzen, um die eindeutigen Werte eines
  Pandas-Series-Objektes aufzulisten.
* Sie kennen den Unterschied zwischen **ungeordneten** und **geordneten**
  kategorialen Daten.
* Sie können mit der Methode **.value_counts()** die Anzahl der eindeutigen
  Werte eines Pandas-Series-Objektes bestimmen lassen.
* Sie wissen, was der **Modalwert** oder **Modus** eines Datensatzes ist und
  können diesen mit der Methode **.mode()** bestimmen lassen.

### Was wir bisher hatten: metrische Daten

In den bisherigen Kapiteln lag unser Fokus auf den Verkaufspreisen von Autos.
Die Methode `.describe()` in Pandas bietet eine schnelle Möglichkeit, einen
Überblick über die statistischen Kennzahlen eines Datensatzes zu erhalten.
Interessanterweise berücksichtigt die `describe()`-Methode nur numerische Werte
(also Zahlen wie Integers und Floats) für die Auswertung. Dennoch bestimmen auch
die nicht-numerischen Eigenschaften eines Autos den Verkaufspreis. Wer würde
schon ein Auto in Pink bevorzugen?

Bevor wir uns den nicht-numerischen Daten widmen, vertiefen wir unser
Verständnis für numerische Daten. Diese werden oft auch als **metrische oder
quantitative Daten** bezeichnet.

> Metrische Daten sind Informationen, die gemessen werden können. Daher können sie
durch Zahlen (ganze Zahlen, rationale oder reelle Zahlen) auf einer Skala
dargestellt werden und werden numerische Daten genannt. Ein anderer Name für
metrische Daten ist der Begriff quantitative Daten.

Betrachten wir erneut den Datensatz der Autoverkaufspreise von Autoscout24.de,
diesmal jedoch mit allen Autos, die im Jahr 2020 zugelassen wurden. Ein kurzer
Überblick über den Datensatz hilft uns, die Art der Daten besser zu verstehen.

```{code-cell}
import pandas as pd
data = pd.read_csv('autoscout24_DE_2020.csv')
data.info()
```

#### Mini-Übung 5.1

Welche Eigenschaften der Autos sind numerisch (metrisch/quantitativ)? Würden Sie
bei anderen Eigenschaften ebenfalls einen metrischen Datentyp erwarten?

Mit numerischen Daten können wir umfangreiche Datenanalysen durchführen. Wir
können vergleichen, ob zwei Messwerte gleich oder ungleich sind. Wir können
beurteilen, ob ein Messwert kleiner oder größer als ein anderer ist oder sogar
das Minimum und das Maximum aller Messwerte bestimmen. Und vor allem können wir
mit metrischen Daten rechnen. Erst dadurch ist es möglich, einen Mittelwert zu
bilden oder Streuungsmaße wie Spannweite, Standardabweichung und
Interquartilsabstand zu berechnen. Solche detaillierten Berechnungen sind nur
bei metrischen (quantitativen) Daten möglich.

### Das Gegenteil von numerischen Daten: kategoriale Daten

Während numerische Daten messbare Informationen darstellen, sind **kategoriale
Daten** durch ihre Zugehörigkeit zu bestimmten Kategorien oder Gruppen
definiert. Ein weiterer Begriff für kategoriale Daten ist **qualitative Daten**.
Ein gutes Beispiel für kategoriale Daten ist die Farbe eines Autos. Oft gibt der
Datentyp einer bestimmten Eigenschaft in einem Datensatz bereits Hinweise
darauf, ob es sich um kategoriale oder numerische Daten handelt.

Der obige Ausführung der Anweisung `data.info()` hat gezeigt, dass einige Daten
als `objects` gespeichert sind, was oft auf kategoriale Daten hinweist. Ein
Blick in die Spalte »Marke« gibt uns weitere Einblicke.

```{code-cell}
data['Marke'].head(10)
```

Die ersten 10 Autos sind offensichtlich Alfa Romeos. Sind vielleicht nur Alfa
Romeos in der Tabelle enthalten? Wir schauen uns die letzten 10 Einträge an.

```{code-cell}
data['Marke'].tail(10)
```

Die letzten Einträge sind Volvos. Vielleicht sind die Autos nach Marken
alphabetisch geordnet? Es wäre schön zu wissen, welche verschiedenen Marken im
Datensatz enthalten sind. Dazu gibt es die Methode `.unique()`. Sie gehört zu
der Datenstruktur Pandas-Series. Wenn wir eine einzelne Spalte eines
Pandas-DataFrames herausgreifen, liegt automatisch ein Pandas-Series-Objekt vor,
so dass wir diese Methode hier benutzen können.

```{code-cell}
data['Marke'].unique()
```

Obwohl der Datensatz insgesamt 18566 Autos umfasst, gibt es nur 41 verschiedene
Marken. Allgemein gesagt, gibt es für die Eigenschaft »Marke« 41 Kategorien.
Eine nicht-metrische Eigenschaft, die nur eine begrenzte Anzahl von Werten
annehmen kann, wird als kategoriale Variable bezeichnet. Ihre konkreten Werte
werden als kategoriale Daten oder qualitative Daten bezeichnet.

> Was sind ... kategoriale/qualitative Daten?
Kategoriale Daten sind Informationen, die nicht gemessen werden können.
Stattdessen werden sie durch die Zugehörigkeit zu einer Kategorie dargestellt.
Sie bekommen ein Etikett. Kategoriale Daten nehmen nur eine begrenzte Anzahl
von verschiedenen Werten an. Oft bezeichnet man kategoriale Daten auch als
qualitative Daten.

Diese Definition ist nicht präzise. Der Begriff "begrenzte Anzahl" von Werten
ist etwas schwammig. Sind damit 10 Kategorien gemeint oder 100 oder 1000? Welche
Eigenschaften des Auto-Datensatzes sind kategorial?

### Mini-Übung 5.2

Suchen Sie sich drei nicht-metrische Eigenschaften aus und bestimmen sie
die Anzahl der einzigartigen Einträge dieser Eigenschaft.

Tipp: Die Methode `.unique()` liefert ein sogenanntes NumPy-Array zurück, das
hier wie eine Liste benutzt werden kann. Die Python-Funktion `len()` kann die
Länge einer Liste, also die Anzahl der Elemente der Liste, bestimmen.

```{code-cell}
# Hier Ihr Code
```

### Kategoriale Daten: ungeordnet oder geordnet?

Innerhalb der kategorialen bzw. qualitativen Daten gibt es wiederum zwei Arten
von Datentypen:

* ungeordnete kategoriale Daten und
* geordnete kategoriale Daten.

Ein typisches Beispiel für ungeordnete kategoriale Daten sind die Farben der
Autos. Es gibt keine natürliche Reihenfolge für Farben. Wir könnten die Farbe
alphabetisch anordnen, doch sobald wir eine andere Sprache benutzen, wäre auch
die Reihenfolge anders. Farben sind ungeordnete Kategorien. Für statistische
Analysen heißt das, dass nur bestimmt werden kann, ob die Farbe eines Autos in
eine bestimmte Kategorie fällt oder nicht. Entweder das Auto ist gelb oder es
ist nicht gelb. Mathematisch gesehen können wir also nur auf Gleichheit oder
Ungleichheit prüfen. Es gibt keine Vergleiche bzgl. der Reihenfolge und auch
keine Minimum oder Maximum. Auch sämtliche Rechenoperationen entfallen und daher
können auch nicht Mittelwerte oder Streuungsmaße bestimmt werden. Stattdessen
wird der **Modus** oder **Modalwert** bestimmt.

> Was ist ... der Modus (Modalwert)?
Der Modus, auch Modalwert genannt, ist der häufigste auftretende Wert in dem
Datensatz. Er gehört zu den Lageparametern in der Statistik. Er existiert sowohl
für ungeordnete und geordnete kategoriale Daten als auch für metrische Daten.

Pandas bietet eine Methode zur Bestimmung des Modus namens `.mode()`. Diese
Methode funktioniert wieder nur für ein Pandas-Series-Objekt, so dass zuerst
eine einzelne Spalte aus der Tabelle herausgeschnitten wird. Auf diese Spalte
wird dann die Methode angewendet.

```{code-cell}
modus_farben = data['Farbe'].mode()
print(f'Die häufigste Farbe ist {modus_farben}.')
```

Und wie häufig kommen die anderen Farben vor? Die Methode `.value_counts()`
zählt die Anzahl an Autos mit einer bestimmten Farbe.

```{code-cell}
data['Farbe'].value_counts()
```

Bleiben noch die geordneten kategorialen Daten. Dazu inspizieren wir die
einzigartigen Werte der Erstzulassung.

```{code-cell}
data['Erstzulassung'].unique()
```

Offensichtlich verbergen sich hinter der Erstzulassung die Monate Januar bis
Dezember des Jahres 2020. Diese haben eine natürliche Ordnung, Januar ist der
erste Monat des Jahres und Dezember ist der letzte Monat des Jahres. Bei
geordneten kategorialen Daten lässt sich ebenfalls der Modus (Modalwert)
bestimmen.

```{code-cell}
data['Erstzulassung'].mode()
```

Die meisten Autos wurden im Februar zugelassen. Wir können auch mit
`.value_counts()` für jeden Monat die Anzahl der Zulassungen auszählen lassen.

```{code-cell}
data['Erstzulassung'].value_counts()
```

Insgesamt scheinen im ersten Quartal besonders viele Autos zugelassen zu werden.
Dadurch, dass aber die Monate geordnet werden können, ist auch möglich den
Median oder allgemein Quantile zu bestimmen. Allerdings kennt Pandas nicht die
natürliche Reihenfolge der Monatsangaben 01/2020, 02/2020, 03/2020, usw. Wir
ersetzen daher die Strings durch Integer, also "01/2020" --> 1, "02/2020" --> 2,
usw. Diesen Vorgang nennt man in der Informatik **Kodierung**. Das Thema, wie
kategoriale (qualitative Daten) am geschicktesten kodiert werden, wird uns noch
intensiver beschäftigen. Jetzt gehen wir händisch vor und benutzen die
`.replace()`-Methode von Pandas.

```{code-cell}
# alte Datenwerte (Monatsangaben als Strings)
kodierung_string = ["01/2020", "02/2020", "03/2020", "04/2020", "05/2020", "06/2020",
"07/2020", "08/2020", "09/2020", "10/2020", "11/2020", "12/2020"]

# neue Datenwerte (nur der Monat als Integer)
kodierung_integer = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Ersetzung der alten Datenwerte durch die neue Kodierung
data['Erstzulassung'].replace(kodierung_string, kodierung_integer, inplace=True)

# Anzeige, ob alles funktioniert hat
data['Erstzulassung'].unique()
```

Jetzt können wir von Pandas den Median berechnen lassen, den die ganzen Zahlen
von 1 bis 12 haben eine natürliche Reihenfolge. Der Median ist

```{code-cell}
data['Erstzulassung'].median()
```

Die Hälfte aller Autos wurde von Januar bis irgendwann im Juni zugelassen. Den
genauen Stichtag im Juni können wir aufgrund fehlender Informationen über den
Tag der Zulassung nicht ermitteln.

### Zusammenfassung und Ausblick Kapitel 5.1

In diesem Kapitel haben wir uns mit dem Unterschied zwischen numerischen
(metrischen bzw. quantitativen) und kategorialen (qualitativen) Daten
beschäftigt. Dabei wird bei kategorialen Daten noch zwischen ungeordneten und
geordneten Kategorien unterschieden. Abhängig von der Art der Daten können
unterschiedliche statistische Kennzahlen erhoben werden. Bei ungeordneten
kategorialen Daten kann nur der Modus (Modalwert) berechnet werden. Bei
geordneten kategorialen Daten können zusätzlich noch Quantile (insbesondere der
Median) berechnet werden. Nur bei numerischen (metrischen bzw. qualitativen)
Daten ist es möglich, den Mittelwert und zusätzlich die Streuungsmaße
(Spannbreite, Standardabweichung und Interquartilsabstand) zu bestimmen. Im
nächsten Kapitel werden wir uns mit der Visualisierung der kategorialen Daten
beschäftigen

## 5.2 Barplots und Histogramme

Barplots (Balken- oder Säulendiagramme) sind die am häufigsten verwendeten
Visualisierungen für kategoriale Daten. In diesem Kapitel lernen wir, wie mit
Plotly ein Barplot erstellt und von einem Histogramm unterschieden wird.

## Lernziele Kapitel 5.2

* Sie wissen, was ein **Barplot** ist.
* Sie können ein **Säulendiagramm** von einem **Balkendiagramm** unterscheiden.
* Sie können mit der Funktion **bar()** des Plotly-Express-Moduls einen Barplot
  generieren lassen.
* Sie wissen, wie aus numerischen Daten ein **Histogramm** erzeugt wird.
* Sie können mit der Funktion **histogram()** des Plotly-Express-Moduls ein
  Histogramm erzeugen lassen.

### Barplots

Im letzten Kapitel haben wir uns mit kategorialen (qualitativen) Daten
auseinandergesetzt. Um solche Daten zu visualisieren und zu vergleichen,
benötigen wir geeignete Diagramme. Ein Boxplot ist hierfür nicht geeignet, da
man mit kategorialen Daten keine Rechenoperationen wie Mittelwertbildung oder
die Berechnung von Streuungsmaßen durchführen kann. Eine Methode, die wir
bereits kennengelernt haben, ist `.value_counts()`. Sie zählt, wie oft jeder
einzigartige Wert in einer Datenreihe vorkommt. Die Anzahl der Werte pro
Kategorie wird mit dem sogenannten **Barplot** visualisiert.

Ein Barplot muss nicht nur die Anzahl der Werte pro Kategorie zeigen. Er kann
jede numerische Information darstellen, die einer Kategorie zugeordnet ist.
Dabei werden prinzipiell zwei Varianten unterschieden. Zum einen können die
Kategorien entlang der x-Achse aneinandergereiht werden. Die Höhe der Rechtecke
repräsentiert dann den Zahlenwert dieser Kategorie. Da die Rechtecke an Säulen
erinnern, wird diese Variante **Säulendiagramm** genannt. Die andere Möglichkeit
ist, die Kategorien untereinander entlang der y-Achse aufzuführen. Dann ist die
Länge der Rechtecke repräsentativ für den Zahlenwert dieser Kategorie. Diese
Variante wird **Balkendiagramm** genannt.

> Was ist ... ein Barplot?
Ein Barplot ist ein Diagramm, das kategoriale Daten visualisiert. Jede Kategorie
wird durch die Höhe oder Länge eines Rechtecks repräsentiert, das den
zugehörigen Wert darstellt.

Probieren wir Barplots am Beispiel der Autoscout24-Verkaufspreise für Autos aus,
die 2020 zugelassen wurden. Zuerst laden wir die Daten und verschaffen uns einen
Überblick.

```{code-cell}
import pandas as pd

data = pd.read_csv('autoscout24_DE_2020.csv')
data.info()
```

Mit der Methode `.value_counts()` lassen wir Python die Anzahl der Autos pro
Marke bestimmen.

```{code-cell}
anzahl_pro_marke = data['Marke'].value_counts()
anzahl_pro_marke.info()
```

Schauen wir uns die ersten zehn Einträge an:

```{code-cell}
anzahl_pro_marke.head(10)
```

Die Methode `.value_counts()` sortiert die Einträge standardmäßig von der
höchsten zur niedrigsten Anzahl.

Mit nur wenigen Zeilen Code können wir mit der Funktion `bar()` aus dem
Plotly-Express-Modul eine Visualisierung erstellen. Zuerst importieren wir das
Modul, dann erzeugen wir das Diagramm mit `bar()` und zuletzt lassen wir das
Diagramm mit `show()` anzeigen.

```{code-cell}
import plotly.express as px

fig = px.bar(anzahl_pro_marke)
fig.show()
```

Obwohl Plotly Express bereits eine ansprechende Visualisierung bietet, könnten
die automatisch generierten Beschriftungen "index", "value" und "variable"
verbessert werden. Außerdem sollte ein Diagrammtitel hinzugefügt werden. Der
Titel kann direkt in der `bar()`-Funktion über das `title=` Argument gesetzt
werden. Für die Achsenbeschriftungen und den Legendentitel verwenden wir die
Funktion `update_layout()`. Die Argumente `xaxis_title=` und `yaxis_title=`
modifizieren die Beschriftung der x- und y-Achse. Mit `legend_title=` wird der
Titel der Legende neu beschriftet.

```{code-cell}
fig = px.bar(anzahl_pro_marke, title='Autoscout24 (Zulassungsjahr 2020)')
fig.update_layout(
    xaxis_title='Marke',
    yaxis_title='Anzahl Autos',
    legend_title='Anzahl Autos pro Marke',
)
fig.show()
```

### Histogramm

Während Barplots in erster Linie kategoriale Daten visualisieren, dienen
Histogramme zur Darstellung numerischer Daten. Ein Barplot zeigt typischerweise
die Anzahl der Werte pro Kategorie. Bei numerischen Daten wäre eine solche
Darstellung oft nicht sinnvoll. Nehmen wir als Beispiel die Kilometerstände von
Autos. Wir lassen zuerst mit der Methode `.unique()` die verschiedenen
Kilometerstände bestimmen. Das Ergebnis ist ein sogenanntes NumPy-Array, das
hier wie eine Liste benutzt werden kann. Mit Hilfe der `len()`-Funktion können
wir die Anzahl der Einträge berechnen.

```{code-cell}
kilometerstaende = data['Kilometerstand (km)'].unique()
anzahl_kilometerstaende = len(kilometerstaende)
print(f'Es gibt {anzahl_kilometerstaende} verschiedene Kilometerstände.')
```

Mit über 10.000 verschiedenen Kilometerständen wäre eine direkte Visualisierung
nicht zielführend. Um dennoch eine sinnvolle Analyse durchzuführen, können wir
die Daten in Kategorien einteilen. Dazu bestimmen wir das Minimum und das
Maximum der Kilometerstände.

```{code-cell}
minimaler_kilometerstand = data['Kilometerstand (km)'].min()
maximaler_kilometerstand = data['Kilometerstand (km)'].max()

print(f'minimaler Kilometerstand: {minimaler_kilometerstand}')
print(f'maximaler Kilometerstand: {maximaler_kilometerstand}')
```

Die Daten reichen von Neuwagen (minimaler Kilometerstand 0 km) bis zu Autos mit
hohem Kilometerstand (maximaler Kilometerstand 435909 km). Wir können diesen
Bereich in gleichmäßige Kategorien unterteilen. Wählen wir beispielsweise 10
Kategorien, so würde die 1. Kategorie alle Autos mit einem Kilometerstand von 0
km bis 50000 km umfassen. Die 2. Kategorie geht dann von 50000 km bis 100000 km
usw. Um jetzt zu ermitteln, wie viele Autos in die jeweilige Kategorie fallen,
könnten wir ein kleines Python-Programm schreiben. Tatsächlich brauchen wir das
nicht, denn diese Funtkionalität ist bereits in der `histogram()`-Funktion
integriert, die auch die Visualisierung übernimmt.

Wir übergeben der Funktion als erstes Argument die Daten und als (optionales)
Argument, wie viele Kategorien wir uns wünschen. Die künstlich erfundenen
Kategorien werden auch als Bins (Tonnen) bezeichnet. Daher lautet das Argument
zum Setzen der Anzahl der Bins `nbins=`, so wie der englische Begriff »number of
bins«.

```{code-cell}
fig = px.histogram(data['Kilometerstand (km)'], nbins=10, 
    title='10 künstlich erzeugte Kategorien bzgl. des Kilometerstandes (km)')
fig.update_layout(
    xaxis_title='Kategorien der Kilometerstände (km)',
    yaxis_title='Anzahl Autos',
    legend_title='Anzahl Autos pro Kategorie',
)
fig.show()
```

Die meisten Autos haben weniger als 200000 km auf dem Kilometerzähler.

Ein charakteristisches Merkmal von Histogrammen ist, dass die Balken ohne Lücke
aneinander liegen, was die kontinuierliche Natur der numerischen Daten
widerspiegelt. Die Anzahl der Kategorien (Bins) beeinflusst die Darstellung
maßgeblich und sollte sorgfältig gewählt werden. Auch können die
Histogramm-Kategorien nicht in eine andere Reihenfolge gebracht werden.

Die Anzahl der Kategorien ist ein sehr wichtiger Faktor bei der Visualisierung.
Werden zu wenige Kategorien gewählt, werden auch nicht die Unterschiede
sichtbar. Werden zu viele Kategorien gewählt, sind ggf. einige Kategorien leer.

#### Mini-Übung 5.3

Wählen Sie verschiedene Werte für die Anzahl der Kategorien aus. Welche Anzahl
an Kategorien ist für diesen Datensatz sinnvoll und warum?

Zusammenfassend wird ein Histogramm folgendermaßen beschrieben.

> Was ist ... ein Histogramm?
Ein Histogramm ist eine grafische Darstellung, bei der numerische Daten in
Kategorien eingeteilt und dann die Anzahl der Werte pro Kategorie durch die Höhe
eines Balkens dargestellt wird

### Zusammenfassung und Ausblick Kapitel 5.2

In diesem Kapitel wurden zwei wichtige Diagrammtypen vorgestellt: der Barplot
und das Histogramm. Obwohl beide mit Rechtecken arbeiten, haben sie
unterschiedliche Anwendungsbereiche und sollten nicht verwechselt werden.
Während der Barplot ideal für kategoriale Daten ist, eignet sich das Histogramm
zur Visualisierung numerischer Daten. Im nächsten Kapitel widmen wir uns dem
Thema Datenfilterung

## 5.3 Daten filtern und gruppieren

Im vorherigen Kapitel haben wir Autos basierend auf ihrem Kilometerstand
gruppiert und visualisiert. Während diese Gruppierung automatisch im Hintergrund
stattfand, werden wir in diesem Kapitel lernen, wie wir direkt auf die
gruppierten Daten zugreifen und zusätzliche Analysen durchführen können.

## Lernziele Kapitel 5.3

* Sie wissen, dass die Wahrheitswerte `True` (wahr)  oder `False` (falsch) in
  dem Datentyp **bool** gespeichert werden.
* Sie kennen die wichtigsten Vergleichsoperatoren (`<`, `<=`, `>`, `>=`, `==`,
  `!=`, `in`, `not in`) in Python.
* Sie können ein Pandas-DataFrame-Objekt nach einem Wert filtern.
* Sie können ein Pandas-DataFrame-Objekt mit den Methoden `groupby()` und
  `get_group()` gruppieren.

### Daten filtern

Im vorherigen Kapitel haben wir die Kilometerstände von Autos untersucht, die im
Jahr 2020 zugelassen und Mitte 2023 auf Autoscout24.de angeboten wurden. Bei der
Kategorisierung der Kilometerstände fiel auf, dass Fahrzeuge mit einer
Laufleistung von über 200000 km selten sind. Trotzdem beeinflusste dies die
Aufteilung in zehn gleichmäßige Gruppen, die von 0 km bis 435909 km reichten,
erheblich. Um eine genauere Analyse zu ermöglichen, wäre es sinnvoll, Fahrzeuge
mit einer Laufleistung von bis zu 200.000 km in den Fokus zu nehmen und die
Ausreißer auszuschließen. Daher widmen wir uns in diesem Kapitel der Filterung
von tabellarischen Datensätzen mithilfe von Pandas.

Zuerst laden wir den Datensatz und überprüfen den Inhalt.

```{code-cell}
import pandas as pd

data = pd.read_csv('autoscout24_DE_2020.csv')
data.info()
```

Um die Autos mit einem Kilometerstand von bis zu 200000 km zu filtern,
vergleichen wir die entsprechende Spalte mit dem Wert 200000, indem wir den aus
der Mathematik bekannten Kleiner-gleich-Operators `<=` benutzen. Das Ergebnis
dieses Vergleichs speichern wir in der Variable `bedingung`.

```{code-cell}
bedingung = data['Kilometerstand (km)'] <= 200000
```

Aber was genau ist in der Variable `bedingung` enthalten? Schauen wir uns den
Datentyp an:

```{code-cell}
type(bedingung)
```

Offensichtlich handelt es sich um ein Pandas-Series-Objekt. Für weitere
Informationen können wir die `.info()`-Methode aufrufen:

```{code-cell}
bedingung.info()
```

In dem Series-Objekt sind 18566 Einträge vom Datentyp `bool` gespeichert. Diesen
Datentyp haben wir bisher nicht kennengelernt. Wir lassen die ersten fünf
Einträge ausgeben:

```{code-cell}
bedingung.head()
```

Sind alle Einträge mit dem Wert `True` gefüllt? Wie viele und vor allem welche
einzigartige Einträge gibt es in diesem Series-Objekt?

```{code-cell}
bedingung.unique()
```

Das Series-Objekt enthält nur `True` und `False`, was den Datentyp `bool`
charakterisiert. In diesem Datentyp können nur zwei verschiedene Werte
gespeichert werden, nämlich wahr (True) und falsch (False). Oft sind
Wahrheitswerte das Ergebnis eines Vergleichs, wie das folgende Code-Beispiel
zeigt:

```{code-cell}
x = 19
print(x  < 100)
```

In der Python-Programmierung wird der Datentyp bool oft verwendet, um
Programmcode zu verzweigen. Damit ist gemeint, dass Teile des Programms nur
durchlaufen und ausgeführt werden, wenn eine bestimmte Bedingung wahr (True)
ist. In dieser Vorlesung benutzen wir bool-Werte hauptsächlich zum Filtern von
Daten.

> Welche Vergleichsoperatoren kennt Python
In Python können die mathematischen Vergleichsoperatoren in ihrer gewohnten
Schreibweise verwendet werden:

* `<` kleiner als
* `<=` kleiner als oder gleich
* `>` größer
* `>=` größer als oder gleich
* `==` gleich (`=` ist der Zuweisungsoperator, nicht mit Gleichheit
  verwechseln!)
* `!=` ungleich

Darüber hinaus kann mit `in` oder `not in` getestet werden, ob
ein Element in einer Liste ist oder eben nicht.

Aber was machen wir jetzt mit diesem Series-Objekt? Wir können es als Index
benutzen für den ursprünglichen Datensatz benutzen. Die Zeilen, in denen `True`
steht, werden übernommen, die anderen verworfen.

```{code-cell}
autos_bis_200000km = data[bedingung]
autos_bis_200000km.info()
```

Von den 18566 Autos wurden 18525 Autos übernommen. Ist denn die Filterung
geglückt? Wir verschaffen uns mit der `.describe()`-Methode einen schnellen
Überblick.

```{code-cell}
autos_bis_200000km.describe()
```

Der maximale Eintrag für die Spalte `Kilometerstand (km)` ist 199000 km. Mit dem
Tilde-Operator `~` können wir das Pandas-Series-Objekt `bedingung` in das
Gegenteil umwandeln. Damit können wir also die Autos, bei denen der Vergleich
`<= 200000` zu `False` ausgewertet wurde, herausfiltern.

```{code-cell}
autos_ab_200000km = data[~bedingung]
autos_ab_200000km.info()
```

41 Autos, die 2020 zugelassen wurden, sollten Mitte 2023 mit einem
Kilometerstand von mehr als 200000 km verkauft werden. Schauen wir uns die
Statistik an.

```{code-cell}
autos_ab_200000km.describe()
```

Und was sind das für Autos?

```{code-cell}
autos_ab_200000km.head(10)
```

### Gruppieren

Eine Filterung nach Kilometerstand ermöglicht es uns, die Autos in zwei
Datensätze zu teilen: Autos mit bis zu 200000 km Laufleistung und jene mit mehr
als 200000 km (hierzu kann der Tilde-Operator (~) verwendet werden).

Wenden wir nun diese Technik an, um die Fahrzeuge basierend auf ihrer Marke zu
trennen. Ein Beispiel: Um alle "Audi"-Fahrzeuge zu extrahieren, verwenden wir
den folgenden Code:

```{code-cell}
bedingung_audi = data['Marke'] == 'audi'
audis = data[bedingung_audi]
audis.info()
```

Diese Bedingung erfüllen 1.190 Autos. Der Gesamtdatensatz enthält jedoch 41
unterschiedliche Automarken. Es wäre ineffizient, für jede Marke eine separate
Filterung durchzuführen. Deshalb bietet Pandas die `.groupby()`-Methode, die es
erlaubt, die Daten automatisch nach den einzigartigen Einträgen einer Spalte zu
gruppieren:

```{code-cell}
autos_nach_marke = data.groupby('Marke')
type(autos_nach_marke)
```

Das Resultat ist eine spezielle Pandas-Datenstruktur namens `DataFrameGroupBy`.
Es sind nicht alle bisher bekannten Methoden auf dieses Objekt anwendbar, aber
beispielsweise die `.describe()`-Methode darf verwendet werden:

```{code-cell}
autos_nach_marke.describe()
```

Für jede Automarke werden nun für jede Spalte mit metrischen (quantitativen)
Informationen die statistischen Kennzahlen ermittelt. Die entstehende Tabelle
ist etwas unübersichtlich. Besser ist daher, sich die statistischen Kennzahlen
einzeln ausgeben zu lassen. Im Folgenden ermitteln wir die Mittelwerte der
metrischen Informationen nach Automarke. Damit tatsächlich auch nur die
metrischen Daten gemittelt werden, müssen wir als Argument noch zusätzlich
`numeric_only=True` setzen.

```{code-cell}
autos_nach_marke.mean(numeric_only=True)
```

Eine sehr wichtige Methode der GroupBy-Datenstruktur ist die
`get_group()`-Methode. Damit können wir ein bestimmtes DataFrame-Objekt aus dem
GroupBy-Objekt extrahieren:

```{code-cell}
audis_alternativ = autos_nach_marke.get_group('audi')
audis_alternativ.info()
```

In der Variablen `audis_alternativ` steckt nun der gleiche Datensatz wie in der
Variablen `audis`, den wir bereits durch das Filtern des ursprünglichen
Datensatzes extrahiert haben.

## Zusammenfassung und Ausblick Kapitel 5.3

In diesem Kapitel haben wir die Technik des Datenfilterns kennengelernt. Um
spezifische Einträge aus einem Datensatz basierend auf einem bestimmten Wert zu
extrahieren, nutzen wir Vergleichsoperationen und verwenden das resultierende
Series-Objekt als Index. Wenn das Ziel darin besteht, Daten anhand der
einzigartigen Werte einer Spalte zu gruppieren, dann ist die Kombination von
`.groupby()` und `.get_group()` oft der effizienteste Weg. Damit haben wir
unsere Einführung in die Datenexploration abgeschlossen, obwohl es noch viele
weitere Möglichkeiten gibt, die Daten zu erkunden. Im nächsten Kapitel beginnen
wir mit den Grundlagen des maschinellen Lernens und beschäftigen uns mit
Entscheidungsbäumen

## Übungen

### Übung 5.1

Schauen Sie sich die Datei 'kaggle_germany-wind-energy.csv' im Texteditor bzw.
im JupyterLab an. Welche Spalte könnte als Zeilenindex dienen? Importieren Sie
passend die Daten.

Verschaffen Sie sich einen Überblick über die Daten. Welche Messwerte stehen in
den Spalten? Weitere Informationen erhalten Sie hier:
<https://www.kaggle.com/datasets/aymanlafaz/wind-energy-germany>

Ermitteln Sie die statistischen Kennzahlen und visualisieren Sie die
statistischen Kennzahlen als Boxplot. Ist es sinnvoll, einen gemeinsamen Boxplot
zu verwenden? Interpretieren Sie die statistischen Kennzahlen.

Visualisieren Sie die drei Eigenschaften als Sctterplots. Welche
Schlussfolgerungen ziehen Sie aus den Plots?

Visualisieren Sie drei Eigenschaften als Scattermatrix. Gibt es Abhängigkeiten?

```{code-cell}
# Hier Ihr Code
```

### Übung 5.2

Importieren Sie den Datensatz 'kaggle_ikea.csv' und verschaffen Sie sich einen
Überblick über die Daten.

In welche verschiedenen Katgorien (category) sind die IKEA-Artikel unterteilt?
Erstellen Sie für jede Kategorie einen Boxplot der Verkaufspreise und ziehen Sie
Schlussfolgerungen daraus.

Lassen Sie für jede Katgeorie den durschschnittlichen Preis dieser Kategorie als
Barplot visualisieren. Lesen Sie danach ab: welche Kategorie hat den geringsten
Durchschnittspreis?

Lassen Sie für diese Kategorie die Anzahl der Artikel bestimmen. Visualisieren
Sie dann den Preis in Abhängigkeit des Namens als Scatterplot. Bei welchem Namen
gibt es die größte Spannweite an Verkaufspreisen? Welches ist der teuerste
Artikel (ID?) in dieser Kategorie und wie wird er beschrieben?

```{code-cell}
# Hier Ihr Code
```

### Übung 5.3

Lesen Sie die csv-Datei 'statistic_id1301764_formel1-fahrerwertung-saison-2022.csv' (Formel 1 Fahrerwertung, Stand 30.10.2022, Quelle: <https://de.statista.com/statistik/daten/studie/1301764/umfrage/formel-1-wm-stand/)> ein.

Führen Sie eine statistische Datenanalyse inklusive Visualisierung durch. Visualisieren Sie zusätzlich die Fahrerwertung.

```{code-cell}
# Hier Ihr Code
```
