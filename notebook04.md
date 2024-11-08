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

# 4. Tabellarische Daten erkunden

## 4.1 Datenstruktur DataFrame

Bisher haben wir uns mit Datenreihen beschäftigt, sozusagen eindimensionalen
Arrays. Das Modul Pandas stellt zur Verwaltung von Datenreihen die Datenstruktur
Series zur Verfügung. In diesem Kapitel lernen wir die Datenstruktur
**DataFrame** kennen, die die Verwaltung von tabellarischen Daten ermöglicht,
also sozusagen zweidimensionalen Arrays.

### Lernziele Kapitel 4.1

* Sie kennen die Datenstruktur **DataFrame**.
* Sie kennen das **csv-Dateiformat**.
* Sie können eine csv-Datei mit **read_csv()** einlesen.
* Sie können die ersten Zeilen eines DataFrames mit **.head()** anzeigen lassen.
* Sie konnen mit **.info()** sich einen Überblick über die importierten Daten
  verschaffen.
* Sie können mit **.describe()** die statistischen Kennzahlen ermitteln.

### Was ist ein DataFrame?

Bei Auswertung von Messungen ist der häufigste Fall der, dass Daten in Form
einer Tabelle vorliegen. Ein DataFrame-Objekt entspricht einer Tabelle, wie man
sie beispielsweise von Excel, LibreOffice oder Numbers kennt. Sowohl Zeile als
auch Spalten sind indiziert. Typischerweise werden die Daten in der Tabelle
zeilenweise angeordnet. Damit ist gemeint, dass jede Zeile einen Datensatz
darstellt und die Spalten die Eigenschaften speichern.

Ein DataFrame kann direkt über mehrere Pandas-Series-Objekte oder verschachtelte
Listen erzeugt werden. Da es in der Praxis nur selten vorkommt und nur für sehr
kleine Datenmengen praktikabel ist, Daten händisch zu erfassen, fokussieren wir
gleich auf die Erzeugung von DataFrame-Objekten aus einer Datei.

### Import von Tabellen mit .read_csv()

Tabellen liegen werden oft in dem Dateiformat abgespeichert, das die jeweilige
Tabellenkalkulationssoftware Excel, Numbers oder LibreOffice Calc als Standard
eingestellt hat. Wir betrachten in dieser Vorlesung Tabellen, die in einem
offenen Standardformat vorliegen und damit unabhängig von der verwendeten
Software und dem verwendeten Betriebssystem sind.

Das **Dateiformat CSV** speichert Daten zeilenweise ab. Dabei steht CSV für
"comma separated value". Die Trennung der Spalten erfolgt durch ein
Trennzeichen, normalerweise durch das Komma. Im deutschsprachigen Raum wird
gelegentlich ein Semikolon verwendet, weil im deutschprachigen Raum das Komma
als Dezimaltrennzeichen verwendet wird.

Um Tabellen im csv-Format einzulesen, bietet Pandas eine eigene Funktion namens
`read_csv` an (siehe [Dokumentation →
read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)).
Wird diese Funktion verwendet, um die Daten zu importieren, so wird automatisch
ein DataFrame-Objekt erzeugt. Beim Aufruf der Funktion wird mindestens der
Dateiname übergeben. Zusäztliche Optionen können über optionale Argumente
eingestellt werden. Beispielweise könnte auch das Semikolon als Trennzeichen
eingestellt werden.

Am besten sehen wir uns die Funktionsweise von `read_csv` an einem Beispiel an.
Sollten Sie mit einem lokalen Jupyter Notebook arbeiten, laden Sie bitte die
Datei
[autoscout24_xxs.csv](https://gramschs.github.io/book_ml4ing/data/autoscout24_xxs.csv)
herunter und speichern Sie sie in denselben Ordner, in dem auch dieses Jupyter
Notebook liegt. Alternativ können Sie die csv-Datei auch über die URL
importieren, wie es in der folgenden Code-Zelle gemacht wird. Die csv-Datei
enthält die Angaben zu 10 Autos, die auf
[Autoscout24](https://www.autoscout24.de) zum Verkauf angeboten wurden.

Führen Sie dann anschließend die folgende Code-Zelle aus.

```{code-cell}
import pandas as pd

url = 'https://gramschs.github.io/book_ml4ing/data/autoscout24_xxs.csv'
tabelle = pd.read_csv(url)
```

Es erscheint keine Fehlermeldung, aber den Inhalt der geladenen Datei sehen wir
trotzdem nicht. Dazu verwenden wir die Methode `.head()`.

### Anzeige der ersten Zeilen mit .head()

Probieren wir einfachmal aus, was die Anwendung der Methode `.head()` bewirkt.

```{code-cell}
tabelle.head()
```

Die Methode `.head()` zeigt uns die ersten fünf Zeilen der Tabelle an. Wenn wir
beispielsweise die ersten 10 Zeilen anzeigen lassen wollen, so verwenden wir die
Methode .head() mit dem Argument 10, also `.head(10)`:

```{code-cell}
tabelle.head(10)
```

Offensichtlich wurde beim Import der Daten wieder ein impliziter Index 0, 1, 2,
usw. gesetzt. Das ist nicht weiter verwunderlich, denn Pandas kann nicht wissen,
welche Spalte wir als Index vorgesehen haben. Und manchmal ist ein automatisch
erzeugter impliziter Index auch nicht schlecht. In diesem Fall würden wir aber
gerne als Zeilenindex die Auto-IDs verwenden. Daher modifizieren wir den Befehl
read_csv mit dem optionalen Argument `index_col=`. Die Namen stehen in der 1.
Spalte, was in Python-Zählweise einer 0 entspricht.

```{code-cell}
tabelle = pd.read_csv('autoscout24_xxs.csv', index_col=0)
tabelle.head(10)
```

### Übersicht verschaffen mit .info()

Das obige Beispiel zeigt uns zwar nun die ersten 10 Zeilen des importierten
Datensatzes, aber wie viele Daten insgesamt enthalten sind, können wir mit der
`.head()`-Methode nicht erfassen. Dafür stellt Pandas die Methode `.info()` zur
Verfügung. Probieren wir es einfach aus.

```{code-cell}
tabelle.info()
```

Mit `.info()` erhalten wir eine Übersicht, wie viele Spalten es gibt und auch
die Spaltenüberschriften werden aufgelistet.

Weiterhin entnehmen wir der Ausgabe von `.info()`, dass in jeder Spalte 10
Einträge sind, die 'non-null' sind. Damit ist gemeint, dass diese Zellen beim
Import nicht leer waren. Zudem wird bei jeder Spalte noch der Datentyp
angegeben. Für die Marke oder das Modell, die als Strings gespeichert sind, wird
der allgemeine Datentyp 'object' angegeben. Beim Jahr oder dem Preis wurden
korrektweise Integer erkannt. Der Verbrauch (Liter pro 100 Kilometer) wird als
Float gespeichert.

### Statistische Kennzahlen mit .describe()

So wie die Methode `.info()` uns einen schnellen Überblick über die Daten eines
DataFrame-Objektes gibt, so liefert die Methode `.describe()` eine schnelle
Übersicht über statistische Kennzahlen.

```{code-cell}
tabelle.describe()
```

Da es sich eingebürgert hat, Daten zeilenweise abzuspeichern und die Eigenschaft
pro einzelnem Datensatz in den Spalten zu speichern, wertet `.describe()` jede
Spalte für sich aus. Für jede Eigenschaft werden dann die statistischen
Kennzahlen

* count
* mean
* std
* min
* max
* Quantile 25 %, 50 % und 75 %
* max

ausgegeben.

Die Bedeutung der Kennzahlen wird in der
[Dokumentation → describe()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)
erläutert. Sie entsprechen den statistischen Kennzahlen, die die Methode
.describe() für Series-Objekte liefert. Pandas hat hier auch auf den Datentyp
reagiert. Nur für numerische Werte (Integer oder Float) wurden die statistischen
Kennzahlen ermittelt.

### Zusammenfassung und Ausblick Kapitel 4.1

Mit Hilfe der Datenstruktur DataFrame können tabellarische Daten verwaltet
werden. In den nächsten Kapiteln werden wir uns damit beschäftigen, auf einzelne
Spalten oder Zeilen zuzugreifen und die Datenpunkte als sogenannten Scatterplot
zu visualisieren.

## 4.2 Arbeiten mit Tabellendaten

In Tabellenkalkulationssoftware ist es möglich, einzelne Zeilen oder Spalten zu
bearbeiten. Pandas mit seiner Dantestruktur DataFrame bietet diese Möglichkeit
ebenfalls. Wie auf einzelne Spalten und Zeilen zugegriffen wird und wie die
Daten bearbeitet werden können, zeigt dieses Kapitel.

## Lernziele Kapitel 4.2

* Sie können mit eckigen Klammern **[]** und dem Spaltenindex auf eine ganze
  Spalte zugreifen.
* Sie können mit **.loc[]** und dem Zeilenindex auf eine ganze Zeile zugreifen.
* Sie können mit **.loc[zeileindex, spaltenindex]** auf eine einzelne Zelle der
  Tabelle zugreifen.
* Sie können mehrere unzusammenhängende Zeilen/Spalten mittels Liste auswählen.
* Sie können zusammenhängende Bereich mittels **Slicing** auswählen.
* Sie können eine Tabelle um eine Zeile oder Spalte erweitern.

### Zugriff auf Spalten

Bei einer Liste oder der Pandas-Datenstruktur Series haben wir auf ein einzelnes
Element zugegriffen, indem wir eckige Klammern benutzt haben. Bei Tabellen und
damit auch DataFrames ist es üblich, dass die Eigenschaften in den Spalten
stehen und in den Zeilen die einzelnen Datensätze. Mit den eckigen Klammern und
dem Indexnamen greifen wir diesmal also nicht nur ein Element heraus, sondern
gleich eine ganze Spalte.

Falls Sie aus dem vorherigen Kapitel den Datensatz `'autoscout24_xxs.csv'` noch
geladen haben, können Sie sich mit `.info()` die Spaltenüberschriften, also den
Index, direkt anzeigen lassen. Ansonsten importieren Sie zuerst Pandas mit
seiner üblichen Abkürzung pd und laden den Datensatz.

```{code-cell}
import pandas as pd
tabelle = pd.read_csv('autoscout24_xxs.csv', index_col=0)
tabelle.info()
```

Die Farbe der 10 Autos können wir folgendermaßen aus der Tabelle auswählen:

```{code-cell}
farbe = tabelle['Farbe']
```

Was steckt jetzt in der Variable `farbe`? Ermitteln wir zunächst, welchen
Datentyp das Objekt hat, das in `farbe` gespeichert ist.

```{code-cell}
type(farbe)
```

Es ist ein Series-Objekt mit dem Namen Farbe, also dem Spaltenindex. Das neu
erzeugte Series-Objekt kann also beispielsweise mit `.head()` angezeigt werden.

```{code-cell}
farbe.head()
```

Ein DataFrame besteht aus Series-Objekten.

### Zugriff auf Zeilen

Natürlich kann es auch Gründe geben, sich einen einzelnen Datensatz mit allen
Eigenschaften herauszugreifen. Oder anders ausgedrückt, vielleicht möchte man in
der Tabelle eine einzelne Zeile auswählen. Dazu gibt es das Attribut `.loc`.
Danach werden wieder eckige Klammern benutzt, wobei diesmal der Zeilenindex
verwendet wird.

Der folgende Code-Schnippsel speichert die Zeile des 4. Autos (= BMW Nr. 1) in
der Variable `viertes_auto` ab. Wir ermitteln gleich den Datentyp dazu.

```{code-cell}
viertes_auto = tabelle.loc['BMW Nr. 1']
type(viertes_auto)
```

Auch eine einzelne Zeile ist eine Series-Datenstruktur, die wir mit den
Series-Methoden weiter bearbeiten können. Der Name des Series-Objektes ist
diesmal der alte Zeilenindex. Wir lassen den Datensatz mit `.head()` anzeigen.

```{code-cell}
viertes_auto.head()
```

### Zugriff auf Zellen

Es kann auch vorkommen, dass man gezielt auf eine einzelne Zelle zugreifen
möchte. Auch dazu benutzen wir das Attribut `.loc[]`. Für eine einzelne Zelle
müssen wir angeben, in welcher Zeile und in welcher Spalte sich diese Zelle
befindet. Das Attribut `.loc[]` ermöglicht auch zwei Angaben, also Zeile und
Spalte, indem beide Werte durch ein Komma getrennt werden.

Wollen wir beispielsweise wissen, wann der Audi Nr. 3 zum zugelassen wurde, so
gehen wir folgendermaßen vor:

```{code-cell}
erstzulassung_audi3 = tabelle.loc['Audi Nr. 3', 'Erstzulassung']
type(erstzulassung_audi3)
```

Jetzt erhalten wir keine Series-Datenstruktur zurück, sondern den Datentyp des
Elements in dieser Zelle. In unserem Beispiel ist die Erstzulassung als String
gespeichert, den wir mit der print()-Funktion ausgeben lassen können:

```{code-cell}
print(erstzulassung_audi3 )
```

Wir Menschen können diesen String natürlich interpretieren und sehen, dass der
Audi Nr. 3 im November 2018 zum ersten Mal zugelassen wurde. Für Python ist es
an dieser Stelle aber nicht möglich, eine korrekte Interpretation des Strings zu
bieten.

### Mehrere Zeilen oder Spalten

Sollen mehrere Zeilen oder Spalten gleichzeitig ausgewählt werden, so werden die
entsprechenden Indizes als eine Liste in die eckigen Klammern gesetzt.

Der folgende Code wählt sowohl die Erstzulassung als auch den Preis aus.

```{code-cell}
mehrere_spalten = tabelle[ ['Erstzulassung', 'Preis (Euro)'] ]
mehrere_spalten.head()
```

Wenn die Spalten oder Zeilen nacheinander kommen, also zusammenhängend sind,
brauchen wir nicht alle Indizes in die Liste schreiben. Dann genügt es, den
ersten Index und den letzten Index zu nehmen und dazwischen einen Doppelpunkt zu
setzen. Diese Art, Zeilen oder Spalten auszuwählen, wird in der Informatik als
**Slicing** bezeichnet. Alle Autos der Marke Citroën werden also folgendermaßen
extrahiert:

```{code-cell}
citroens = tabelle.loc[ 'Citroen Nr. 1' : 'Citroen Nr. 5'] 
citroens.head()
```

Jetzt kann beispielsweise der mittlere Verkaufspreis aller Citroëns
folgendermaßen ermittelt werden:

```{code-cell}
mittelwert = citroens['Preis (Euro)'].mean()
print(f'Der mittlere Verkaufspreis der Citroens ist {mittelwert:.2f} EUR.')
```

Beim Slicing können wir den Angangsindex oder den Endindex oder sogar beides
weglassen. Wenn wir den Anfangsindex weglassen, fängt Pandas bei der ersten
Zeile/Spalte an. Lassen wir den Endindex weg, geht der Slice automatisch bis zum
Ende.

### Neue Spalte oder Zeile einfügen

Eine neue Spalte einzufügen, funktioniert recht einfach. Dazu wird ein neuer
Spaltenindex erzeugt.

```{code-cell}
tabelle['Verbrauch pro Leistung'] = tabelle['Verbrauch (l/100 km)'] / tabelle['Leistung (PS)']
tabelle.head(10)
```

Nach demselben Prinzip können wir einen neuen Datensatz aufnehmen und eine neue
Zeile einfügen. Da wir uns auf die Zeilen beziehen, verwenden wir wieder
`loc[]`.

```{code-cell}
tabelle.loc['Dacia Nr. 1'] = ['dacia', 'Dacia Duster', 'orange', '03/2023', 2023, 25749, 84, 114, 'Schaltgetriebe', 'Diesel', 5.3, 140, 5.0, 'Journey Blue dCi 115 4x4', 5.3/114] 
tabelle.head(11)
```

### Zusammenfassung und Ausblick Kapitel 4.2

In diesem Kapitel haben wir uns damit beschäftigt, wie tabellarische Daten
verwaltet werden. Im nächsten Kapitel geht es darum, diese zu visualisieren.

## 4.3 Scatterplots und Scattermatrix

Bei der Datenvisualisierung geht es darum, Daten durch eine Grafik so
aufzubereiten, dass Muster oder Unregelmäßigkeiten in den Daten entdeckt werden
können. Dabei kann die visuelle Darstellung der Daten helfen, Muster in den
Daten zu entdecken, aber sie kann auch irreführend sein. Abhängig davon, wie die
Art der Daten beschaffen ist, die wir visualisieren wollen, gibt es verschiedene
Darstellungsformen, die sogenannten **Diagrammtypen**. Im Folgenden betrachten wir
die Diagrammtypen

* Scatterplot und
* Scattermatrix.

Danach beschäftigen wir uns mit der Gestaltung bzw. dem Styling von Diagrammen.

## Lernziele Kapitel 4.3

* Sie können mit der Funktion **scatter()** einen **Scatterplot** erzeugen, der
  numerische Daten als Ursache-Wirkungs-Diagramm visualisiert.
* Sie kennen die folgenden Styling-Optionen
  * Textannotation **text=**,
  * Farbe **color=** und
  * Größe **size=**.
* Sie können mit **title=** den Titel des Diagramms setzen.
* Sie können eine **Scattermatrix** mit **scatter_matrix()** erzeugen und
  interpretieren.

### Scatterplots

Scatterplots (deutsch: Streudiagramme) werden eingesetzt, wenn der Zusammenhang
zwischen zwei numerischen Größen untersucht werden soll. Das ist vor allem bei
Experimenten häufig der Fall.

Im Folgenden soll der Scatterplot anhand des Autoscout24-Beispiels
{download}`Download autoscout24_xxs.csv
<https://gramschs.github.io/book_ml4ing/data/autoscout24_xxs.csv>` demonstriert
werden. Dazu laden wir die Tabelle wie üblich mit Pandas.

```{code-cell}
import pandas as pd
daten = pd.read_csv('autoscout24_xxs.csv', index_col=0)
daten.info()
```

Uns interessieren zunächst die Verkaufspreise der Autos. Zu jedem Auto soll
entlang der y-Achse der Verkaufspreis aufgetragen werden. Dazu wird zuerst
Plotly Express mit der üblichen Abkürzung px importiert. Danach nutzen wir die
Funktion `scatter()`. Das erste Argument in den runden Klammern ist die
komplette Tabelle, also `daten`. Danach geben wir direkt den Spaltenindex der
Spalte an, die visualisiert werden soll, also `y = 'Preis (Euro)'`. Zuletzt
lassen wir den Scatterplot auch mit `.show()` anzeigen.

```{code-cell}
import plotly.express as px
diagramm = px.scatter(daten, y = 'Preis (Euro)')
diagramm.show()
```

Da wir für die x-Achse keine Angaben gemacht haben, wird automatisch der
Zeilenindex für die x-Achse verwendet.

Der Scatterplot bietet im Vergleich zum Boxplot weitere Informationen.
Beispielsweise erkennen wir nun, dass die Autos der Marke Citroën eher unter dem
Durchschnitt liegen. Scatterplots bieten uns auch die Möglichkeit, Muster zu
visuell zu erkunden, um Abhängigkeiten von Ursache und Wirkung zu erkunden. Wir
könnten beispielsweise auf die Idee kommen, dass der Preis (= Wirkung) auch
abhängig ist von der Anzahl der gefahrenen Kilometer (= Ursache). Wir setzen die
vermutete Ursache auf die x-Achse mit dem Argument `x = 'Kilometerstand (km)'`
und die vermutete Wirkung auf die y-Achse mit `y = 'Preis (Euro)'`.

```{code-cell}
diagramm = px.scatter(daten, x = 'Kilometerstand (km)', y = 'Preis (Euro)')
diagramm.show()
```

Von der Tendenz her scheint unsere Vermutung richtig zu sein. Je mehr Kilometer
ein Auto bereits gefahren wurde, desto günstiger ist sein Verkaufspreis.
Allerdings scheint es zwei Autos zu geben, die nicht ganz in dieses Muster
passen. Ein Auto wird trotz eines Kilometerstandes von 117433 km für 46 TEUR
angeboten, an anderes hat nur 15200 km auf dem Buckel, soll aber trotzdem für
nur 12 TEUR verkauft werden. Aber welche Autos sind die beiden Ausnahmen? Um
mehr Informationen aus den Daten zu holen, beschäftigen wir uns mit dem Styling
von Scatterplots.

### Styling von Scatterplots

Die Voreinstellungen von Plotly sind bereits sehr gut gewählt, so dass ohne
weitere Optionen bereits gut aussehende und informative Diagramme erstellt
werden können. Eine Möglichkeit, durch das Styling der Diagramme
Zusatzinformationen zu visualisieren, bietet die Option `text=`. Wir verwenden
den Zeilenindex als Text, der in dem Attribut `.index` gespeichert ist.

```{code-cell}
diagramm = px.scatter(daten, x = 'Kilometerstand (km)', y = 'Preis (Euro)', text=daten.index)
diagramm.show()
```

An jedem Datenpunkt wird nun zusätzlich die Auto-ID eingeblendet. Leider
überschreibt der Text den Datenpunkt selbst. Das kann nachträglich geändert
werden, indem die Textposition relativ zu den Datenpunkten auf einen anderen
Wert gesetzt wird. Die einzelnen Bestandteile eines Plotly-Express-Diagramms
heißen **trace**. Sie werden durch `update_traces()` aktualisiert oder anders
ausgedrückt, die Voreinstellungen werden dadurch überschrieben. Wir möchten,
dass die Position der Texte oberhalb der Datenpunkte ist, aber dennoch zentriert
zum Datenpunkt. Durch das Argument `textposition='top center'` erreichen wir
dieses Ziel, wie der folgende Scatterplot zeigt.

```{code-cell}
diagramm = px.scatter(daten, x = 'Kilometerstand (km)', y = 'Preis (Euro)', text=daten.index)
diagramm.update_traces(textposition='top center')
diagramm.show()
```

Als nächstes möchten wir weitere Zusatzinformationen in das Diagramm packen.
Nicht immer ist es sinnvoll, so viele Zusatzinformationen in ein Diagramm zu
bringen, da damit das Publikum auch schnell überfordert werden kann. Daher
sollte gut überlegt werden, ob die beiden nächsten Möglichkeiten gleichzeitig
genutzt werden sollen.  

Die Farbe ist eine weitere Möglichkeit, Zusatzinformationen zu visualisieren.
Wie alt ist ein Auto? Hat es ebenfalls einen Einfluss auf den Verkaufspreis? Wir
nutzen das Jahr der Erstzulassung, um das Alter der Autos abzuschätzen. Die
Anweisung an Python, die Punkte des Scatterplots nach der Erstzulassung
einzufärben, wird durch das optionale Argument `color='Jahr'` gegeben.

```{code-cell}
diagramm = px.scatter(daten, x = 'Kilometerstand (km)', y = 'Preis (Euro)', text=daten.index, color='Jahr')
diagramm.update_traces(textposition='top center')
diagramm.show()
```

Die Farbe scheint links gelber zu sein als rechts, wo Auto 'Audi Nr. 1' violett
gefärbt ist. Also scheint das Jahr der Erstzulassung und damit das Alter der
Fahrzeuge auch etwas mit dem Kilometerstand zu tun zu haben, der auf der x-Achse
aufgetragen ist. Je jünger das Fahrzeug ist, desto weniger Kilomter wurde es
bisher gefahren.

Als zweite Möglichkeit, Zusatzinformationen direkt mit den Datenpunkten im
Scatterplot zu visualisieren, dient die Größe der Punkte. Mit dem optionalen
Argument `size=` wird sie gesteuert. Wiederum verwenden wir einen Spaltenindex
als Argument. Die Leistung könnte erfahrungsgemäß ebenfalls den Verkaufspreis
beeinflussen. Also setzen wir `size='Leistung (PS)'` und betrachten das so
erweiterte Diagramm.

```{code-cell}
diagramm = px.scatter(daten, x = 'Kilometerstand (km)', y = 'Preis (Euro)', text=daten.index, color='Jahr', size='Leistung (PS)')
diagramm.update_traces(textposition='top center')
diagramm.show()
```

Das Auto 'BMW Nr. 1', das uns schon zuvor aufgefallen ist, weil der Preis recht
hoch ist, obwohl das Auto schon einen mittleren Kilomterstand hat, scheint
besonders viel PS zu haben. Vielleicht erklärt das den hohen Preis?

Als letzte Styling-Möglichkeit betrachten wir den Titel. Im Gegensatz zu den
vorherigen Styling-Möglichkeiten, ist der Titel stets Pflicht. **Jedes Diagramm
muss einen Titel haben!** Der Titel wird mit dem Argument `title=` gesetzt.

```{code-cell}
diagramm = px.scatter(daten, x = 'Kilometerstand (km)', y = 'Preis (Euro)', text=daten.index, color='Jahr', size='Leistung (PS)', title='Verkaufsdaten von 10 Autos (Quelle: Autocout24.de)')
diagramm.update_traces(textposition='top center')
diagramm.show()
```

### Scattermatrix

Unsere Tabelle hat sieben Spalten mit numerischen Werten: Jahr, Preis (Euro),
Leistung (kW), Leistung (PS), Verbrauch (l/100 km), Verbrauch (g/km) und
Kilometerstand (km). Damit können sieben Eigenschaften als Ursache interpretiert
werden und auf der x-Achse aufgetragen werden. Zu jeder dieser sieben
Eigenschaften können dann die verbleibenden sechs Eigenschaften als Wirkung
interpretiert werden und auf der y-Achse dargestellt werden. Also müssten wir 42
Scatterplots untersuchen. Die Scattermatrix vereinfacht das Zusammenstellen
dieser Kombinationen. Dazu legen wir erst eine Liste mit den Spaltenindizes an,
die in die Scattermatrix aufgenommen werden sollen. Danach erzeugen wir mit der
Funktion `scatter_matrix()` die gewünschten Kombinationen. Als erstes Argument
werden die Daten aus der Tabelle übergeben, dann folgt die Liste der
ausgewählten Spalten als Argument für den Parameter `dimensions=`.

```{code-cell}
auswahl = ['Jahr', 'Preis (Euro)', 'Leistung (kW)', 'Leistung (PS)', 'Verbrauch (l/100 km)', 'Verbrauch (g/km)', 'Kilometerstand (km)']
diagramm = px.scatter_matrix(daten, dimensions=auswahl)
diagramm.show()
```

Es werden 49 Diagramme angezeigt, die allerdings kaum lesbar sind. Warum 49 und
nicht 42? Tatsächlich wird auch jede Eigenschaft als Ursache mit ihrer Wirkung
auf sich selbst dargestellt. Da das Diagramm so kaum lesbar ist, reduzieren wir
die Auswahl weiter und nehmen nur die ersten vier Eigenschaften.

```{code-cell}
auswahl = ['Jahr', 'Preis (Euro)', 'Leistung (kW)', 'Leistung (PS)']
diagramm = px.scatter_matrix(daten, dimensions=auswahl)
diagramm.show()
```

Auf der Diagonalen befinden sich die Scatterplots, bei denen dieselbe
Eigenschaft auf der x- und auf der y-Achse aufgetragen ist. Daher müssen diese
Punkte immer auf der Winkelhalbierenden liegen. Diese Darstellung zeigt uns
schnell, dass die Auswahl der Autos nicht gleichmäßig bezogen auf das Jahr der
Erstzulassung erfolgt ist. Im Scatterplot Jahr - Jahr ist ein Punkt (1997) sehr
weit von den restlichen Autos entfernt. Beim Preis hingegen sieht es besser aus,
diese Punkte sind entlang der Winkelhalbierenden relativ gleichmäßig verteilt.
Allerdings zeigen beide Scatterplots für die Leistung wiederum, dass ein Auto
(kW = 294 bzw. PS = 400) von den anderen Autos entfern ist. Bei beiden Autos
könnte man argumentieren, dass sie nicht repräsentativ für den Datensatz sind,
sondern als Ausreißer betrachtet werden müssen. Es stellt sich die Frage, ob sie
für die weitere Datenverarbeitung aus dem Datensatz gelöscht werden sollen.  

Betrachten wir den Scatterplot Leistung (kW) vs. Leistung (PS), so stellen wir
fest, dass die Punkte ebenfalls auf der Winkelhalbierenden liegen. Tatsächlich
ist das nicht verwunderlich, da die Leistung ja nur eine einzige Eigenschaft
darstellt, aber in zwei verschiedenen Einheiten angegeben wird. 1 Watt (W) sind
ungefähr 0,00136 Pferdestärken (PS). Die Scattermatrix zeigt uns nun (wenn wir
es nicht schon vorher wussten), dass wir nur eine der beiden Spalten brauchen.
Diese Spalte könnte für die weitere Datenexploration gelöscht werden.

```{code-cell}
auswahl = ['Jahr', 'Preis (Euro)', 'Leistung (kW)']
diagramm = px.scatter_matrix(daten, dimensions=auswahl)
diagramm.show()
```

Als letztes Interpretationsbeispiel betrachten wir die zweite Zeile der Matrix,
wo der Preis (Euro) auf der y-Achse aufgetragen ist. Betrachten wir den ersten
Scatterplot der zweiten Zeile, so scheint das Jahr keinen besonderen Einfluss
auf den Verkaufspreis zu haben. Beim dritten Scatterplot in der zweiten Zeile,
scheint es aber ein Muster zu geben. Mit wachsender Leistung scheint auch der
Verkaufspreis zu steigen. Die Scattermatrix hilft also gerade zu Beginn der
Datenexploration schnell, interessante Zusammenhänge zwischen einzelnen
Eigenschaften aufzudecken, die dann durch einzelne Scatterplots näher untersucht
werden können.

### Zusammenfassung und Ausblick Kapitel 4.3

In diesem Kapitel haben wir uns mit der Visualisierung von numerischen Werten
beschäftigt. Die Scattermatrix ordnet alle Kombinationen von einzelnen
Scatterplots in einer Matrix an. Damit können schnell Muster in den Daten
gefunden werden, deren Abhängigkeiten dann wiederum durch einzelne Scatterplots
detaillierter beleuchtet werden können. Bisher haben wir aber nur die
numerischen Werte untersucht. Wie auch die nicht-numerischen Werte wie
beispielsweise die Farbe der Autos mit in die Visualisierung einbezogen werden
können, sehen wir im nächsten Kapitel.

## Übungen

Daten für die Übungen:

* [12612-0001_de.csv](https://nextcloud.frankfurt-university.de/s/ddfDzAAnJtJ4FZQ)
* [stromverbrauch_hessen.csv](https://nextcloud.frankfurt-university.de/s/bpZ8TZrzafKqAxq)

### Übung 4.1 - Teilaufgabe a)

Schauen Sie sich die csv-Datei '12612-0001_de.csv' im Texteditor an. Der
Datensatz enthält die Lebendgeburten in Deutschland getrennt nach männlich,
weiblich und insgesmt (Quelle:
<https://www-genesis.destatis.de/datenbank/beta/statistic/12612/table/12612-0001>).
Ab welcher Zeile beginnen die Daten und in welcher Zeile enden sie? Importieren
Sie dann die Daten, wobei Kopf- und Fußzeilen beim Import übersprungen werden
sollen. Welche Spalte ist als Index-Spalte geeignet? Setzen Sie eine passende
Spalte als Index. Verschaffen Sie sich einen Überblick über die Daten und
erstellen Sie eine Übersicht der statistischen Kennzahlen.

Tipp: Importieren Sie pandas mit dem üblichen Alias pd und führen Sie dann die
folgende Code-Zeile in einer Code-Zelle aus:

```none
pd.read_csv?
```

Welches Argument könnte für das Überspringen der Fußzeilen stehen?

## Übung 4.1 - Teilaufgabe b)

Kontrollieren Sie, ob die Spalte 'insgesamt' tatsächlich die Summe der beiden
Spalten 'männlich' und 'weiblich' ist. Bilden Sie dazu die Differenz 'insgesamt' -
'männlich' - 'weiblich' und fügen Sie diese Differenz als neue Spalte dem
DataFrame hinzu. Wie lauten die statistischen Kennzahlen dieser Spalte? Welchen
Schluss ziehen Sie daraus? Haben Sie eine Vermutung, was passiert ist?

### Übung 4.1 - Teilaufgabe c)

Lassen Sie die statistischen Kennzahlen der Lebendgeburten 'männlich' und
'weiblich' als Boxplot visualisieren. Wurden mehr Jungen oder Mädchen geboren?
Liegt der Median mittig zwischen Q1 und Q3?

### Übung 4.1 - Teilaufgabe d)

Visualisieren Sie die Anzahl der männlichen und weiblichen Lebendgeburten pro
Jahr als Scatterplot. Beschriften Sie auch die Achsen und setzen Sie einen
Titel.

### Übung 4.2 - Teilaufgabe a)

Schauen Sie sich die csv-Datei 'stromverbrauch_hessen.csv' im Texteditor an.
Welche Daten enthält die Datei? Ab welcher Zeile beginnen die Daten und in
welcher Zeile enden sie? Importieren Sie dann die Daten, wobei Kopf- und
Fußzeilen beim Import übersprungen werden sollen. Welche Spalte ist als
Index-Spalte geeignet? Setzen Sie eine passende Spalte als Index. Verschaffen
Sie sich einen Überblick über die Daten und erstellen Sie eine Übersicht der
statistischen Kennzahlen.

## Übung 4.2 - Teilaufgabe b)

Checken Sie, ob die Spalte 'insgesamt' tatsächlich die Summe der anderen Spalten ist.

## Übung 4.2 - Teilaufgabe c)

Fertigen Sie Boxplots an und bewerten Sie die statistischen Kennzahlen.

## Übung 4.2 - Teilaufgabe d)

Visualisieren Sie den Stromverbrauch der drei relvanten Sektoren Industrie,
Verkehr und Haushalte abhängig vom Jahr in einem gemeinsamen Scatterplot. Setzen
Sie einen Titel und beschriften Sie auch die Achsen.

Gibt es Auffälligkeiten?
