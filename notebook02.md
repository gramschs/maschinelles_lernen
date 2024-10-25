---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 2. Crashkurs Python

In dieser Vorlesung werden wir Python für maschinelles Lernen nutzen. Daher
folgt in diesem Part ein Crashkurs in Python. Wer daran interessiert ist, die
Grundlagen von Python zu erlernen, bevor es mit maschinellem Lernen losgeht, ist
herzlich dazu eingeladen, das
[Python-Vorlesungsskript](https://gramschs.github.io/book_python/intro.html) für
Studierende der Ingenieurwissenschaften nutzen.

## 2.1 Datentypen, Variablen und print()

Beim maschinellen Lernen geht es um Daten und Algorithmen. Dabei können die
Daten alles Mögliche umfassen, beispielsweise Zahlen oder Texte. Daher
beschäftigen wir uns zuerst mit Datentypen. Dann geht es um Variablen und deren
Ausgabe auf dem Bildschirm.

### Lernziele Kapitel 2.1

* Sie kennen die einfachen Datentypen:
  * **Integer**
  * **Float**
  * **String**
* Sie wissen, was eine **Variable** ist und kennen den **Zuweisungsoperator**.
* Sie können die **print()**-Funktion zur Ausgabe auf dem Bildschirm anwenden.

### Einfache Datentypen

Beim maschinellen Lernen geht es um die Sammlung, Erkundung und Analyse von
Daten, um Antworten auf vorgegebene Fragen zu finden. Schematisch stellen wir
maschinelles Lernen folgendermaßen dar:

![Schematische Darstellung der Vorgehensweise im maschinellen Lernen](https://gramschs.github.io/book_ml4ing/_images/fig01_prozess.png)

Der Computer kann Informationen aber nur als 0 und 1 verarbeiten. Auf dem
Speichermedium oder im Speicher selbst werden Daten daher als eine Folge von 0
und 1 gespeichert. Damit es für uns Programmierinnen und Programmierer einfacher
wird, Daten zu speichern und zu verarbeiten, wurden Datentypen eingeführt.  

**Datentypen** fassen gleichartige Objekte zusammen und stellen den
Programmierer:innen passende Operationen zur Verfügung. Mit Operationen sind die
Aktionen gemeint, die mit diesen Datenobjekten durchgeführt werden dürfen.
Zahlen dürfen beispielsweise addiert werden, Buchstaben aber nicht. Es hängt von
der Programmiersprache ab, welche Datentypen zur Verfügung stehen, wie diese im
Hintergrund gespeichert werden und welche Operationen möglich sind.

In diesem Kapitel beschäftigen wir uns mit den einfachen Datentypen

* Integer,
* Float und
* String.

#### Integer und Float

In der Programmierung unterscheidet man grundsätzlich zwischen zwei Zahlenarten,
den Ganzzahlen und den Gleitkommazahlen/Fließkommazahlen. Die Ganzzahlen werden
in der Mathematik als **ganze Zahlen** bezeichnet. In der Informatik wird jedoch
der englische Begriff dafür verwendet: **Integer**.

Mit Integern können wir ganz normal rechnen, also arithmetische Operationen
ausführen:

```{code-cell} ipython3
2+3
```

```{code-cell} ipython3
2*3
```

```{code-cell} ipython3
6-7
```

```{code-cell} ipython3
3*(4+7)
```

```{code-cell} ipython3
25/5
```

```{code-cell} ipython3
4**2
```

Mit einer Operation verlassen wir aber bereits den Bereich der ganzen Zahlen,
den Bereich der Integers. `25/5` ist wieder eine ganze Zahl, nicht jedoch
`25/3`. Damit sind wir bei den Fließkommazahlen. Auch hier wird üblicherweise
der englische Begriff **Float** für **Fließkommazahl** verwendet.

```{code-cell} ipython3
2.3 + 4.5
```

```{code-cell} ipython3
5.6 - 2.1
```

```{code-cell} ipython3
2.1 * 3.5
```

```{code-cell} ipython3
3.4 / 1.7
```

```{code-cell} ipython3
3.4 ** 2
```

```{code-cell} ipython3
3.5 * (2.6 - 3.8 / 1.9)
```

Das folgende Video gibt eine Einführung in das Thema »Zahlen mit Python«: <https://youtu.be/VtiDkRDPA_c>.

#### String

Daten sind aber sehr oft keine Zahlen. Beispielsweise könnte man sich
vorstellen, eine Einkaufsliste zu erstellen und diese im Computer oder in einer
Notiz-App auf dem Handy zu speichern. Eine solche **Zeichenkette** heißt in der
Informatik **String**. Mit Zeichen meint man dabei Zahlen, Buchstaben oder
andere Zeichen wie beispielsweise !"§$%&/()=?.

Strings werden in Python durch einfache Hochkomma oder Anführungszeichen
definiert.

```{code-cell} ipython3
'Dies ist ein String!'
```

Mit Strings kann man ebenfalls "rechnen", nur ist das Ergebnis vielleicht anders
als erwartet.

```{code-cell} ipython3
2 * 'Dies ist ein String!'
```

```{code-cell} ipython3
'String 1 ' + 'String 2' 
```

Das folgende Video gibt eine Einführung zum Thema »Strings in Python«: <https://youtu.be/sTEf4_mrLvw>.

### Variablen und Zuweisung

**Variablen** sind beschriftete Schubladen. Oder anders formuliert sind
Variablen Objekte, denen man einen Namen gibt. Technisch gesehen sind diese
Schubladen ein kleiner Bereich im Arbeitsspeicher des Computers. Was in diesen
Schubladen aufbewahrt wird, kann sehr unterschiedlich sein. Beispielsweise die
Telefonnummer des ADAC-Pannendienstes, die 10. Nachkommastelle von $\pi$ oder
die aktuelle Position des Mauszeigers können in den Schubladen enthalten sein.

#### Zuweisungen sind keine Gleichungen

Wir verwenden Variablen, um bestimmte Werte oder ein bestimmtes Objekt zu
speichern. Eine Variable wird durch eine **Zuweisung** erzeugt. Damit meinen
wir, dass eine Schublade angelegt wird und die Schublade dann erstmalig gefüllt
wird. Das erstmalige Füllen der Schublade nennt man in der Informatik auch
Initialisieren. Für die Zuweisung wird in Python das `=`-Zeichen verwendet.

```{code-cell} ipython3
x = 0.5
```

Sobald die Variable `x` in diesem Beispiel durch eine Zuweisung von 0.5 erstellt
wurde, können wir sie verwenden:

```{code-cell} ipython3
x * 3
```

```{code-cell} ipython3
x + 17
```

Wichtig ist, dass das `=` in der Informatik eine andere Bedeutung hat als in der
Mathematik. `=` meint nicht das Gleichheitszeichen, sondern den sogenannten
**Zuweisungsoperator**. Das ist in der Programmierung ein Kommando, das eine
Schublade befüllt oder technischer ausgedrückt, ein Objekt einer Variable
zuweist.

Variablen müssen initalisiert (erstmalig mit einem Wert versehen) werden, bevor
sie verwendet werden können, sonst tritt ein Fehler auf.

##### Mini-Übung 1

Fügen Sie eine Code-Zelle ein und schreiben Sie in die Code-Zelle einfach nur `n`. Lassen Sie die Code-Zelle ausführen. Was passiert?

```{code-cell} ipython3
# hier Ihr Code
```

Sehr häufig findet man Code wie

```{code-cell} ipython3
x = x + 1
```

Würden wir dies als Gleichung lesen, wie wir es aus der Mathematik gewohnt sind,
x = x + 1, könnten wir x auf beiden Seiten subtrahieren und erhalten 0 = 1. Wir
wissen, dass dies nicht wahr ist, also stimmt hier etwas nicht.

In Python sind Gleichungen keine mathematischen Gleichungen, sondern
Zuweisungen. "=" ist kein Gleichheitszeichen im mathematischen Sinne, sondern
eine Zuweisung. Die Zuweisung muss immer in der folgenden Weise zweistufig
gelesen werden:

1. Berechne den Wert auf der rechten Seite (also x+1).
2. Weise den Wert auf der rechten Seite dem auf der linken Seite stehenden
   Variablennamen zu (in Python-Sprechweise: binde dem Namen auf der linken
   Seite an das auf der rechten Seite angezeigte Objekt).

```{code-cell} ipython3
x = 4     
x = x + 1
x
```

#### Richtlinien für Variablennamen

Früher war der Speicherplatz von Computern klein, daher wurden häufig nur kurze
Variablennamen wie beispielsweise `i` oder `N` verwendet. Heutzutage ist es
Standard, nur in Ausnahmefällen (z.B. in Schleifen, dazu kommen wir noch) kurze
Variablennamen zu nehmen. Stattdessen werden Namen benutzt, bei denen man
erraten kann, was die Variable für einen Einsatzzweck hat. Beispielsweise lässt
der Code

```{code-cell} ipython3
m = 0.19
n = 80
b = n + m*n
b
```

nur schwer vermuten, was damit bezweckt wird. Dagegen erahnt man bei diesem Code
schon eher, was bezweckt wird:

```{code-cell} ipython3
mehrwertsteuersatz = 19/100
nettopreis = 80
bruttopreis = nettopreis + mehrwertsteuersatz * nettopreis
bruttopreis
```

Code-Zellen zeigen arithmetische Operationen direkt nach Ausführen der
Code-Zelle an. Wenn allerdings in der Code-Zelle mehrere Python-Anweisungen
sind, müssen wir in die letzte Zeile nochmal die Variable selbst hinschreiben,
deren Wert angezeigt werden soll. Das ist etwas umständlich und funktioniert
auch nur mit Jupyter Notebooks. Normalerweise gibt es dazu eine Python-Funktion
`print()`, auf die wir später noch zurückkommen.

Verwenden Sie für Variablennamen nur ASCII-Zeichen, also keine Umlaute wie ö, ü
oder ß. Zahlen sind erlaubt, aber nicht am Anfang des Namens. Es ist sinnvoll,
lange Variablen durch einen Unterstrich besser lesbar zu gestalten (sogenannte
Snake-Case-Formatierung). Ich empfehle für Variablennamen beispielsweise

`dateiname_alt` oder `dateiname_neu`

wenn beispielsweise eine Datei umbenannt wird. Sie sind frei in der Gestaltung
der Variablennamen, verboten sind nur die sogannnten Schlüsselwörter.

Die folgenden beiden Videos fassen die beiden Themen Variablen und Zuweisungen
nochmals zusammen.

* <https://youtu.be/jfOLXKPGXJ0>
* <https://youtu.be/XKFQ2_et5k8>

#### Datentypen ermitteln mit type()

Werden zwei Integer geteilt, so wird das Ergebnis automatisch in einen Float
umgewandelt. Mit Hilfe der Funktion `type()` können wir den Python-Interpreter
bestimmen lassen, welcher Datentyp in einer Variable gespeichert ist. Mit
Funktion ist an dieser Stelle keine mathematische Funktion gemeint. Eine
**Funktion** sind viele Anweisungen nacheinander, um eine bestimmte Teilaufgabe
zu lösen. Damit klar ist, dass es sich um eine Funktion und nicht um eine
Variable handelt, werden runde Klammern an den Namen der Funktion gehängt.

In diesem Fall soll der Datentyp eines Objektes ermittelt werden. Damit der
Python-Interpreter weiß, von welcher Variable der Datentyp ermittelt werden
soll, schreiben wir die Variable in runde Klammern.

```{code-cell} ipython3
x = 25 * 5
type(x)
```

```{code-cell} ipython3
x = 25 / 5
type(x)
```

Nicht immer ist es aber möglich, Datentypen zu mischen. Dann meldet Python einen
Fehler.

Das folgende Video fasst die Datentypen Integer, Float und String nochmal zusammen: <https://youtu.be/1WqFJ5wsA4o>.

### Ausgaben mit print()

Jetzt lernen Sie eine weitere Python-Funktion kennen. Bei den obigen
Rechenaufgaben wurde automatisch das Ergebnis der Rechnung angezeigt, sobald die
Code-Zelle ausgeführt wurde. Dies ist eine Besonderheit der Jupyter Notebooks,
würde aber in einem normalen Python-Programm nicht funktionieren. Auch möchte
man vielleicht ein Zwischenergebnis anzeigen lassen. Die interaktive Ausgabe der
Jupyter Notebooks zeigt jedoch immer nur den Inhalt der letzten Zeile an.

Für die Anzeige von Rechenergebnissen oder Texten gibt es in Python die
**print()**-Funktion. Die print()-Funktion in Python gibt den Wert am Bildschirm
aus, der ihr als sogenanntes **Argument** in den runden Klammern übergeben wird.
Das kann zum Beispiel eine Zahl oder eine Rechenaufgabe sein, wie in dem
folgenden Beispiel.

```{code-cell} ipython3
print(2)
print(3+3)
```

In der ersten Zeile ist das Argument für die print()-Funktion die Zahl 2. Das
Argument wird in runde Klammern hinter den Funktionsnamen `print` geschrieben.
Ein Argument ist sozusagen der Input, der an die print()-Funktion übergeben
wird, damit der Python-Interpreter weiß, welcher Wert auf dem Bildschirm
angezeigt werden soll.

Das zweite Beispiel in der zweiten Zeile funktioniert genauso. Nur wird diesmal
eine komplette Rechnung als Argument an die print()-Funktion übergeben. In dem
Fall rechnet der Python-Interpreter erst den Wert der Rechnung, also `3+3=6` aus
und übergibt dann die `6` an die print()-Funktion. Die print()-Funktion wiederum
zeigt dann die `6` am Bildschirm an.

#### Mini-Übung 2

Lassen Sie Python den Term $3:4$ berechnen und geben Sie das Ergebnis mit der print()-Funktion aus.

```{code-cell} ipython3
# Geben Sie nach diesem Kommentar Ihren Code ein:
```

Python kann mit der print()-Funktion jedoch nicht nur Zahlen ausgeben, sondern
auch Texte, also Strings.

```{code-cell} ipython3
print('Hallo')
```

#### Mini-Übung 3

Probieren Sie aus was passiert, wenn Sie die einfachen Anführungszeichen `'`
durch doppelte Anführungszeichen `"` ersetzen. Lassen Sie den Text Hallo Welt
ausgeben :-)

```{code-cell} ipython3
# Geben Sie nach diesem Kommentar Ihren Code ein:
```

Zum Schluss behandeln wir noch formatierte Strings, die sogenannten f-Strings.
Seit Python 3.6 erleichtert dieser Typ von String die Programmierung. Falls Sie
Python-Code sehen, in dem Prozentzeichen vorkommen (ganz, ganz alt) oder die
`.format()`-Methode benutzt wird, wundern Sie sich nicht. In dieser Vorlesung
verwenden wir jedoch f-Strings.

**f-Strings** sind die Abkürzung für "formatted string literals". Sie
ermöglichen es, den Wert einer Variable oder einen Ausdruck direkt in den String
einzubetten. Dazu werden geschweifte Klammern verwendet, also `{` und `}` und zu
Beginn des Strings wird ein `f` eingefügt, um aus dem String einen f-String zu
machen. Der Python-Interpreter fügt dann zur Laufzeit des Programms den
entsprechenden Wert der Variable in den String ein.

Hier ein Beispiel:

```{code-cell} ipython3
name = 'Alice'
alter = 20
print(f'Mein Name ist {name} und ich bin {alter} Jahre alt.')
```

Insbesondere bei Ausgabe von Zahlen sind f-Strings besonders nützlich. Wenn nach
dem Variablennamen ein Doppelpunkt eingefügt wird, kann danach die Anzahl der
gewünschten Stellen vor dem Komma (hier natürlich ein Punkt) und der
Nachkommastellen festgelegt werden. Zusätzlich setzen wir ein `f` in die
geschweiften Klammern, um einen Float anzeigen zu lassen. Im folgenden Beispiel
geben wir $\pi$ auf zwei Nachkommastellen an.

```{code-cell} ipython3
pi = 3.141592653589793238462643383279
print(f'Pi = {pi:1.2f}')
```

Es ist schwierig, sich alle Formatierungsoptionen zu merken. Auf der
Internetseite
[https://cheatography.com/brianallan/cheat-sheets/python-f-strings-basics/](https://cheatography.com/brianallan/cheat-sheets/python-f-strings-basics/)
finden Sie eine umfangreiche Übersicht und können sich zudem ein pdf-Ddokument
herunterladen.

#### Mini-Übung 4

Schreiben Sie ein Programm, mit dem der Flächeninhalt eines Rechtecks berechnet werden soll. Die beide Seitenlängen werden jeweils in den Variablen `laenge` und `breite` gespeichert (suchen Sie sich eigene Zahlen aus). Ausgegeben werden soll dann: "Der Flächeninhalt eines Rechtecks mit den Seiten XX und XX ist XX.", wobei XX durch die korrekten Zahlen ersetzt werden und der Flächeninhalt auf eine Nachkommastelle gerundet werden soll.

```{code-cell} ipython3
#
```

### Zusammenfassung und Ausblick Kapitel 2.1

In diesem Kapitel haben wir gelernt, was ein Datentyp ist, wie eine Variable mit
einem Wert gefüllt wird und mit der print()-Funktion am Bildschirm ausgegeben
wird. Die einfachsten Datentypen Integer, Float und String reichen allerdings
nicht aus, um z.B. die eine Adresse mit Straße (String), Hausnummer (Integer)
und Postleitzahl (Integer) in einer Variablen gemeinsam zu speichern. Dazu
lernen wir im nächsten Abschnitt den Datentyp Liste kennen

## 2.2 Listen und for-Schleifen

Bisher haben wir drei verschiedene Datentypen kennengelernt:

* Integer (ganze Zahlen),
* Floats (Fließkommazahlen) und
* Strings (Zeichenketten).

Damit können wir einzelne Objekte der realen Welt ganz gut abbilden. Mit einem
String können wir den Namen einer Person erfassen, mit einem Integer das Alter
der Person und mit einem Float die Körpergröße der Person gemessen in Meter. Was
uns aber bisher fehlt ist eine Sammlung von Namen oder eine Sammlung von
Körpergrößen. Daher werden wir uns in diesem Kapitel mit dem Datentyp **Liste**
beschäftigen.

Oft kommt es vor, dass für jedes Element der Liste bestimmte Aktionen
durchgeführt werden sollen. Daher werden wir uns auch mit der Wiederholung von
Code-Abschnitten mittels der sogenannten **for-Schleife** beschäftigen.

### Lernziele Kapitel 2.2

* Sie kennen den Datentyp **Liste**.
* Sie können Listen mit eckigen Klammern erzeugen.
* Sie können Listen mit dem **Plus-Operator** verketten, Elemente mit
  **.append()** anhängen oder Werte mit **.remove()** aus der Liste löschen.
* Sie können über den **Index** auf einzelne Listenelementer zugreifen.
* Sie können eine **for-Schleife mit Liste** programmieren.
* Sie wissen, wie die Fachbegriffe der einzelnen Bestandteil der Schleife
  lauten:
  * **Kopfzeile**, wird mit **Doppelpunkt :** abgeschlossen
  * Schlüsselwörter **for** und **in**
  * **Schleifenvariable**  
* Sie wissen, dass der Anweisungsblock des Schleifeninneren eingerückt werden
  muss. Die **Einrückung** muss immer mit der gleichen Anzahl von Zeichen
  (Leerzeichen oder Tab) erfolgen.

### Datentyp Liste

Eine Liste ist eine Sammlung von Objekten. Dabei können die Objekte einen
beliebigen Datentyp aufweisen. Eine Liste wird durch eckige Klammern erzeugt.

Beispielsweise könnte eine Liste drei Integer enthalten:

```{code-cell} ipython3
a = [34, 12, 54]
print(a)
```

Das folgende Beispiel zeigt eine Liste mit vier Namen, die durch Strings
repräsentiert werden:

```{code-cell} ipython3
a = ['Alice', 'Bob', 'Charlie', 'Dora']
print(a)
```

Eine leere Liste wird durch `[]` definiert:

```{code-cell} ipython3
a = []
print(a)
```

Der Datentyp heißt formal `list`:

```{code-cell} ipython3
type(a)
```

Listen können gekürzt und erweitert werden. Eine sehr nützliche Funktion ist
daher die `len()`-Funktion. Das `len` steht dabei für `length`. Wird die
Funktion `len()` mit einer Liste (oder mit einem String) als Argument
aufgerufen, gibt sie die Anzahl der Listenelemente (oder Anzahl der Zeichen im
String) zurück.

```{code-cell} ipython3
a = ['Hund', 'Katze', 'Maus', 'Affe','Elefant']
len(a)
```

Listen müssen nicht nur Elemente eines Datentyps enthalten. In Python ist es
erlaubt, in eine Liste Objekte mit verschiedenen Datentypen zu sammeln. Das
folgende Beispiel zeigt eine Mischung aus Elementen der drei Datentypen Integer,
Float und String.

```{code-cell} ipython3
a = [123, 'Ente', -42, 17.4, 0, 'Elefant']
print(a)
```

#### Mini-Übung 5

Erzeugen Sie eine Einkaufsliste, um einen Obstsalat zuzubereiten und speichern
Sie diese Liste in der Variablen `einkaufsliste`. Lassen Sie dann den Computer
bzw. den Python-Interpreter zählen, wie viele Zutaten Ihre Liste enthält und
geben Sie dann die Anzahl aus.

```{code-cell} ipython3
# Hier Ihr Code:
```

### Listen bearbeiten

Listen sind in Python veränderlich. Besonders häufig kommt es vor, dass zwei
Listen zu einer neuen Liste kombiniert werden sollen. Da diese Aktion so wichtig
ist, kann dies in Python direkt mit dem `+`-Operator erledigt werden. Der
Fachbegriff für das Aneinanderhängen von Listen ist **Verkettung** oder auf
Englisch **Concatenation**.

```{code-cell} ipython3
a = [37, 3, 5] + [3, 35, 100]
print(a)
```

Um an das Ende der Liste ein neues Element einzufügen, verwendet man die Methode
`append()`. Eine **Methode** ist eine spezielle Funktion, die zu dem Datentyp
gehört und daher an die Variable angehängt wird, indem man einen Punkt schreibt
und dann den Methodennamen.

```{code-cell} ipython3
a = [34, 56, 23]
print(a)

a.append(42)
print(a)
```

Aus der Liste können Elemente durch die `remove()`-Methode gelöscht werden.
Dabei wird das Element, das gelöscht werden soll, der Methode als Argument
übergeben. Es wird der erste auftretende Wert aus der Liste gelöscht.

```{code-cell} ipython3
a = [34, 56, 23, 42]
print(a)

a.remove(56)
print(a)
```

#### Mini-Übung 6

Nehmen Sie Ihre Einkaufsliste für den Obsalat von vorhin. Fügen Sie noch Zimt und Zucker hinzu. Leider passen in Ihren Einkaufswagen nur maximal 5 Sachen. Lassen Sie daher die Anzahl der Elemente ausgeben.

```{code-cell} ipython3
# Hier Ihr Code:
```

Video: <https://youtu.be/ihF8bZoauBs>

### Zugriff auf einzelne Listenelemente

Listen sind (übrigens genau wie Strings) ein sogenannter **sequentieller
Container**. Sequentielle Container sind Sammlungen von Datenobjekten. Was heißt
das? Eine Straße ist eine Sammlung von Häusern. Aber in einer Straße sind die
Häuser zusätzlich durchnummeriert. Eine Hausnummer ermöglicht es, die Position
eines Hauses innerhalb der Straße zu bestimmen. Sammlungen von Objekten mit
einer Positionsnummer werden sequentielle Container genannt. Sie sind mit ganzen
Zahlen, also Integern, durchnummeriert. In Python beginnt die Nummerierung bei
0. Die Nummer eines Elementes aus der Sequenz nennt man **Index**.
Umgangssprachlich könnte man den Index also auch als Hausnummer bezeichnen.

In einer Liste hat also das erste Element den Index 0. Das zweite Element hat
den Index 1 usw. Um ein einzelnes Element einer Liste herausgreifen zu können,
schreibt man `liste[i]`. Dabei ist `liste` der Name der Liste und `i` der Idnex.
Um einfach auf das letzte Element einer Liste zugreifen zu können, hat Python
den Index -1 eingeführt.

Probieren wir ein Beispiel aus:

```{code-cell} ipython3
a = [34, 56, 23, 42]
erstes = a[0]
print(f'Das erste Element in der Liste ist: {erstes}')
```

```{code-cell} ipython3
# Erzeugung Liste
meine_liste = ['rot', 'grün', 'blau', 'gelb', 'weiß', 'schwarz']

# das fünfte Elment weiß wird durch lila ersetzt
meine_liste[4] = 'lila'
print(meine_liste)
```

Das Bearbeiten von einzelnen Listenelementen wird auch **Zugriff** genannt. Das
folgende Video zeigt die Zugriffsmöglichkeiten von Listen: <https://youtu.be/_XzWPXvya2w>.

### Code wiederholen mit der for-Schleife

Manchmal möchte man für jedes Element einer Liste eine oder mehrere Aktionen
durchführen. Dazu gibt es die **for-Schleife**.

```{code-cell} ipython3
for i in [2, 4, 6, 8, 10]:
    print(i)
```

Eine Schleife beginnt mit dem Schlüsselwort **for**. Danach kommt der Name der
sogenannten **Schleifenvariable**, in diesem Fall also `i`. Als nächstes folgt
wieder ein Schlüsselwort, nämlich **in** und zuletzt Liste. Diese Zeile nennt
man **Kopfzeile**.

Python muss wissen, welche Kommandos für jeden Schleifendurchgang ausgeführt
werden sollen. Daher wird die Kopfzeile der Schleife mit einem Doppelpunkt `:`
beendet. Danach werden alle Kommandos aufgelistet, die ausgeführt werden sollen.
Damit Python weiß, wann es wieder mit dem normalen Programm weitergehen soll,
müssen wir dem Python-Interpreter das Ende der Schleife signalisieren. In vielen
Programmiersprachen wird das mit dem Schlüsselwort `end` gemacht oder es werden
Klammern gesetzt. In Python wird stattdessen mit **Einrückung** gearbeitet. Alle
Zeilen mit Anweisungen, die eingerückt sind, werden in der Schleife wiederholt.

Wie sieht das nun bei unserem Beispiel aus? Die Schleifenvariable heißt `i`. Sie
nimmt beim 1. Schleifendurchgang den Wert `2` an. Dann werden die Anweisungen im
Schleifeninneren ausgeführt, also die print()-Funktion für `i = 2` angewendet
und eine 2 ausgegeben. Dann wird die Schleife ein 2. Mal durchlaufen. Diesmal
nimmt die Schleifenvariable `i` den Wert `4` an und die print()-Funktion gibt 4
aus. Das geht so weiter bis zum 5. Schleifendurchgang, wo die Schleifenvariable
den Wert `i = 10` annimmt und eine 10 auf dem Bildschirm angezeigt wird. Da die
`10` das letzte Element der Liste war, macht der Python-Interpreter mit dem
normalen Programm weiter. Bei unserem kurzen Beispiel ist aber schon das Ende
des Programmes erreicht. Zusammengefasst, werden nacheinander die Elemente der
Liste `[2, 4, 6, 8, 10]` auf dem Bildschirm ausgegeben.

Schauen wir uns ein erstes Beispiel an. Jedes Element der Liste `[4,5,7,11,21]`
soll um 2 erhöht werden.

```{code-cell} ipython3
for zahl in [4,5,7,11,21]:
    summe = zahl + 2
    print(f'Wenn ich {zahl} + 2 rechne, erhalte ich {summe}.')
print('Ich bin fertig!')
```

#### Mini-Übung 7

Lassen Sie nacheinander die Zutaten Ihrer Einkaufsliste ausgeben.

```{code-cell} ipython3
# Hier Ihr Code:
```

Video: <https://youtu.be/ISo1uqLcVw8>

Es kommt sehr häufig vor, dass über Listen mit Zahlen iteriert werden soll.
Dafür stellt Python3 die Funktion `range()`zur Verfügung.

```{code-cell} ipython3
for zahl in range(3):
    print(zahl)
print('Fertig!')
```

Wird `range(endzahl)` mit nur einem Parameter aufgerufen, dann beginnt der
Python-Interpreter stets von `0` an zu zählen. Dabei ist die `endzahl` nicht
inkludiert, d.h. der Python-Interpreter stoppt bei `endzahl - 1`. Es ist auch
möglich, eine Startzahl vorzugeben, also:

```{code-cell} ipython3
for zahl in range(1,4):
    print(zahl)
print('Fertig!')
```

Zusätzlich kann der Zahlenbereich noch durch die Angabe einer Schrittweite
spezifiziert werden. Dadurch ist es beispielsweise möglich, nur ungerade Zahlen
zu generieren:

```{code-cell} ipython3
for zahl in range(3, 13, 2):
    print(zahl)
print('Fertig!')
```

#### Mini-Übung 8

Lassen Sie alle geraden Zahlen zwischen 100 und 120 ausgeben.

```{code-cell} ipython3
# Hier Ihr Code:
```

Durch Angabe einer negativen Schrittweite kann auch rückwärts gezählt werden:

```{code-cell} ipython3
for zahl in range(13, 3, -2):
    print(zahl)
print('Fertig!')
```

Zum Abschluss folgt hier noch ein weiteres Video zu for-Schleifen: <https://youtu.be/pQh5Idw2sKM>.

### Zusammenfassung und Ausblick 2.2

In diesem Abschnitt haben wir uns mit dem Datentyp Liste befasst, der zur
Sammlung verschiedener Datenobjekte dient. Für die sequentielle Bearbeitung von
Listenelementen ist die for-Schleife besonders geeignet. Es existieren
zusätzliche Datentypen wie Dictionary, Tupel und Set, die sich ebenfalls zum
Speichern von Datenobjekten eignen. Neben der for-Schleife gibt es eine
alternative Schleifenstruktur, die while-Schleife, die Code wiederholt, solange
eine bestimmte Bedingung erfüllt ist. Anstatt uns weiterhin auf solche Aspekte
der Python-Programmierung zu konzentrieren, werden wir im nächsten Kapitel den
Fokus auf die Implementierung eigener Funktionen legen und einen kurzen Ausflug
in die objektorientierte Programmierung unternehmen

## 2.3 Funktionen und Methoden

Sobald die Funktionalitäten komplexer werden, lohnt es sich Code in eigene
Funktionsbausteine auszulagern und vor allem auf Code von anderen
Programmier:innen zurückzugreifen. Code, der eine Teilaufgabe löst und einen
eigenständigen Namen bekommt, wird **Funktion** genannt. Ist die Funktion direkt
an einen Datentyp gekoppelt, wird die Funktion **Methode** genannt. In diesem
Kapitel gehen wir sehr kurz auf die wichtigsten Grundlagen von Funktionen und
Methoden ein.

### Lernziele Kapitel 2.3

* Sie können selbst eine **Funktion** mit Parametern und Rückgabewert
  implementieren.
* Sie kennen das Konzept der **objektorientierten Programmierung**.
* Sie wissen, was **Klassen** und **Methoden** sind.

### Funktionen

Eine Funktion ist eine Zusammenfassung von Code, der eine bestimmte Teilaufgabe
löst. Dabei arbeitet die Funktion nach dem EVA-Prinzip (Eingabe, Verarbeitung,
Ausgabe). Die Funktion übernimmt Objekte als Eingabe, verarbeitet diese und
liefert Objekte als Ergebnis zurück. Wie die Funktion dabei im Inneren genau
funktioniert (Verarbeitung), ist unwichtig.

Insbesondere muss die Teilaufgabe, die die Funktion löst, nichts mit Mathematik
zu tun haben. Eine Funktion in der Informatik hat nichts mit einer
mathematischen Funktion zu tun, auch wenn oft mathematische Funktionen als
Beispiel verwendet werden. Ein Beispiel für eine nicht-mathematische Funktion
haben Sie mit `print()` bereits kennengelernt.

#### Die Benutzung von Funktionen (oder der Aufruf von Funktionen)

Eine Funktion wird benutzt, indem man den Namen der Funktion hinschreibt und
dann in runden Klammern ihre Argumente. Welche Argumente für eine Funktion
verwendet werden dürfen, hängt von der Implementierung der Funktion ab.

Beispielsweise kann als Argument für die `len()`-Funktion ein String übergeben
werden oder eine Liste.

```{code-cell} ipython3
len('Hallo')
```

```{code-cell} ipython3
len([1,2,3,4,8,2])
```

In der Regel geben Funktionen wieder Ergebnisse zurück. Diese können einer
Variable zugewiesen werden, um weiter mit dem Ergebnis zu arbeiten.

```{code-cell} ipython3
wort = 'Hallo'
anzahl_zeichen = len(wort)
print(f'Mein Wort {wort} hat {anzahl_zeichen} Zeichen.')
```

#### Definition von einfachen Funktionen

Um selbst eine Funktion zu definieren, benutzen wir das Schlüsselwort `def`.
Danach wählen wir einen Funktionsnamen und hängen an den Funktionsnamen runde
Klammern gefolgt von einem Doppelpunkt. Die Anweisungen, die ausgeführt werden
sollen, sobald die Funktion aufgerufen wird, werden eingerückt.

Als erstes Beispiel einer sehr einfachen Funktion betrachten wir die folgende
Funktion:

```{code-cell} ipython3
def gruesse_ausrichten():
    print('Ich grüße Sie!')
```

Die Funktion hat keine Argumente und keine Rückgabe, sondern gibt einfqach nur
einen Text auf dem Bildschirm aus. Nachdem die Funktion `gruesse_ausrichten()`
so implementiert wurde, können wir sie im Folgenden direkt verwenden.

```{code-cell} ipython3
gruesse_ausrichten()
```

Und natürlich kann man sie in Programmverzweigungen und Schleifen einbauen.

##### Mini-Übung 9

Schreiben Sie eine Funktion, die den Namen `hallihallo` hat und das Wort Hallihallo ausgibt. Testen Sie Ihre Funktion auch.

```{code-cell} ipython3
# Hier Ihr Code:
```

Das folgende Video zeigt, wie Funktionen selbst definiert werden: <https://youtu.be/LQCfN5HS9xI>.

#### Funktionen mit Parametern

Meistens haben Funktionen Argumente, um Eingaben/Input entgegennehmen und
verarbeiten zu können. Das Argument wird bei der Implementierung der Funktion
mit einer Variable eingeführt, wie in dem folgenden Beispiel `name`.

```{code-cell} ipython3
def gruesse_ausrichten_mit_parameter(name):
    print(f'Ich grüße {name}')
```

Der Aufruf einer Funktion ohne passende Argumente führt zu einer Fehlermeldung.
Entfernen Sie das Kommentarzeichen `#` und führen Sie die nachfolgende
Code-Zelle aus:

```{code-cell} ipython3
#gruesse_ausrichten_mit_parameter()
```

Daher müssen wir die modifizierte Funktion nun wie folgt aufrufen:

```{code-cell} ipython3
gruesse_ausrichten_mit_parameter('Bob')
```

Die Funktion `gruesse_ausrichten_mit_parameter()` hat aber keinen Rückgabewert.
Das können wir wie folgt testen:

```{code-cell} ipython3
x = gruesse_ausrichten_mit_parameter('Alice')
type(x)
```

`x` ist vom Typ `NoneType` oder anders ausgedrückt, es hat keinen Datentyp.

Sind Funktionen ohne Rückgabewert sinnvoll? Ja, denn so können Codeblöcke
vereinfacht werden. Sollte in einem Programm Code mehrmals ausgeführt werden,
lohnt es sich, diesen in eine Funktion auszulagern, um diese einfach aufrufen zu
können.

##### Mini-Übung 10

Schreiben Sie eine Funktion mit zwei Parametern, nämlich Vor- und Nachname. Wenn
die Funktion z.B. mit `(Alice, Miller)` aufgerufen wird, soll sie Folgendes auf
dem Bildschirm ausgeben:

```python
Vorname: Alice
Nachname: Miller
```

```{code-cell} ipython3
# Hier Ihr Code:
```

Das folgende Video zeigt, wie Funktionen mit Parametern in Python implementiert
werden: <https://youtu.be/af9ORp1Pty0>

#### Funktionen mit Rückgabewert

In der Regel jedoch haben Funktionen einen Rückgabewert. Schauen wir uns ein
Beispiel an:

```{code-cell} ipython3
def berechne_quadrat(x):
    return x*x

# Aufruf der Funktion
berechne_quadrat(7)
```

Die Rückgabe wird durch das Schlüsselwort `return` erzeugt. Es ist auch möglich,
mehrere Ergebnisse gleichzeitig zurückzugeben. Diese werden einfach nach dem
Schlüsselwort `return` mit Kommas getrennt gelistet.

##### Mini-Übung 11

Schreiben Sie eine Funktion mit zwei Parametern, nämlich den beiden Seitenlängen eines Rechtecks. Lassen Sie die Fläche des Rechtecks berechnen und zurückgeben. Testen Sie Ihr Funktion auch.

```{code-cell} ipython3
# Hier Ihr Code:
```

Video: <https://youtu.be/ehSP-sYoKCY>

### Objektorientierte Programmierung

In den ersten beiden Teilen unseres Crashkurses Python haben wir uns die
Grundlagen der Programmierung erarbeitet:

* Datentypen (Integer, Float, String, List)
* Kontrollstrukturen: for-Schleife
* Funktionen.

In einigen Programmiersprachen wie beispielsweise C hätten wir damit auch alle
Sprachelement kennengelernt. Diese Programmierung nennt man **prozedurale
Programmierung**. Python gehört jedoch zu den objektorientierten
Programmiersprachen, so dass wir uns jetzt noch dem Thema Objektorientierung
widmen.

#### Konzept

Bei der bisherigen prozeduralen Programmierweise haben wir Funktionen und Daten
getrennt. Die Daten werden in Variablen gespeichert. Funktionen funktionieren
nach dem EVA-Prinzip. In der Regel erwartet eine Funktion eine Eingabe von
Daten, verarbeitet diese Daten und gibt Daten zurück.

Angenommen, wir wollten ein Programm zur Verwaltung von Lottoscheinen schreiben.
Zu einem Lottoschein wollen wir Name, Adresse und die angekreuzten Zahlen
speichern. Dann müssten wir mit unserem bisherigen Wissen folgende Variablen pro
Lottoschein einführen:

* vorname
* nachname
* strasse
* postleitzahl
* stadt
* liste_mit_sechs_zahlen

Wenn jetzt viele Spielerinnen und Spieler Lotto spielen wollen, wie gehen wir
jetzt mit den Daten um? Legen wir eine Liste für die Vornamen und eine Liste für
die Nachnamen usw. an? Und wenn jetzt der 17. Eintrag in der Liste mit den sechs
angekreuzten Lottozahlen sechs Richtige hat, suchen wir dann den 17. Eintrag in
der Liste mit den Vornamen und den 17. Eintrag in der Liste mit den Nachnamen
usw.? Umständlich...

Die Idee der objektorientierten Programmierung ist, für solche Szenarien
**Objekte** einzuführen. Ein Objekt fasst verschiedene Eigenschaften wie hier
Vorname, Nachname, Straße, usw. zu einem Objekt Lottoschein zusammen. In der
Informatik wird eine Eigenschaft eines Objekts **Attribut** genannt.

Damit hätten wir erst einmal nur einen neuen Datentyp. Ein Objekt macht noch
mehr aus, denn zu dem neuen Datentyp kommen noch Funktionen dazu, die die
Verwaltung des Objektes erleichtern. Funktionen, die zu einem Objekt gehören,
nennt man **Methoden**.

Video: <https://youtu.be/46yolPy-2VQ>

#### Klassen und Methoden

Im Folgenden sehen Sie, wie ein Objekt in Python definiert wird. Die
Implementierung erfolgt als sogenannte **Klasse**.

```{code-cell} ipython3
class Adresse:
    def __init__(self, strasse, hausnummer, plz, stadt):
        self.strasse = strasse
        self.hausnummer = hausnummer
        self.postleitzahl = plz
        self.stadt = stadt
    
    def print(self):
        print('Straße = ', self.strasse)
        print('Hausnummer = ', self.hausnummer)
        print('Postleitzahl = ', self.postleitzahl)
        print('Stadt = ', self.stadt)
```

Eingeleitet wird eine Klasse mit dem Schlüsselwort `class` und dann dem Namen
der Klasse. Da Klassen Objekte sind, ist es Standard, den ersten Buchstaben des
Klassennamens groß zu schreiben. Um Variablen von Objekten leichter zu
unterscheiden, werden Variablennamen klein geschrieben.

Danach folgt ein Abschnitt namens `def __init__(self):`, in dem die
Eigenschaften der Klasse aufgelistet werden. `init` steht dabei für
initialisieren, also den ersten Zustand, den das Objekt später haben wird.

Wie Sie sehen, können die Eingabe-Parameter der `init()`-Methode die gleichen
Namen tragen wie die Attribute der Klasse, also `self.strasse = strasse`, müssen
sie aber nicht. Das Beispiel `self.postleitzahl = plz` zeigt, dass das Attribut
`self.postleitzahl` einfach den Wert des 4. Parameters bekommt, egal wie der
heißt.

Eine Adresse wird nun folgendermaßen initialisiert:

```{code-cell} ipython3
adresse_fra_uas = Adresse('Nibelungenplatz', 1, 60318, 'Frankfurt am Main')
```

Würden wir nun versuchen, mit `print(adresse_fra_uas)` die Adresse am Bildschirm
ausgeben zu lassen, würden wir eine Fehlermeldung erhalten. Die Funktion
`print()` ist nicht für den Datentyp `Adresse` entwickelt worden. Schließlich
können die Python-Entwickler nicht wissen, welche Klassen Sie entwickeln... Wir
müssen also eine eigene Adressen-print()-Funktion implementieren. Da diese
print()-Funktion nicht allgemeingültig sein kann, sondern nur für die Objekte
`Adresse` funktionieren wird, gehört sie auch folgerichtig zur Klasse selbst.
Sie ist also keine Funktion, sondern eine **Methode**.

Eine Methode wird definiert, indem innerhalb des Anweisungsblocks der Klasse
eine Funktion mit dem Schlüsselwort `def` definiert wird. Der erste Eingabewert
muss zwingend der `self`-Parameter sein. Ansonsten gelten aber die gleichen
Regeln für Methoden wie für Funktionen.

Bleibt nur noch eine Frage? Wie wird nun die Methode ausgeführt? Methoden werden
ausgeführt, indem die Variable hingeschrieben wird, dann ein Punkt gesetzt wird
und dann die Methode mit runden Klammern angefügt wird.

```{code-cell} ipython3
adresse_fra_uas.print()
```

Objektorientierung ist ein sehr detailreiches Thema, das wir in diesem Kapitel
nur streifen konnten. Die folgenden Videos geben einen vertieften Einblick in
die Objektorientierung mit Python.

* <https://youtu.be/XxCZrT7Z3G4>
* <https://youtu.be/CLoK-_qNTnU>
* <https://youtu.be/58IjjwHs_4A>

### Zusammenfassung und Ausblick

In diesem Kapitel haben wir gelernt, selbst Funktionen und Klassen mit Methoden
zu definieren. Oft ist es aber praktischer, Funktionen und Klassen zu nutzen,
die bereits implementiert sind, anstatt das Rad neu zu erfinden. Vor allem bei
der Datenexploration und den maschinellen Lernalgorithmen benutzen wir die
vorgefertigten Funktionsbausteine eher als eigene zu definieren, wie wir in den
nächsten Kapiteln sehen werden.

## Aufgaben

### Aufgabe 2.1

Welcher Datentyp liegt vor? Führen Sie einen Doppelklick auf diese Textzelle
aus und schreiben Sie Ihre Antwort hinter den Pfeil.

* 3 -->
* -3 -->
* 'drei'-->
* 3.3 -->
* 3,3 -->
* 3**3 -->
* 3**(1/3) -->

### Aufgabe 2.2

Schreiben Sie ein Programm, dass die Zahlen von 5 bis 15 mit ihrem Quadrat
ausgibt, also "Das Quadrat von 5 ist 25." usw.

```{code-cell} ipython3
# Hier Ihr Code:
```

### Aufgabe 2.3

Schreiben Sie eine For-Schleife, die die Brüche 1/7, 2/7, 3/7, bis 7/7 als
Fließkommazahl gerundet auf 2 Nachkommastellen ausgibt.

```{code-cell} ipython3
# Hier Ihr Code:
```

### Aufgabe 2.4

Schreiben Sie ein Programm, dass eine Liste von Namen durchgeht und jede Person begrüßt. Wenn beispielsweise die Namen Alice, Bob und Charlie in der Liste stehen, lauten die Begrüßungen:

Hallo, Alice!

Hallo, Bob!

Hallo, Charlie!

```{code-cell} ipython3
# Hier Ihr Code:
```

### Aufgabe 2.5

Schreiben Sie ein Programm, dass das kleine 1x1 in schöner Tabellenform ausgibt,
also

1 x 1 = 1

1 x 2 = 2

usw.

```{code-cell} ipython3
# Hier Ihr Code:
```
