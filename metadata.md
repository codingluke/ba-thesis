---
title: Bachelorarbeit
author: Lukas Hodel
documentclass: scrbook
classoption:
   - 12pt
   - a4paper
   - oneside
   - bibliography=totoc
fontsize: 12pt
spacing: onehalfspacing
geometry:
    - left=3cm
    - right=2.5cm
    - top=3.5cm
    - bottom=3cm
    - headsep=1.5cm
header-includes:
    - \usepackage{lipsum}
    - \usepackage{tocbibind}
    - \usepackage{tocloft}
    - \renewcommand{\contentsname}{Table of Contents}
    - \renewcommand{\listfigurename}{Figures}
    - \renewcommand{\listtablename}{Tables}
biblio-files: library.bib
biblio-style: alphabetic
biblio-title: Literaturverzeichnis
toc: true
lof: true
lot: true
include-before:
    - include/titlepage.tex
include-after:
    - include/eigenstaendigkeitserklaerung.tex
chapterheadstartvskip: -1cm
chapterheadendvskip: 1cm
scrlayer-scrpage:
  - \clearpairofpagestyles
  - \cfoot[\pagemark]{\pagemark}
  - \lehead{\headmark}
  - \rohead{\headmark}
  - \pagestyle{scrheadings}
---
