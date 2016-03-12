#!/bin/bash
clear
if [[ $# == 1 ]] 
then
  pandoc -s -N --biblatex --listings --chapters --toc-depth=2 --template=template.tex metadata.md content/*.md -o $1.tex
  pdflatex $1.tex
  biber $1
  pdflatex $1.tex
  pdflatex $1.tex
  mv $1* out/ # copy to the out directory
  open out/$1.pdf
else
  echo "you have set a output filename"
fi

