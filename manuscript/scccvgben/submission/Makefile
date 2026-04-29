.PHONY: all cover r1 r2 clean

all: cover r1 r2

cover:; pdflatex -interaction=nonstopmode cover_letter.tex

r1:; pdflatex -interaction=nonstopmode rebuttal_reviewer_1.tex

r2:; pdflatex -interaction=nonstopmode rebuttal_reviewer_2.tex

clean:; rm -f *.aux *.log *.out *.pdf
