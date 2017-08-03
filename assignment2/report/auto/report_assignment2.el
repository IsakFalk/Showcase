(TeX-add-style-hook
 "report_assignment2"
 (lambda ()
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "mystyle")
   (TeX-add-symbols
    "digitA"
    "digitB"
    "digitC")
   (LaTeX-add-labels
    "fig:LSTM_Xent_1"
    "fig:LSTM_acc_1"
    "fig:GRU_Xent_1"
    "fig:GRU_acc_1"
    "fig:GRU_Xent_2"
    "fig:GRU_acc_2"
    "fig:Xent_hists_1"
    "fig:Xent_hists_b2x2"))
 :latex)

