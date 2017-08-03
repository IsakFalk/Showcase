(TeX-add-style-hook
 "Assignment1"
 (lambda ()
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "mystyle")
   (LaTeX-add-labels
    "fig:err_1a"
    "fig:conf_1a"
    "fig:err_1b"
    "fig:conf_1b"
    "fig:err_1c"
    "fig:conf_1c"
    "fig:err_1d"
    "fig:conf_1d"
    "fig:err_2a"
    "fig:conf_2a"
    "fig:err_2b"
    "fig:conf_2b"
    "fig:err_2c"
    "fig:conf_2c"))
 :latex)

