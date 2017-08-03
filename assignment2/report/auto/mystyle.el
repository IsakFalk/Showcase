(TeX-add-style-hook
 "mystyle"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("hyperref" "hidelinks") ("algorithm" "plain") ("parskip" "parfill")))
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
    "tikz"
    "xfrac"
    "lscape"
    "subfig"
    "bbold"
    "fancyhdr"
    "booktabs"
    "siunitx"
    "hyperref"
    "xcolor"
    "extramarks"
    "amsmath"
    "amsthm"
    "amsfonts"
    "algorithm"
    "enumitem"
    "tabu"
    "bm"
    "amssymb"
    "algpseudocode"
    "parskip"
    "listings"
    "physics"
    "verbatim"
    "color")
   (TeX-add-symbols
    '("E" 1)
    '("pderiv" 2)
    '("deriv" 1)
    '("alg" 1)
    '("mbf" 1)
    "bigCI"
    "nbigCI"
    "CI"
    "nCI"
    "argmin"
    "argmax"
    "sign"
    "hmwkTitle"
    "hmwkDueDate"
    "hmwkClass"
    "hmwkClassTime"
    "hmwkClassInstructor"
    "hmwkAuthorName"
    "hmwkAuthorSurname"
    "githubAccnt"
    "dx"
    "solution"
    "Var"
    "Cov"
    "Bias"))
 :latex)

