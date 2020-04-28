%%
%% This is file `arlticle.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% arlticle.dtx  (with options: `class')
%% 
%% IMPORTANT NOTICE:
%% 
%% For the copyright see the source file.
%% 
%% Any modified versions of this file must be renamed
%% with new filenames distinct from arlticle.cls.
%% 
%% For distribution of the original source see the terms
%% for copying and modification in the file arlticle.dtx.
%% 
%% This generated file may be distributed as long as the
%% original source files, as listed above, are part of the
%% same distribution. (The sources need not necessarily be
%% in the same archive or directory.)
 % \iffalse
%%
%% This is the file `arlticle.cls',
%% composed by Steven B Segletes.
%%
%% Based on LaTeX's standard article.cls and an arltr class file obtained
%% from Brian Krzewinski via Stephen Schraml...
%% this class file was modified quite a bit by Steven Segletes (March 2006).
%% It creates a LaTeX document class for the ARL-TR technical report series.
%% Documentation in the form of a dtx file was added April-May 2006.
%% Changes have been ongoing since its inception, at least through
%% NOV 2016.
%%
%% Please treat this document as limited distribution.
%%
%% Steven may be reached at 278-6010 for questions comments and updates.
%%
\ProvidesClass{arlticle}
[2019/08/22 v2.56
       ARL Technical Report Document Class File]
\NeedsTeXFormat{LaTeX2e}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%
%%
%% The original source files were:
%%
%% classes.dtx  (with options: `article')
%%
%% This is a generated file.
%%
%% Copyright 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004
%% The LaTeX3 Project and any individual authors listed elsewhere
%% in this file.
%%
%% This file was generated from file(s) of the LaTeX base system.
%% --------------------------------------------------------------
%%
%% It may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.3 or later is part of all distributions of LaTeX
%% version 2003/12/01 or later.
%%
%% This file has the LPPL maintenance status "maintained".
%%
%% This file may only be distributed together with a copy of the LaTeX
%% base system. You may however distribute the LaTeX base system without
%% such generated files.
%%
%% The list of all files belonging to the LaTeX base distribution is
%% given in the file `manifest.txt'. See also `legal.txt' for additional
%% information.
%%
%% The list of derived (unpacked) files belonging to the distribution
%% and covered by LPPL is defined by the unpacking scripts (with
%% extension .ins) which are part of the distribution.
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}

\usepackage{needspace}
\usepackage{setspace}
\usepackage[table]{xcolor}
\usepackage{ifthen}
\usepackage{pgf}
\usepackage{stackengine}
%%\usepackage{stringstrings}% KEEP AN EYE OUT IF THINGS BREAK
\usepackage{listofitems}
\usepackage{everypage}
\usepackage[T1]{fontenc}
\usepackage{changepage}% TO DETECT EVEN/ODD-NESS OF PAGE #
%%  \usepackage{opensans}% PART OF ARLtimes
%%  \usepackage{lmodern}% PART OF ARLtimes
  \usepackage{ARLtimes}

%% LOAD IN OPTIONAL FONT PACKAGES.
%%\RequirePackage{palatino}% NO LONGER ACTIVE!!
\RequirePackage{graphicx}
\usepackage[letterpaper,portrait]{geometry}

\def\preset@ARLmargins{\AtBeginDocument{\@ARLmargins}}

\def\@ARLmargins{%
  \newgeometry{hoffset=0pt,left=1.5in,right=1.5in,top=1in,bottom=1in}%
}

\preset@ARLmargins

\long\def \protected@iwrite#1#2#3{%
     \begingroup
     \let\thepage\relax
     #2%
     \let\protect\@unexpandable@protect
     \edef\reserved@a{\immediate\write#1{#3}}%
     \reserved@a
     \endgroup
     \if@nobreak\ifvmode\nobreak\fi\fi
    }
\RequirePackage[nospace, noadjust, super]{cite} %Makes 5,6,7 as 5-7 in cites

%% Do \cite command on line.
%%
\def\@citexp[#1]#2{\@citep{\@cite@n{#2}}{#1}}

\def\@citep#1#2{\leavevmode \cite@adjust
  \citeleftp{#1\if@tempswa\@safe@activesfalse\citemid{#2}\fi
  \spacefactor\@m % punctuation in note doesn't affect outside
  }\citerightp
 \@restore@auxhandle}
%%%%%%%%%%%%%%%
\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagefalse
\if@compatibility\else
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a5paper}
   {\setlength\paperheight {210mm}%
    \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
   {\setlength\paperheight {250mm}%
    \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
   {\setlength\paperheight {11in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{legalpaper}
   {\setlength\paperheight {14in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
   {\setlength\paperheight {10.5in}%
    \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\fi
\if@compatibility
  \renewcommand\@ptsize{0}
\else
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\fi
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}
\if@compatibility\else
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\fi
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\if@compatibility\else
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\fi
\DeclareOption{titlepage}{\@titlepagetrue}
\if@compatibility\else
\DeclareOption{notitlepage}{\@titlepagefalse}
\fi
\if@compatibility\else
\DeclareOption{onecolumn}{\@twocolumnfalse}
\fi
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}%
}
%%EXECUTE THE DEFAULT OPTIONS TO REFLECT THE ARL STANDARD STYLE
\ExecuteOptions{letterpaper,12pt,twoside,onecolumn,final}
\ProcessOptions
\input{size1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}
\if@twoside
  \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\hfil\slshape\leftmark}%
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
    \def\sectionmark##1{%
      \markboth {\MakeUppercase{%
        \ifnum \c@secnumdepth >\z@
          \thesection\quad
        \fi
        ##1}}{}}%
    \def\subsectionmark##1{%
      \markright {%
        \ifnum \c@secnumdepth >\@ne
          \thesubsection\quad
        \fi
        ##1}}}
\else
  \def\ps@headings{%
    \let\@oddfoot\@empty
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\markboth
    \def\sectionmark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
          \thesection\quad
        \fi
        ##1}}}}
\fi
\def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{\thepage\hfil\slshape\leftmark}%
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\@gobbletwo
    \let\sectionmark\@gobble
    \let\subsectionmark\@gobble
    }
  \if@titlepage
  \newcommand\maketitle{\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \let \footnote \thanks
  \null\vfil
  \vskip 60\p@
  \begin{center}%
    {\LARGE \@title \par}%
    \vskip 3em%
    {\large
     \lineskip .75em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
      \vskip 1.5em%
    {\large \@date \par}%       % Set date in \large size.
  \end{center}\par
  \@thanks
  \vfil\null
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\else
\newcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em}
\fi
\setcounter{secnumdepth}{4}
\newcounter {part}
\newcounter {section}
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thesection {\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection.\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
%% SECTIONING \hskip'S BEFORE SECTIONING TITLES
\def\sectionkern      {1em}
\def\subsectionkern   {1em}
\def\subsubsectionkern{1em}
\def\paragraphkern    {1em}
%% SUFFIX ADDED TO \the SECTIONING MACROS
\def\sectionsuffix      {.}
\def\subsectionsuffix   {}
\def\subsubsectionsuffix{}
\def\paragraphsuffix    {}
%% SECTIONING LABEL: \the[section]\[section]suffix\hskip\[section]kern
\renewcommand\@seccntformat[1]{\csname the#1\endcsname%
  \csname#1suffix\endcsname\hskip\csname#1kern\endcsname\relax}

\newcommand\part{%
   \if@noskipsec \leavevmode \fi
   \par
   \addvspace{4ex}%
   \@afterindentfalse
   \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >\m@ne
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >\m@ne
       \Large\bfseries \partname\nobreakspace\thepart
       \par\nobreak
     \fi
     \huge \bfseries #2%
     \markboth{}{}\par}%
    \nobreak
    \vskip 3ex
    \@afterheading}
\def\@spart#1{%
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \huge \bfseries #1\par}%
     \nobreak
     \vskip 3ex
     \@afterheading}

\newlength\Letter@Space

\def\useOpenSans{%
  \renewcommand\sfdefault{\opensansfamily}%
  \def\@magfA{1.14}%
  \def\@magfB{1}%
  \def\boilerplatesize{\fontsize{9pt}{12pt}\selectfont}
  \def\semi@normalsize{\fontsize{12pt}{16pt}\selectfont}
  \def\semi@large{\fontsize{14pt}{20pt}\selectfont}
  \def\@large{\fontsize{16pt}{20pt}\selectfont}
  \def\@Huge{\fontsize{22pt}{28pt}\selectfont}
  \let\Level@onesize\normalsize%
  \let\Level@twosize\small%
  \let\Level@threesize\Level@twosize%
  \let\Level@foursize\footnotesize%
  \Letter@Space=0.2pt\relax%
}
\def\useLMss{%
  \renewcommand\sfdefault{lmss}%
  \def\@magfA{1.07}%
  \def\@magfB{1.05}%
  \def\boilerplatesize{\fontsize{9.5pt}{12pt}\selectfont}
  \def\semi@normalsize{\fontsize{12pt}{16pt}\selectfont}
  \def\semi@large{\fontsize{15pt}{20pt}\selectfont}
  \def\@large{\fontsize{17pt}{20pt}\selectfont}
  \def\@Huge{\fontsize{24pt}{28pt}\selectfont}
  \let\Level@onesize\normalsize%
  \let\Level@twosize\normalsize%
  \def\Level@twosize{\fontsize{11.2pt}{15pt}\selectfont}
  \let\Level@threesize\Level@twosize%
  \let\Level@foursize\small%
  \Letter@Space=0.4pt\relax%
}
\def\useCMss{%
  \renewcommand\sfdefault{cmss}%
  \def\@magfA{1.10}%
  \def\@magfB{1.07}%
  \def\boilerplatesize{\fontsize{10pt}{12pt}\selectfont}
  \def\semi@normalsize{\fontsize{12pt}{16pt}\selectfont}
  \def\semi@large{\fontsize{14pt}{20pt}\selectfont}
  \def\@large{\fontsize{17pt}{20pt}\selectfont}
  \def\@Huge{\fontsize{25pt}{28pt}\selectfont}
  \def\Level@onesize{\fontsize{13pt}{15pt}\selectfont}%
  \def\Level@twosize{\fontsize{12pt}{14pt}\selectfont}%
  \let\Level@threesize\Level@twosize%
  \let\Level@foursize\small%
  \Letter@Space=.4pt\relax%
}
\def\useCalibri{%
  \renewcommand\sfdefault{Calibri}%
  \def\@magfA{1.17}%
  \def\@magfB{1.09}%
  \def\boilerplatesize{\fontsize{10pt}{12pt}\selectfont}
  \def\semi@normalsize{\fontsize{13pt}{16pt}\selectfont}
  \def\semi@large{\fontsize{16pt}{20pt}\selectfont}
  \def\@large{\fontsize{18pt}{20pt}\selectfont}
  \def\@Huge{\fontsize{26pt}{28pt}\selectfont}
  \let\Level@onesize\large%
  \let\Level@twosize\normalsize%
  \let\Level@threesize\Level@twosize%
  \let\Level@foursize\small%
  \Letter@Space=0.5pt\relax%
}
\useOpenSans

%%MODIFY SECTION HEADINGS AS 'large'; RENAME ROUTINE AS \origsection
\newcommand\origsection{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\sffamily\Level@onesize\bfseries\raggedright}}
%%MODIFY SUBSECTION HEADINGS RETAINING 'normalsize' BUT DIFFERENT LINE SPACING
\newcommand\origsubsection{\@startsection{subsection}{2}{\z@}%
                                     {0.0ex\@plus -1ex \@minus -.2ex}%
                                     {0.01ex \@plus .2ex}%
                                     {\sffamily\Level@twosize\bfseries\raggedright}}
%%MODIFY SUBSUBSECTION TO OBTAIN DIFFERENT LINE SPACING
\newcommand\origsubsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {0.0ex\@plus -1ex \@minus -.2ex}%
                                     {0.01ex \@plus .2ex}%
                                     {\sffamily\Level@threesize\bfseries\sloppypar}}
%%MODIFY PARAGRAPH TO OBTAIN DIFFERENT LOOK ALTOGETHER
\newcommand\origparagraph{\@startsection{paragraph}{4}{0pt}% xtra horizontal indent header
                       {0pt \@plus-1ex \@minus.2ex}% additional vertical space before
                       {.01ex \@plus .2ex \@minus .2ex}% negative keeps text on same line
                       {\sffamily\Level@foursize}}% header font size same as subsubsec
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\sffamily\normalsize\bfseries}}
\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
\if@compatibility
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\z@
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\@ne
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        \setcounter{page}\@ne
     \fi
    }
\fi
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\renewcommand \theequation {\@arabic\c@equation}
\newcounter{figure}
\renewcommand \thefigure {\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename\nobreakspace\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}
\renewcommand\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename\nobreakspace\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newenvironment{ARLtable}[1][tbp]
               {\begin{table}[#1]\centering\footnotesize}
               {\end{table}}
\newenvironment{ARLfigure}[1][tbp]
               {\begin{figure}[#1]\centering\footnotesize}
               {\end{figure}}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
%% RESET THE ABOVE- AND BELOW-CAPTIONSKIP PARAMETERS FROM 10,0 TO 12,9.
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{9\p@}
%% PREPARE FOR SINGLE FIGURE OR TABLE, IF NEEDED
\newcommand\useSingleTable{\def\Single@Table{T}}% MUST BE INVOKED BEFORE lot
\newcommand\useSingleFigure{\def\Single@Figure{T}}% MUST BE INVOKED BEFORE lof
\def\Single@Table{F}
\def\Single@Figure{F}
%% DEFINE \makecaption SO THAT PERIOD IS NOT PLACED AFTER FIGURE/TABLE NUMBER,
%% BUT IS PLACED
%% AT END OF CAPTION TEXT, AND MAKE THE FONT \small.  ALSO LEFT JUSTIFIED.
%% NOTE THAT THIS DEFINITION IS LATER SUPERCEDED BY THE boxhandler STYLE FILE
\long\def\@makecaption#1#2{%
  \def\multi@caption{#1\ \ }%
  \IS@nextbyte[q]{T}{#1}%
  \if T\theresult\if T\Single@Table\def\multi@caption{Table\ \ }\fi\fi%
  \IS@nextbyte[q]{F}{#1}%
  \if T\theresult\if T\Single@Figure\def\multi@caption{Figure\ \ }\fi\fi%
  \bfseries%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\footnotesize\multi@caption#2}%
  \ifdim \wd\@tempboxa >\hsize
%% NO HANGING INDENT
    \footnotesize \multi@caption#2\par
%% HANGING INDENT
%%    \sbox\@tempboxa{\footnotesize\multi@caption}%
%%    \footnotesize #1\ \ \begin{minipage}[t]{\dimexpr\textwidth-\wd\@tempboxa}#2
%%      \end{minipage}\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

\def\Primary@SignalChar{@}
\def\Secondary@SignalChar{`}

\newcommand\IS@nextbyte[3][v]{%
  \let\@SignalChar\Primary@SignalChar%
  \edef\@x{\if #2#3\else\@SignalChar\fi}%
  \edef\@x{\if \@SignalChar\@x F\else T\fi}%
  \let\@SignalChar\Secondary@SignalChar%
  \edef\@y{\if #2#3\else\@SignalChar\fi}%
  \edef\@y{\if \@SignalChar\@y F\else T\fi}%
  \ifthenelse{\equal{\@x\@y}{\@y\@x}}
    {\edef\theresult{\@x}}%
    {\edef\theresult{T}}%
  \if q#1\else\theresult\fi
}

\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em plus1fil}
%% Change \@dotsep from 4.5 to 1.0
\newcommand\@dotsep{1.0}
\setcounter{tocdepth}{3}% TOC INCLUDES HEADINGS DOWN TO LEVEL 3 (SUBSUBSECTIONS)
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty\@secpenalty
    \addvspace{2.25em \@plus\p@}%
    \setlength\@tempdima{3em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
       \if@compatibility
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
      \fi
    \endgroup
  \fi}
\newcommand*\l@section[2]{%
  \ifnum \c@tocdepth >\z@
    \addpenalty\@secpenalty
    \addvspace{1.0em \@plus\p@}%
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
    \endgroup
  \fi}
\newcommand*\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{7.0em}{4.1em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{10em}{5em}}
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\let\l@table\l@figure
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \twocolumn[\section*{\indexname}]%
                \@mkboth{\MakeUppercase\indexname}%
                        {\MakeUppercase\indexname}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \columnseprule \z@
                \columnsep 35\p@
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \vfill%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}

\def\ClassificationLabel{}
\newcommand\UseClassMarkings{\def\ClassificationLabel{(U)~}}

\newcommand\contentsname{\ClassificationLabel%
  \spaceouthelpAcopy Contents \relax}% SPACE BEFORE \relax IMPORTANT!!
\newcommand\listfigurename{\ClassificationLabel%
  \spaceouthelpAcopy List of Figures \relax}% SPACE BEFORE \relax IMPORTANT!!
\newcommand\listtablename{\ClassificationLabel%
  \spaceouthelpAcopy List of Tables \relax}% SPACE BEFORE \relax IMPORTANT!!
\newcommand\refname{\ClassificationLabel References}
\newcommand\indexname{Index}
\newcommand\figurename{Fig.}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\appendixname{\ClassificationLabel Appendix}
\newcommand\abstractname{Abstract}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{plain}
\pagenumbering{arabic}
\if@twoside
\else
  \raggedbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi
%%
%% NOW ADD A BUNCH OF ARL-SPECIFIC STUFF
%%
\raggedbottom
%% SET PARAGRAPH SPACING PARAMETERS
\parindent 0pt
\parskip 1em
%% SET LINE FLUSH TYPE.
%% FORMERLY \raggedright IN OLD ARL STYLE.  NOW, IT JUST PASSES THROUGH THE
%% OPTIONAL ARGUMENT. FOR TR2ARTICLE STYLE, OPTIONAL ARGUMENT IS UTILIZED.
\newcommand\setpreferredlineflush[1][]{#1}
%%
%% SET UP CODE TO USE PAGE STAMPS AT TOP AND BOTTOM OF PAGE.  DEFAULT
%% STAMP IS "DRAFT", BUT IT CAN BE USED FOR CLASSIFICATION MARKINGS.,
%% TOO.
\def\PageStampFlag{F}
\def\thePageStamp{}
\def\thePageStampTOP{\thePageStamp}
\def\EmptyStyle{empty}
\def\PlainStyle{plain}
\def\PageStampColor{black}

\newcommand{\ps@pagestamp}{
  \renewcommand{\@oddhead}{\sffamily\color{\PageStampColor}%
                           \hfil\textbf{\Large\thePageStampTOP}\hfil}
  \renewcommand{\@evenhead}{\@oddhead}
  \renewcommand{\@evenfoot}{\sffamily\hfil
                            \begin{tabular}{c}
                            \textrm{\thepage}\\
                            \color{\PageStampColor}
                            \textbf{\Large\thePageStamp}
                            \end{tabular}
                            \hfil}
  \renewcommand{\@oddfoot}{\@evenfoot}
}

\newcommand{\ps@emptypagestamp}{
  \renewcommand{\@oddhead}{\sffamily\color{\PageStampColor}%
                             \hfil\textbf{\Large\thePageStampTOP}\hfil}
  \renewcommand{\@evenhead}{\@oddhead}
  \renewcommand{\@evenfoot}{\sffamily\hfil
                            \begin{tabular}{c}
                            ~\\
                            \color{\PageStampColor}
                            \textbf{\Large\thePageStamp}
                            \end{tabular}
                            \hfil}
  \renewcommand{\@oddfoot}{\@evenfoot}
}

\def\FOIA@exemption{}

\newcommand\FOIAexemptions[1]{%
  \ifthenelse{\equal{\@gobble#1}{}}
  {\def\FOIA@exemption{This document contains information EXEMPT FROM
        MANDATORY DISCLOSURE under the Freedom of Information Act.
        Exemption category #1 applies.}}
  {\def\FOIA@exemption{This document contains information EXEMPT FROM
        MANDATORY DISCLOSURE under the Freedom of Information Act.
        Exemption categories #1 apply.}}
}

%% NEXT TWO COMMANDS DEPRECATED
\newcommand\FOUOunderset[2][3pt]{\stackunder[#1]{#2}{\sffamily\bfseries FOUO}}

\newcommand\FOUOincludegraphics[2][]{\FOUOunderset{\includegraphics[#1]{#2}}}
%%

\newcommand\setportionmark[1]{\gdef\g@pmarking{#1}}
\setportionmark{UNCLASSIFIED}

\newcommand\portionmark[1][\relax]{%
  \ifx\relax#1\relax\def\@pmarking{\g@pmarking}\else\def\@pmarking{#1}\fi%
  \@portionmark%
}

\newcommand\@portionmark[2][3pt]{%
  \stackengine{#1}{%
    \stackengine{#1}{#2}{\sffamily\bfseries\@pmarking}{U}{r}{F}{F}{S}%
  }{\sffamily\bfseries\@pmarking}{O}{l}{F}{F}{S}%
}
%%

\newcommand\FOUOPageStamp[1][FOR OFFICIAL USE ONLY]{%
  \PageStamp[#1]%
  \def\thePageStampTOP{}%
}

\newcommand\PageStamp[1][DRAFT]{
  \def\PageStampFlag{T}
  \setlength\textheight{9in} %textheight+headheight+headsep=9in
  \def\thePageStamp{#1}
  \setlength\topmargin{-1.0in}
  \setlength\headheight{0.6in}
  \setlength\headsep{0.4in}
  \setlength\footskip{38pt}
  \def\PlainStyle{pagestamp}
  \def\EmptyStyle{emptypagestamp}
  \pagestyle{\PlainStyle}
}

\newcommand\StopPageStamp[0]{
  \def\PageStampFlag{F}
  \setlength\topmargin{0in}
  \setlength\headheight{0pt}
  \setlength\headsep{0pt}
  \setlength\textheight {9.0in}
  \setlength\footskip{30pt}
  \def\PlainStyle{plain}
  \def\EmptyStyle{empty}
  \pagestyle{\PlainStyle}
}

%% Define the title page for an ARL report
%%
\def\@ARLaddress{Aberdeen Proving Ground, MD 21005-5066}
\def\@arlrptno{ARL-TR-XXXX}
\def\@pubdate{DRAFT: \today}
\def\@arltitle{Report Title:\\ Multiple Lines if Needed}
\def\@allauthors{Author1, Author2, ..., and Authorx}
\def\@authorsA{Author(s) from lead organization (e.g., directorate)}
\def\@organizationA{\DirectoratePlaceholder\myCCDCorg}
\def\@authorsB{}
\def\@organizationB{}
\def\@authorsC{}
\def\@organizationC{}
\def\@authorsD{}
\def\@organizationD{}
\def\@authorsE{}
\def\@organizationE{}
\def\@titlepagenote{}
\def\@distribution{Proper distribution statement goes here.
                   Use $\backslash$\textsf{distcodes[ofc symbol]\{dstrb.
                   letter\}\{reason number\}\{export\}} in preamble.}
\def\@distributionreason{}
\def\@ecwarningtext{}
\def\@destructionnoticeA{Destroy this report when it is no longer needed.
                        Do not return it to the originator.}
\def\@destructionnoticeB{DESTRUCTION NOTICE---For classified documents,
  follow the procedures in DOD~5220.22-M, National Industrial Security
  Program Operating Manual, Chapter~5, Section~7, or DOD~5200.1-R,
  Information Security Program Regulation, C6.7.  For unclassified,
  limited documents, destroy by any method that will prevent disclosure
  of contents or reconstruction of the document.}
\def\@destructionnotice{\@destructionnoticeA}
\newcommand\@otherrequests[1]{Other requests for this document shall be
                             referred to Director, \myCCDCorgLONG,
                             ATTN: #1, \@ARLaddress.}
\def\@otherreasons{}
\def\@primaryreason{}

\newcommand\arladdress[1]{\def\@ARLaddress{#1}}
\newcommand\arlzipcode[1]{\def\@ARLaddress{Aberdeen Proving Ground, MD #1}}
\newcommand\arlrptno[1]{\def\@arlrptno{#1}%
  \@autodetectSFitemTWO x#1-xx-x\relax%
}
\newcommand\pubdate[1]{\edef\@pubdate{#1}%
  \provide@shortpubdate#1 \unskip \unskip\relax}
\newcommand\arltitle[1]{\def\@arltitle{#1}}
\newcommand\allauthors[1]{\def\@allauthors{#1}}
\newcommand\authorsA[1]{\def\@authorsA{#1}}
\newcommand\organizationA[1]{\def\@organizationA{#1}}
\newcommand\authorsB[1]{\def\@authorsB{#1}}
\newcommand\organizationB[1]{\def\@organizationB{#1}}
\newcommand\authorsC[1]{\def\@authorsC{#1}}
\newcommand\organizationC[1]{\def\@organizationC{#1}}
\newcommand\authorsD[1]{\def\@authorsD{#1}}
\newcommand\organizationD[1]{\def\@organizationD{#1}}
\newcommand\authorsE[1]{\def\@authorsE{#1}}
\newcommand\organizationE[1]{\def\@organizationE{#1}}
\newcommand\titlepagenote[1]{\def\@titlepagenote{#1}}
\newcommand\distribution[1]{\def\@distribution{#1}}
\newcommand\abbrevdistribution[1]{\def\@abbrevdistribution{#1}}
\newcommand\postreason[1]{\@otherreasons (\@pubdate). \@otherrequests{#1}}
\newcommand\AddDistributionReason[1]{%
  \ifthenelse{\equal{#1}{X}}
    {\@AddDistReason{\@otherreasons}{999}}%
    {\@AddDistReason{\@otherreasons}{#1}}%
}
\abbrevdistribution{A distribution statement (and abbreviated version)
  must be specified, preferably via \textbackslash distcodes.}

\def\@shortpubdate{Undefined }
\def\provide@shortpubdate#1 #2\relax{%
 \gdef\@shortpubdate{#1 #2}%
  \ifthenelse{\equal{#1}{January}}{\gdef\@shortpubdate{Jan #2}}{%
  \ifthenelse{\equal{#1}{February}}{\gdef\@shortpubdate{Feb #2}}{%
  \ifthenelse{\equal{#1}{March}}{\gdef\@shortpubdate{Mar #2}}{%
  \ifthenelse{\equal{#1}{April}}{\gdef\@shortpubdate{Apr #2}}{%
  \ifthenelse{\equal{#1}{May}}{\gdef\@shortpubdate{May #2}}{%
  \ifthenelse{\equal{#1}{June}}{\gdef\@shortpubdate{June #2}}{%
  \ifthenelse{\equal{#1}{July}}{\gdef\@shortpubdate{July #2}}{%
  \ifthenelse{\equal{#1}{August}}{\gdef\@shortpubdate{Aug #2}}{%
  \ifthenelse{\equal{#1}{September}}{\gdef\@shortpubdate{Sep #2}}{%
  \ifthenelse{\equal{#1}{October}}{\gdef\@shortpubdate{Oct #2}}{%
  \ifthenelse{\equal{#1}{November}}{\gdef\@shortpubdate{Nov #2}}{%
  \ifthenelse{\equal{#1}{December}}{\gdef\@shortpubdate{Dec #2}}{%
}}}}}}}}}}}}}

\newcommand\@AddDistReason[2]{%
  \def\tmp@cmd{\edef#1}%
  \setcounter{@distreason}{0#2}%
  \ifthenelse{\the@distreason=0}%
    {\tmp@cmd{#1\unskip; public release\ }}%
    {%
  \ifthenelse{\the@distreason=1}%
    {\tmp@cmd{#1\unskip; administrative or operational use\ }}%
    {%
  \ifthenelse{\the@distreason=2}%
    {\tmp@cmd{#1\unskip; contractor performance evaluation\ }}%
    {%
  \ifthenelse{\the@distreason=3}%
    {\tmp@cmd{#1\unskip; critical technology\ }}%
    {%
  \ifthenelse{\the@distreason=4}%
    {\tmp@cmd{#1\unskip; direct military support\ }}%
    {%
  \ifthenelse{\the@distreason=5}%
    {\tmp@cmd{#1\unskip; export controlled\ }}%
    {%
  \ifthenelse{\the@distreason=6}%
    {\tmp@cmd{#1\unskip; foreign government information\ }}%
    {%
  \ifthenelse{\the@distreason=7}%
    {\tmp@cmd{#1\unskip; operations security\ }}%
    {%
  \ifthenelse{\the@distreason=8}%
    {\tmp@cmd{#1\unskip; premature dissemination\ }}%
    {%
  \ifthenelse{\the@distreason=9}%
    {\tmp@cmd{#1\unskip; proprietary information\ }}%
    {%
  \ifthenelse{\the@distreason=10}%
    {\tmp@cmd{#1\unskip; test and evaluation\ }}%
    {%
  \ifthenelse{\the@distreason=11}%
    {\tmp@cmd{#1\unskip; software documentation\ }}%
    {%
  \ifthenelse{\the@distreason=12}%
    {\tmp@cmd{#1\unskip; specific authority\ }}%
    {%
  \ifthenelse{\the@distreason=13}%
    {\tmp@cmd{#1\unskip; vulnerability information\ }}%
    {%
  \ifthenelse{\the@distreason=999}%
    {\tmp@cmd{#1\unskip; export controlled\ }}%
    {%
  }}}}}}}}}}}}}}}%
}

\def\SFitemSEVENTEENvalue{???}
\def\OfficeSymbol{FCDD-xxx-xx}%

\newcounter{@distreason}
\newcommand\distcodes[4][\OfficeSymbol]{%
  \xdef\OfficeSymbol{#1}%
  \def\thedistcode{#2}%
  \@AddDistReason{\@primaryreason}{#3}%
  \def\@distributionreason{\@primaryreason\postreason{#1}}%
  \def\@abbrevdistributionreason{\@primaryreason\@otherreasons (\@pubdate).}
%%
  \if A#2
    \distribution{Approved for public release; distribution is unlimited.}
    \abbrevdistribution{Approved for public release; distribution is unlimited.}
    \def\@destructionnotice{\@destructionnoticeA}
    \def\SFitemSEVENTEENvalue{UU}
  \else
%%  any distribution outside of A gets destruction notice B
    \def\@destructionnotice{\@destructionnoticeB}
    \def\SFitemSEVENTEENvalue{SAR}
  \fi
  \if B#2
    \distribution{Distribution authorized to US Government agencies
                  only\@distributionreason}
    \abbrevdistribution{Distribution authorized to US Government agencies
                  only\@abbrevdistributionreason}
  \fi
  \if C#2
    \distribution{Distribution authorized to US Government agencies and
                  their contractors\@distributionreason}
    \abbrevdistribution{Distribution authorized to US Government agencies and
                  their contractors\@abbrevdistributionreason}
  \fi
  \if D#2
    \distribution{Distribution authorized to DOD and DOD contractors
                  only\@distributionreason}
    \abbrevdistribution{Distribution authorized to DOD and DOD contractors
                  only\@abbrevdistributionreason}
  \fi
  \if E#2
    \distribution{Distribution authorized to DOD components only%
                  \@distributionreason}
    \abbrevdistribution{Distribution authorized to DOD components only%
                  \@abbrevdistributionreason}
  \fi
  \if F#2
    \distribution{Further dissemination only as directed by Director,
                  \myCCDCorgLONG, ATTN: #1, \@ARLaddress{}
                  (\@pubdate) or higher DOD authority.}
    \abbrevdistribution{Further dissemination only as directed by Director,
                  \myCCDCorgLONG.}
  \fi
  \if X#2
    \distribution{**Distcode ``X'' IS NO LONGER A VALID CHOICE**}
    \abbrevdistribution{**Distcode ``X'' IS NO LONGER A VALID CHOICE**}
  \fi
%%
  \ifthenelse{\equal{#4}{}}
    {\def\@ecwarningtext{}}
    {%
     \if A#2
       \def\@ecwarningtext{I'm sorry, you can not list a document with
                      export-controlled information as Distribution A.
                      Please revise your arguments to the command
                      \textsf{distcodes}.}
     \else
       \def\@ecwarningtext{Warning---This document contains technical data
                      whose export is restricted by the Arms Export Control
                      Act (Title~22, U.S.C., Sec.~2751 et seq.)~or the Export
                      Administration Act of 1979, as amended, Title~50,
                      U.S.C. App~2401 et seq.  Violations of these export
                      laws are subject to severe criminal penalties.
                      Disseminate in accordance with provisions of DOD
                      Directive 5230.25.}
     \fi
    }%
  \abbrevdistribution{}% NULLIFY ABBREVIATED DISTRIBUTION DEFAULT (5/22/19)
}

\def\downgradetext{}

\newcommand\downgrading[5]{%
 \def\downgradetext{%
  \begin{tabular}{@{}ll}
   \textbf{CLASSIFIED BY:} & #1, #2\\%
   \textbf{DERIVED FROM:}  & #3, dated #4\\%
   \textbf{DECLASSIFY ON:} & #5\\%
  \end{tabular}%
 }%
  \def\downgradeARGa{#1}%
  \def\downgradeARGb{#2}%
  \def\downgradeARGc{#3}%
  \def\downgradeARGd{#4}%
  \def\downgradeARGe{#5}%
  \def\@abbrevdistribution{}%
}

\definecolor{ReportColor}{cmyk}{0.0132,0.0821,0.7278,.2}%
\definecolor{LoInkReportColor}{rgb}{1,1,1}%

\newcommand\theboilerplate{%
    \def\blank@entry{}%
    \boilerplatesize
    \vspace{\baselineskip}\par
    \ifx\blank@entry\FOIA@exemption\else
      \FOIA@exemption\par
      \vspace{10pt}\par
    \fi
    \ifx\blank@entry\downgradetext\else
      \downgradetext\par
      \vspace{10pt}\par
    \fi
    \ifx\blank@entry\@ecwarningtext\else
      \@ecwarningtext\par
      \vspace{10pt}\par
    \fi
    \@distribution
}

%% ARL DEFAULT VALUES
\gdef\myCCDCorgSHORT{CCDC ARL}
\gdef\myCCDCorg{CCDC Army Research Laboratory}
\gdef\myCCDCorgLONG{US Army Combat Capabilities Development Command
                    Army Research Laboratory}
\gdef\DirectoratePlaceholder{Directorate Name, }
\gdef\LogoImage{"CCDC_ARL only_RGB_Positive"}
\gdef\CoverImage{"Green Cover-2019-02-ARL"}

\newcommand\ARLcover[2][]%
{%
  \ifx S#2\relax% SLAD VALUES
     \gdef\myCCDCorgSHORT{CCDC DAC}
     \gdef\myCCDCorg{CCDC Data \& Analysis Center}
     \gdef\myCCDCorgLONG{US Army Combat Capabilities Development Command
                         Data \& Analysis Center}
     \gdef\DirectoratePlaceholder{}
     \gdef\LogoImage{"CCDC_DAC only_RGB_Positive"}
     \gdef\CoverImage{"Green Cover 2019-02_DAC"}
  \fi
  \def\@coverwhite{white}%
  \def\file@prefix{}%
  \ifx0#2\relax%
    \def\@coverwhite{black!25}\def\file@prefix{LoInk}%
  \else\if0#1\relax
    \def\@coverwhite{black!25}\def\file@prefix{LoInk}%
  \fi\fi
  \newgeometry{hoffset=0pt,textwidth=6.5in,left=1in,right=1in,
               top=1in,bottom=1in}
  \global\def\colorflag{#2}
  \thispagestyle{\EmptyStyle}
  \let\sv@PageStampColor\PageStampColor
  \def\PageStampColor{\@coverwhite}
  \atxy{0in}{11in}{\includegraphics[width=8.5in,height=11in]{%
    \file@prefix\CoverImage}}
  \atxy{5.25in}{1.95in}{\scalebox{\@magfA}{\textcolor{\@coverwhite}{\bfseries%
    \sffamily\bfseries%\sffamily%
    \@arlrptno\,{\large$\bullet$}\,%
    \expandafter\scnextword\@shortpubdate \relax\relax\unskip}}}
  \begin{singlespace}
   \color{\@coverwhite}
   \sffamily
   \vspace*{2.75in minus 0.75in}
   \if F\@CRflag\vspace{.25in}\fi
   \@Huge\bfseries\sffamily\@arltitle\par
   \vspace{.2\baselineskip}
   \vspace{\the\baselineskip minus \baselineskip}
   \parskip 0pt\relax%
   \if F\@CRflag%
     \@large\Author@Attribution\ %
     \@allauthors\par
   \else
    \semi@large\Author@Attribution\ %
    \@authorsA\par% Author(s)
    {\semi@normalsize\itshape\@organizationA\par}% Organization 1 (e.g. WMRD, ARL)
    \conditional@vspace{\@authorsB}
    \@authorsB\par% Author(s)
    {\semi@normalsize\itshape\@organizationB\par}% Organization 2 (e.g. WMRD, ARL)
    \conditional@vspace{\@authorsC}
    \@authorsC\par% Author(s)
    {\semi@normalsize\itshape\@organizationC\par}% Organization 2 (e.g. WMRD, ARL)
    \conditional@vspace{\@authorsD}
    \@authorsD\par% Author(s)
    {\semi@normalsize\itshape\@organizationD\par}% Organization 2 (e.g. SURVICE\\address)
    \conditional@vspace{\@authorsE}
    \@authorsE\par% Author(s)
    {\semi@normalsize\itshape\@organizationE\par}% Organization 2 (e.g. SURVICE\\address)
    \vspace{\baselineskip}
    {\semi@normalsize\@titlepagenote\par}% Title Page Note (e.g. under contract ...)
  \fi
    \ifx 0#1\relax\else#1\fi\par
    \vfill
    \theboilerplate
  \end{singlespace}
  \@disclaimerpage
  \let\PageStampColor\sv@PageStampColor
  \clearpage
  \let\thePageStamp\savethePageStamp
}

\def\scnextword#1 #2\relax{\mytextsc#1\relax\ #2}
\def\mytextsc#1#2\relax{#1{\footnotesize\uppercase{#2}}}

\def\PageTopMargin{1in}
\def\PageLeftMargin{1in}
\newcommand\atxy[3]{%
 \AddThispageHook{\smash{\hspace*{\dimexpr-\PageLeftMargin-\hoffset+#1\relax}%
  \raisebox{\dimexpr\PageTopMargin+\voffset-#2\relax}{#3}}\\}}

\newcommand\@disclaimerpage{%
  \clearpage%
  \newgeometry{hoffset=0pt,textwidth=5.5in,left=1.5in,right=1.5in,
               top=1in,bottom=1in}
  \let\savethePageStamp\thePageStamp%
  \thispagestyle{\EmptyStyle}%
  \begin{normalsize}%
    \vspace*{2.8in}%
    \begin{center}\textbf{\Level@twosize\sffamily NOTICES}\end{center}%
    \vspace{-2.2em}%
    \begin{center}\textbf{\Level@twosize\sffamily Disclaimers}\end{center}%
    \vspace{-0.8em}%
    \begin{onehalfspace}
      \sloppy
      The findings in this report are not to be construed as an official
      Department of the Army position unless so designated by other
      authorized documents.

      Citation of manufacturer's or trade names does not constitute an
      official endorsement or approval of the use thereof.

      \@destructionnotice
    \end{onehalfspace}%
  \end{normalsize}%
}

\newcommand\Author@Attribution{by}

\def\@CRflag{F}
\let\Contract@Number\relax

\newcommand\UnderContract[1]{%
  \def\@CRflag{T}%
  \renewcommand\Author@Attribution{prepared by}%
  \def\Contract@Number{#1}%
  \titlepagenote{under contract \Contract@Number}%
  \ifthenelse{\equal{\@arlrptno}{ARL-TR-XXXX}}{\def\@arlrptno{ARL-CR-XXXX}}{}%
}

\newcommand\conditional@vspace[1]{%
  \if\relax#1\relax\else\vskip .75\baselineskip minus 30pt\relax\fi%
}

\newcommand\arltitlepage{
  \clearpage
  \newgeometry{hoffset=0pt,textwidth=6.5in,left=1in,right=1in,
               top=1in,bottom=1in}
  \setcounter{page}{1}
  \def\barcolor{black!50}
  \thispagestyle{\EmptyStyle}
  \atxy{1in}{2.09in}{\textcolor{\barcolor}{\rule{6.5in}{1.2mm}}}
  \atxy{7.5in}{3in}{\llap{\includegraphics[width=2.3in]{\LogoImage}}}
  \atxy{5.25in}{1.98in}{\scalebox{\@magfA}{\bfseries%
    \sffamily\bfseries%\sffamily%
    \@arlrptno\,{\large$\bullet$}\,%
    \expandafter\scnextword\@shortpubdate\ \relax\relax\unskip}}
  \begin{singlespace}
    \sffamily
    \vspace*{2.75in minus 0.75in}
    \if F\@CRflag\vspace{.25in}\fi
      \@Huge\bfseries\sffamily\@arltitle\par
    \vspace{.2\baselineskip}
    \vspace{\the\baselineskip minus \baselineskip}
    \parskip 0pt\relax%
    \if F\@CRflag%
      \@large%
    \else
      \semi@large%
    \fi
    \Author@Attribution\ %
    \@large\@authorsA\par% Author(s)
    {\semi@normalsize\itshape\@organizationA\par}% Organization 1 (e.g. WMRD, ARL)
    \conditional@vspace{\@authorsB}
    \@authorsB\par% Author(s)
    {\semi@normalsize\itshape\@organizationB\par}% Organization 2 (e.g. WMRD, ARL)
    \conditional@vspace{\@authorsC}
    \@authorsC\par% Author(s)
    {\semi@normalsize\itshape\@organizationC\par}% Organization 2 (e.g. WMRD, ARL)
    \conditional@vspace{\@authorsD}
    \@authorsD\par% Author(s)
    {\semi@normalsize\itshape\@organizationD\par}% Organization 2 (e.g. SURVICE\\address)
    \conditional@vspace{\@authorsE}
    \@authorsE\par% Author(s)
    {\semi@normalsize\itshape\@organizationE\par}% Organization 2 (e.g. SURVICE\\address)
    \vspace{\baselineskip}
    {\semi@normalsize\@titlepagenote\par}% Title Page Note (e.g. under contract ...)
    \vfill
    \theboilerplate
  \end{singlespace}
  \clearpage
}

\newenvironment{MultipleSourcesMemo}[2][\OfficeSymbol]% P.iii IN CLAS RPTS W/ MULT. SRCS.
{%
  \clearpage%
  \singlespacing%
  \parskip 1em\relax%
  \atxy{0in}{11in}{\includegraphics[width=8.5in,height=11in]{LH-CCDC-ARL(NoFooter)}}%
  \vspace*{.4in}
  #1\hfill #2 \@pubdate

  MEMORANDUM FOR RECORD

  SUBJECT: Classification sources for \@arlrptno,
  {``\def\\{}\@arltitle,'' by \@allauthors}

  \begin{enumerate}[topsep=8pt,itemsep=8pt]
  \item (U) The subject report has been marked with the following regrading
  instructions:\vspace{6pt}

  {\let\textbf\relax\downgradetext}

  \item (U) The classification sources used are as follows:

  \begin{enumerate}[label=\alph*.,topsep=4pt,itemsep=10pt]
}
{\end{enumerate}\end{enumerate}\vspace{.5in}}

\newcommand\signMultipleSourcesMemo[2]{%
  \begin{center}%
    \leavevmode\rlap{%
    \stackengine{14.5pt}% THE \singlespacing BASELINE SKIP
      {#1}{#2}{U}{l}{F}{F}{L}}%
   \end{center}
  \clearpage%
}

\newenvironment{createSFtwoNINEeight}{
  \clearpage
  \begin{singlespace}
    \leftskip-.04in% Fine adjustment based on graphics image provided
    \begin{picture}(612,650)(70,82)
      \includegraphics{SF298}
      \footnotesize
      \@SFitemONE{\@pubdate}
      \@SFitemTWO{\@reporttype}
      \@SFitemFOUR{\@arltitle}
      \@SFitemFIVEa{\Contract@Number}
      \@SFitemSIX{\@allauthors}
      \@SFitemEIGHT{\@arlrptno}
      \@SFitemTWELVE{\raggedright\@distribution}
      \def\SFitemSIXTEENaVALUE{Unclassified}
      \def\SFitemSIXTEENbVALUE{Unclassified}
      \def\SFitemSIXTEENcVALUE{Unclassified}
}{
      \put(-565,101){\parbox[c]{0.8in}{
          \SFitemSIXTEENaVALUE}}
      \put(-495,101){\parbox[c]{0.8in}{
          \SFitemSIXTEENbVALUE}}
      \put(-423,101){\parbox[c]{0.8in}{
          \SFitemSIXTEENcVALUE}}
      \put(-353,101){\parbox[c]{0.8in}{\centering
          \SFitemSEVENTEENvalue}}
      \normalsize
    \end{picture}
  \end{singlespace}
}
\newcommand\@SFitemONE[1]{\put(-564.5,656){#1}}
\newcommand\SFitemTWO[1]{\def\@reporttype{#1}}
\newcommand\@SFitemTWO[1]{\put(-424,656){#1}}
\newcommand\SFitemTHREE[1]{\put(-218,656){#1}}
\newcommand\@SFitemFOUR[1]{\put(-564.5,630){\parbox[t]{4.65in}{%
  \raggedright\def\\##1{\unskip\@sptoken##1}#1}}}
\newcommand\SFitemFOUR[1]{}
\newcommand\@SFitemFIVEa[1]{\put(-218,630){#1}}
\newcommand\SFitemFIVEb[1]{\put(-218,604){#1}}
\newcommand\SFitemFIVEc[1]{\put(-218,578){#1}}
\newcommand\SFitemFIVEd[1]{\put(-218,554){#1}}
\newcommand\SFitemFIVEe[1]{\put(-218,528){#1}}
\newcommand\SFitemFIVEf[1]{\put(-218,501){#1}}
\newcommand\@SFitemSIX[1]{\put(-564.5,554){\parbox[t]{4.65in}{%
  \raggedright\def\\##1{\unskip\@sptoken##1}#1}}}
\newcommand\SFitemSEVEN[1]{\put(-564.5,474){\parbox[t]{4.65in}{\raggedright#1}}}
\newcommand\@SFitemEIGHT[1]{\put(-218,462){#1}}
\newcommand\SFitemNINE[1]{\put(-564.5,412){\parbox[t]{4.65in}{\raggedright#1}}}
\newcommand\SFitemTEN[1]{\put(-218,410){#1}}
\newcommand\SFitemELEVEN[1]{\put(-218,383){#1}}
\newcommand\@SFitemTWELVE[1]{%
  \setbox0=\hbox{\parbox[b]{7.15in}{\setstretch{0.8}#1}}%
  \ifdim\ht0>1.8\baselineskip%
    \put(-564.5,356){\scriptsize\parbox[c]{7.15in}{\setstretch{1.0}\hspace{160pt}#1}}%
  \else
    \ifdim\ht0>1.2\baselineskip%
      \put(-564.5,346){\scriptsize\parbox[b]{7.15in}{\setstretch{1.0}#1}}%
    \else%
      \put(-564.5,353){\parbox[c]{7.15in}{\setstretch{0.8}#1}}%
    \fi%
  \fi%
}
\newcommand\SFitemTHIRTEEN[1]{\put(-564.5,325){%
  \parbox[t]{7.1in}{\raggedright#1}}}
\newcommand\SFitemFOURTEEN[1]{\put(-564.5,282){%
  \parbox[t]{7.1in}{\raggedright#1}}}
\newcommand\SFitemFIFTEEN[1]{\put(-564.5,148){%
  \parbox[b]{7.1in}{\raggedright#1}}}
\newcommand\SFitemSIXTEENa[1]{\def\SFitemSIXTEENaVALUE{#1}}
\newcommand\SFitemSIXTEENb[1]{\def\SFitemSIXTEENbVALUE{#1}}
\newcommand\SFitemSIXTEENc[1]{\def\SFitemSIXTEENcVALUE{#1}}
\newcommand\SFitemSEVENTEEN[1]{\def\SFitemSEVENTEENvalue{#1}}
\newcommand\SFitemEIGHTEEN[1]{\put(-285,101){\parbox[c]{0.8in}{\centering#1}}}
\newcommand\SFitemNINETEENa[1]{\put(-218,124){#1}}
\newcommand\SFitemNINETEENb[1]{\put(-218,99){#1}}

\SFitemTWO{Unknown Report Type}% DEFAULT SF298 ITEM 2
\def\@autodetectSFitemTWO#1-#2#3#4#5\relax{%
  \ifx-#4%
    \ifx R#3%
      \ifx T#2\SFitemTWO{Technical Report}\else%
        \ifx M#2\SFitemTWO{Memorandum Report}\else%
          \ifx S#2\SFitemTWO{Special Report}\else%
            \ifx C#2\SFitemTWO{Contractor Report}\else%
              \ifx I#2\SFitemTWO{Internal Report}%
      \fi\fi\fi\fi\fi%
    \else\ifx N#3%
      \ifx T#2\SFitemTWO{Technical Note}\fi%
    \else\ifx P#3%
      \ifx R#2\SFitemTWO{Reprint Report}\fi%
    \fi\fi\fi%
  \fi%
}

%%THE NEXT ENVIRONMENT ACCOUNTS FOR FOUR DIFFERENT SPELLINGS OF
%%  ACKNOWLEDGEMENT, ACKNOWLEDGEMENTS, ACKNOWLEDGMENT, ACKNOWLEDGMENTS
\newcommand \acknowledgement  {\def\ackname{%
  \spaceouthelpAcopy Acknowledgment \relax}  \ack}% SPACE BEFORE \relax IMPORTANT!!
\newcommand \acknowledgements {\def\ackname{%
  \spaceouthelpAcopy Acknowledgments \relax} \ack}% SPACE BEFORE \relax IMPORTANT!!
\newcommand \acknowledgment   {\def\ackname{%
  \spaceouthelpAcopy Acknowledgment \relax}  \ack}% SPACE BEFORE \relax IMPORTANT!!
\newcommand \acknowledgments  {\def\ackname{%
  \spaceouthelpAcopy Acknowledgments \relax} \ack}% SPACE BEFORE \relax IMPORTANT!!
%%INSERT NEW ENVIRONMENT FOR ack[knowledgment]
\newcommand\theack{\section*{\ClassificationLabel \ackname}}
\newenvironment{ack}{
  \clearpage
  \addcontentsline{toc}{section}{\ClassificationLabel \ackname}
  \theack}{}
\newcommand{\blankpage}{
  \clearpage
%% "Intentionally left blank" REMOVED AS OF V2.44
}

%% spaceout algorithm for section headings
\def\Letter@Gap{\kern\Letter@Space}
\newcommand\spaceout[2][\Letter@Space]{%
  \edef\tmp{\the\Letter@Space}%
  \setlength\Letter@Space{#1}%
  \spaceouthelpA#2{} \relax\relax%
  \setlength{\Letter@Space}{\tmp}%
}
\def\spaceouthelpA#1 #2\relax{%
  \spaceouthelpB#1{}\relax\relax%
  \ifx\relax#2\relax\else\ \Letter@Gap%
    \spaceouthelpA#2\relax\fi
}
\def\spaceouthelpB#1#2\relax{%
  \ifx\relax#2\relax#1\else
  \ifx#1\color\interceptColor#2\relax\relax\else
  \ifx#1\label\interceptLabel#2\relax\relax\else
      #1\protect\Letter@Gap\spaceouthelpB#2\relax%
  \fi\fi\fi%
}
\def\interceptColor#1#2\relax{%
  \color{#1}%
  \ifx\relax#2\else%
    \spaceouthelpB#2\relax%
  \fi%
}
\def\interceptLabel#1#2\relax{%
  \label{#1}%
  \ifx\relax#2\else%
    \spaceouthelpB#2\relax%
  \fi%
}
\def\sectspace{\let\spaceouthelpAcopy\spaceouthelpA}
\def\nosectspace{\let\spaceouthelpAcopy\relax}
\nosectspace

%% Redefine original commands called \section and \section* so as to place
%% ruler lines above and below section names in the ARL format for report
%% sections
\newcommand\section{\@ifstar{\ruledheading}{\ruledsection}}
\newcommand\subsection{\@ifstar{\ruledsubheading}{\ruledsubsection}}
\newcommand\subsubsection{\@ifstar{\ARLsubsubheading}{\ARLsubsubsection}}
\newcommand\paragraph{\@ifstar{\ARLparaheading}{\ARLparagraph}}

%%\ruledsection is called by \section

\newcommand\ruledsection[1]{\sectionprelude\origsection{%
  \spaceouthelpAcopy#1 \relax\relax%
  }\sectionpostlude
  \par%
}

\newcommand\ruledsubsection[1]{\bgroup\onehalfspacing%
  \subsectionprelude\origsubsection{#1}\subsectionpostlude\par\egroup%
}

\newcommand\ARLsubsubsection[1]{%
  \origsubsubsection{#1}\par\subsubsectionpostlude\par%
}

\newcommand\ARLparagraph[1]{%
  \origparagraph{#1}\par\paragraphpostlude\par%
}

%% \ruledheading is called by \section*
%% and is like \ruledsection, but unnumbered and un-toc'ed

\newcommand\ruledheading[1]{\bgroup\onehalfspacing%
  \sectionprelude\origsection*{%
  \spaceouthelpAcopy#1 \relax\relax%
  }\sectionpostlude
  \par\egroup%
}

\newcommand\ruledsubheading[1]{\subsectionprelude%
  \origsubsection*{#1}\subsectionpostlude\par%
}

\newcommand\ARLsubsubheading[1]{%
  \origsubsubsection*{#1}\par\subsubsectionpostlude\par%
}

\newcommand\ARLparaheading[1]{%
  \origparagraph*{#1}\par\paragraphpostlude\par%
}

%%%%%

\newcommand\sectionprelude{%
  \needspace{\sectionSpaceMultiplier\defaultsectionspace}%
  \vspace{-6mm}%
  \def\sectionSpaceMultiplier{1}%
}
\newcommand\sectionpostlude{
  \vspace{-12mm}%
  {%
   \ooalign{\color{ReportColor}\rule{\textwidth}{1.6mm}%
    \cr\color{white}\rule[0.6mm]{\textwidth}{0.4mm}\cr}%
  }%
  \vspace{-3mm}
}
\newcommand\subsectionprelude{%
  \needspace{\sectionSpaceMultiplier\defaultsubsectionspace}%
  \def\sectionSpaceMultiplier{1}%
}
\newcommand\subsectionpostlude{
  \vspace{-9mm}%
  {%\color{ReportColor}%
    \rule{\textwidth}{0.01in}%
  }%
  \vspace{-3mm}
}

\newcommand\subsubsectionpostlude{\vspace{-3mm}}

\newcommand\paragraphpostlude{\vspace{-2mm}}

%% \baresection is like \section*, but is placed in the toc
\newcommand\baresection[1]{\section*{#1}\addcontentsline{toc}{section}{#1}}

%% See file header to read about \allocateSpaceOnce and \sectionSpaceMultiplier
\newlength\defaultsectionspace
\setlength\defaultsectionspace{5\baselineskip}
\newlength\defaultsubsectionspace
\setlength\defaultsubsectionspace{1\baselineskip}
\def\sectionSpaceMultiplier{1}
\newcommand\allocateSpaceOnce[1]{\def\sectionSpaceMultiplier{#1}}

%%
%% Create new bibliography environment that makes ruled section header
\newdimen\bibindent
\setlength\bibindent{1.5em}
\newenvironment{thebibliography}[1]
 {\section{\refname}\vspace*{-0.9em}%
  \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
  \list{\@biblabel{\@arabic\c@enumiv}}%
       {\settowidth\labelwidth{\@biblabel{#1}}%
        \leftmargin\labelwidth
        \advance\leftmargin\labelsep
        \@openbib@code
        \usecounter{enumiv}%
        \let\p@enumiv\@empty
        \renewcommand\theenumiv{\@arabic\c@enumiv}}%
  \sloppy
  \clubpenalty4000
  \@clubpenalty \clubpenalty
  \widowpenalty4000%
  \sfcode`\.\@m}
 {\def\@noitemerr
   {\@latex@warning{Empty `thebibliography' environment}}%
  \endlist}
%% Redefine the appearance of bibliography labels for ARL style..
\renewcommand*{\@biblabel}[1]{#1.\hfill}
%%
%% Define \arlbibliography, so that modified achemso style, named ARL
%% is utilized.  Abbreviated journal titles, by default
\newcommand {\arlbibliography}[2][s p]{%
  \clearpage%
  \bibliographystyle{ARL}%
  \def\journaltitles{journalshort}%
  \def\statenames{statesfull}%
  \setsepchar{ }%
  \readlist*\@bibarg{#1}%
  \foreachitem\x\in\@bibarg[]{%
    \if f\x\relax\def\journaltitles{journalfull} \fi%
    \if p\x\relax\def\statenames{statespostal} \fi%
  }%
  \bibliography{\journaltitles,\statenames,#2}%
}
%% Create the environment {references} which is used to list references in
%% a "verse" style, for those who don't use BiBTeX.
%%
\newcounter{refcount}
\newenvironment{references}
  {\clearpage \section{References}
   \begin{list}{\makebox[1.5em]{\arabic{refcount}.\hfill}}{
     \topsep -0.2em \leftmargin 1.6em \labelsep 0.1em \usecounter{refcount}}
  }
  {\end{list}}

%% Free footnote, without any associated number or citation.
\let\svthefootnote\thefootnote
\newcommand\freefootnote[1]{%
  \let\thefootnote\relax%
  \footnotetext{#1}%
  \let\thefootnote\svthefootnote%
}

%% Instead of changing footnote style for appendices, just introduce
%% \footnotecite for temporarily using numbered footnotes with a new
%% counter.
%% Note: user will have to add bibentry package and \nobibliography*
%% declaration.
\newcounter{citefn}
\newcommand\footnotecite[2][unspecified]{%
  \renewcommand\thefootnote{\arabic{footnote}}%
  \stepcounter{citefn}%
  \footnotemark[\arabic{citefn}]%
  \footnotetext[\arabic{citefn}]{#2}%
  \expandafter\edef\csname #1citemark\endcsname{\arabic{citefn}}%
  \let\thefootnote\svthefootnote%
}
%% To recite a reference in an appendix that has already been cited earlier
%% in the appendix. Use the tag specified earlier in optional argument of
%% \footnotecite
\newcommand\recite[1]{$^{\csname #1citemark\endcsname}$}

%% Create commands to create an unnumbered (lone) appendix or a series of
%% appendices, as needed
\newcommand\appendix{%
  \renewcommand\addcontentsline[3]{%
    \def\qtest{##2}%
    \def\qmatch{subsection}%
    \ifx\qmatch\qtest\else%
      \def\qmatch{subsubsection}%
      \ifx\qmatch\qtest\else%
        \svaddcontentsline{##1}{##2}{##3}%
    \fi\fi%
  }%
  \@ifstar{\loneappendix}{\anappendix}%
}

\AtBeginDocument{\let\svaddcontentsline\addcontentsline}

\newcounter{appndx}
\setcounter{appndx}{0}

\newcommand\loneappendix[2][p]{
  \clearpage
  \refstepcounter{appndx}
  \setcounter{section}{\arabic{appndx}}
  \renewcommand\thesection {\appendixname~\Alph{appndx}.}
  \setcounter{subsection}{0}
  \renewcommand\thesubsection {\Alph{appndx}.\@arabic\c@subsection}
  \setcounter{paragraph}{0}
  \setcounter{subparagraph}{0}
  \setcounter{figure}{0}
  \renewcommand\thefigure{\Alph{appndx}-\@arabic\c@figure}
  \setcounter{table}{0}
  \renewcommand\thetable{\Alph{appndx}-\arabic{table}}
  \setcounter{equation}{0}
  \renewcommand\theequation {\Alph{appndx}-\arabic{equation}}
  \setcounter{footnote}{0}
  \def\appendixtitle{\appendixname. #2}
  \addcontentsline{toc}{section}\appendixtitle
  \if p#1\vspace*{\fill}\fi
  \theappendix\appendixtitle
  \if p#1\vspace*{\fill}\clearpage\fi
}

%% The following patch is needed to fix the use of the \p@<counter>
%% command
\renewcommand*\refstepcounter[1]{\stepcounter{#1}%
  \protected@edef\@currentlabel{%
    \csname p@#1\expandafter\endcsname
      \csname the#1\endcsname
  }%
}
\renewcommand{\p@appndx}[1]{\Alph{appndx}}

\newcommand\anappendix[2][p]{
  \clearpage
  \refstepcounter{appndx}
  \setcounter{section}{\arabic{appndx}}
  \renewcommand\thesection {\appendixname~\Alph{appndx}.}
  \setcounter{subsection}{0}
  \renewcommand\thesubsection {\Alph{appndx}.\@arabic\c@subsection}
  \setcounter{paragraph}{0}
  \setcounter{subparagraph}{0}
  \setcounter{equation}{0}
  \setcounter{figure}{0}
  \renewcommand\thefigure{\Alph{appndx}-\@arabic\c@figure}
  \setcounter{table}{0}
  \renewcommand\thetable{\Alph{appndx}-\arabic{table}}
  \renewcommand\theequation {\Alph{appndx}-\arabic{equation}}
  \setcounter{footnote}{0}
  \def\appendixtitle{\appendixname~\Alph{appndx}. #2}
  \addcontentsline{toc}{section}\appendixtitle
  \if p#1\vspace*{\fill}\fi
  \theappendix\appendixtitle
  \if p#1\vspace*{\fill}\clearpage\fi
}
\newcommand\theappendix[1]{
  \section*{\centering#1}
}
%% Create a new command called \distributionlistpage that inserts a
%% placeholder page for the Distribution List page.
\newcommand \distributionlistpage{
  \clearpage
  \addcontentsline{toc}{section}{\ClassificationLabel Distribution List}
  \begin{center}
  \rule{0em}{4.5in}
  REPLACE THIS PAGE WITH DISTRIBUTION LIST
  \end{center}
}

%% To create your own distribution list, first call on \distlistsetup
\newcommand\distlistsetup[1][w]{%
  \clearpage%
  \singlespace%
  \addcontentsline{toc}{section}{\ClassificationLabel Distribution List}%
  \newlength\itemsepSAVE%
  \let\itemsepSAVE\itemsep%
  \ifthenelse{\equal{x}{#1}}{}{%
    \def\f@open{\immediate\openout\tempfile=}%
    \edef\f@name{"DL_\@arlrptno.tex"}%
    \newwrite\tempfile%
    \expandafter\f@open\f@name%
    \immediate\write\tempfile{\noexpand\documentclass{arlticle}}%
    \immediate\write\tempfile{\noexpand\makeatletter}%
    \protected@iwrite\tempfile{\let\\\relax}%
        {\def\noexpand\noexpand\noexpand\@arltitle{\@arltitle}}
    \protected@iwrite\tempfile{\let\\\relax}%
        {\def\noexpand\noexpand\noexpand\@allauthors{\@allauthors}}
    \protected@iwrite\tempfile{\let\\\relax}%
        {\def\noexpand\noexpand\noexpand\@arlrptno{\@arlrptno}}
    \protected@iwrite\tempfile{\let\\\relax}%
        {\def\noexpand\noexpand\noexpand\@shortpubdate{\@shortpubdate}}
    \protected@iwrite\tempfile{\let\\\relax}%
        {\def\noexpand\noexpand\noexpand\MandatoryDL{\MandatoryDL}}
    \@ifundefined{LocalMandatoryDL}{}%
    {%
    \protected@iwrite\tempfile{\let\\\relax}%
        {\def\noexpand\noexpand\noexpand\LocalMandatoryDL{\LocalMandatoryDL}}
    }%
    \protected@iwrite\tempfile{\let\\\relax}%
        {\def\noexpand\noexpand\noexpand\UserDL{\UserDL}}
    \protected@iwrite\tempfile{\let\\\relax}%
        {\def\noexpand\noexpand\noexpand\ecwarningtext{\@ecwarningtext}}
    \@ifundefined{downgradeARGa}{}%
    {%
    \protected@iwrite\tempfile{\let\\\relax}%
        {\noexpand\noexpand\noexpand\downgrading%
          {\downgradeARGa}{\downgradeARGb}{\downgradeARGc}%
          {\downgradeARGd}{\downgradeARGe}}%
    }%
    \protected@iwrite\tempfile{\let\\\relax}%
        {\def\noexpand\noexpand\noexpand\@distribution{\@distribution}}
    \immediate\write\tempfile{\noexpand\input{DLprint.tex}}%
    \immediate\write\tempfile{\noexpand\makeatother}%
    \immediate\closeout\tempfile%
    \edef\@DLcompile{pdflatex \f@name}
    \if p#1\write18{\@DLcompile}\fi%
  }%
}

\newcommand\dlt{\rule{0.92em}{0in}} % \dlt = [Distribution List Tab]

\newcommand\apg{\vspace{1.5em}\item\underline{ABERDEEN PROVING GROUND}}

\newcounter{Lines}
\def\addressdefaultlines{2}
\newcommand\dlitem[2][\addressdefaultlines]{
  \setcounter{Lines}{#2}
  \addtocounter{Lines}{#1}
  \item[#2]
  \needlines{\arabic{Lines}}
}

%% X-TENDED VERSION OF \dlitem, 3 ARGUMENTS, #3 CAN CONTAIN LINEBREAKS
%% AND APPEARS BELOW TOTAL COPY COUNT (WHICH IS ARGUMENT #2)
\newcommand\xdlitem[3][\addressdefaultlines]{
  \setcounter{Lines}{#2}
  \addtocounter{Lines}{#1}
  \setlength{\labelwidth}{0.61in}\setlength{\labelsep}{0.05in}%
  \item[\hfill%
    \tabcolsep=0pt\smash{\begin{tabular}[t]{c}#2\\#3\end{tabular}}%
  \hfill]
%% RESTORE ORIGINAL VALUES
  \setlength{\labelwidth}{0.31in}\setlength{\labelsep}{0.295in}%
  \needlines{\arabic{Lines}}
}

%% (PDF) VERSION OF \dlitem
\newcommand\pdlitem[2][\addressdefaultlines]{
  \setcounter{Lines}{#2}
  \addtocounter{Lines}{#1}
  \setlength{\labelwidth}{0.61in}\setlength{\labelsep}{0.05in}%
  \item[\hfill%
    \tabcolsep=0pt\smash{\begin{tabular}[t]{c}#2\\(PDF)\end{tabular}}%
  \hfill]
%% RESTORE ORIGINAL VALUES
  \setlength{\labelwidth}{0.31in}\setlength{\labelsep}{0.295in}%
  \needlines{\arabic{Lines}}
}

%% (HC) VERSION OF \dlitem
\newcommand\hdlitem[2][\addressdefaultlines]{
  \setcounter{Lines}{#2}
  \addtocounter{Lines}{#1}
  \setlength{\labelwidth}{0.61in}\setlength{\labelsep}{0.05in}%
  \item[\hfill%
    \tabcolsep=0pt\smash{\begin{tabular}[t]{c}#2\\(HC)\end{tabular}}%
  \hfill]
%% RESTORE ORIGINAL VALUES
  \setlength{\labelwidth}{0.31in}\setlength{\labelsep}{0.295in}%
  \needlines{\arabic{Lines}}
}

%% NEW \ldlitem, USES SAME CENTERING AS \xdlitem, \pdlitem, AND \hdlitem
\newcommand\ldlitem[2]{%
  \setlength{\labelwidth}{0.61in}\setlength{\labelsep}{0.05in}%
  \item[\hfill%
    \tabcolsep=0pt\smash{\begin{tabular}[t]{c}#1\\#2\end{tabular}}%
  \hfill]%
%% RESTORE ORIGINAL VALUES
  \setlength{\labelwidth}{0.31in}\setlength{\labelsep}{0.295in}%
}

\newcommand\dlem[2][1]{}

\newcommand\needlines[1]{\needspace{#1\baselineskip}} % With \item to allocate
%%                                                      space for full item
%% At end of distribution list, call on \distlistcleanup to restore
%% settings
\newcommand\distlistcleanup{
  \clearpage
  \onecolumn
  \pagestyle{\PlainStyle}
  \onehalfspacing%       Undoes the single spacing of the distribution list
  \if T\PageStampFlag%
    \def\PlainStyle{pagestamp}
    \def\EmptyStyle{emptypagestamp}
    \pagestyle{\PlainStyle}
  \fi
}

%% After calling \distlistsetup, enter the distributionlist environment
%% to input your distribution list entries
%% Entries should be of the form:
%% \item [no. to distribute]
%% line 1\\
%% line 2\\
%% \dlt indented-line-as-needed
%%
\newenvironment{distributionlist}{
   \columnsep 0.2in
   \twocolumn
   \footnotesize
   \begin{list}{}{
     \setlength{\labelwidth}{0.24in}%
     \setlength{\labelsep}{0.29in}%
     \setlength{\leftmargin}{0.5in}%
     \setlength\itemsep{0.5em}
   }
  }
  {\setlength\itemsep{\itemsepSAVE}
   \clearpage
   \end{list}
   \normalsize
}

%% \backcover prints out a totally blank page, for use as a report's
%% backcover
\newcommand\backcover{
  \clearpage
  \let\@abbrevdistribution\relax
%%  \pagecolor{\file@prefix ReportColor}% IF ONE WANTED COLORED BACK COVER
  \def\PageStampColor{black}% SINCE \@coverwhite TOO PALE
  \def\thePageStamp{}%
  \thispagestyle{\EmptyStyle}
  \checkoddpage\ifoddpage% NEEDED TO MAKE OVERALL CLASS APPEAR ON EVEN PAGE #
    \textrm{}
    \clearpage
  \fi
  \PageStamp[]
  \let\thePageStamp\savethePageStamp
  \thispagestyle{\EmptyStyle}
  \textrm{}
}

\newcommand\widow{%
  \widowpenalty=10000

  \widowpenalty=150}
\newcommand\orphan{%
  \needlines{2}%
}

%% To make single spacing following periods, colons, etc.
\frenchspacing

\usepackage{tocloft}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% MODIFIED TOCLOFT PACKAGE TO FACILITATE PROPER FORMATTING OF
%% LISTS OF FIGURES AND TABLES IN THE ARL STYLE
%%
%%  Set \cftfigpresnum to "Fig.~" - this adds "Fig. " before each figure number in LOF
%%  Set \cfttabpresnum to "Table~" - this adds "Table " before each table number in LOT
%%
%%  Search for following commands to change values
%%  \setlength{\cftfigindent}{0in}       No indentation in LOF
%%  \setlength{\cftfignumwidth}{0.56in}  Width of box containing "Figure xx."
%%  \setlength{\cfttabindent}{0in}       No indentation in LOT
%%  \setlength{\cfttabnumwidth}{0.64in}  Width of box containing "Table xx."
%%  \setlength{\growloflabel}{0.em}          Extra width added to \cftfignumwidth
%%  \setlength{\growlotlabel}{0.em}          and/or \cfttabnumwidth
%%  to allow for wider caption widths
%%                                       from Figure or Table  A-1, etc.
%%  (set in document preamble).
%%
%% Modified \cfttabfillnum and \cftfigfillnum to do full dot leader.

\renewcommand{\cftdotsep}{0.6}

\renewcommand\cftsecfont{\bfseries\sffamily\Level@twosize}
\renewcommand\cftsubsecfont{\sffamily\Level@foursize}
\renewcommand\cftsubsubsecfont{\sffamily\Level@foursize}
\renewcommand\cftparafont{\sffamily\Level@foursize}
\renewcommand\cftsecpagefont{\bfseries\sffamily}
\renewcommand\cftsubsecpagefont{\sffamily\small}
\renewcommand\cftsubsubsecpagefont{\sffamily\footnotesize}
\renewcommand\cftparapagefont{\sffamily\footnotesize}
\renewcommand\cftsecaftersnum{.}

\setlength{\cftbeforesecskip}{0.5em \@plus\p@}
\setlength{\cftbeforesubsecskip}{-.5em \@plus\p@}
\setlength{\cftbeforesubsubsecskip}{-.5em \@plus\p@}
\setlength{\cftbeforeparaskip}{-.5em \@plus\p@}
\setlength{\cftbeforefigskip}{-.5em \@plus\p@}
\setlength{\cftbeforetabskip}{-.5em \@plus\p@}

\setlength{\cftfigindent}{0in}
\setlength{\cftfignumwidth}{0.56in}
\renewcommand{\cftfigpresnum}{Fig.~}
%% HACK \cftfigfillnum TO RUN DOT LEADER ALL THE WAY UP TO THE PAGE#
\renewcommand{\cftfigfillnum}[1]{%
  {\cftfigleader}\nobreak
%% \hb@xt@\@pnumwidth{\hfil\cftfigpagefont #1}\cftfigafterpnum\par}%ORIG
  \cftfigpagefont #1\cftfigafterpnum\par}

\setlength{\cfttabindent}{0in}
\setlength{\cfttabnumwidth}{0.64in}
\renewcommand{\cfttabpresnum}{Table~}
%% HACK \cfttabfillnum TO RUN DOT LEADER ALL THE WAY UP TO THE PAGE#
\renewcommand{\cfttabfillnum}[1]{%
  {\cfttableader}\nobreak
%% \hb@xt@\@pnumwidth{\hfil\cfttabpagefont #1}\cfttabafterpnum\par}%ORIG
  \cfttabpagefont #1\cfttabafterpnum\par}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\tableofcontents{}
\def\listoffigures{}
\def\listoftables{}
\def\sec@star{\section*}
\AtBeginDocument{%
  \let\svnumberline\numberline%
  %%MODIFY TABLE OF CONTENTS TO INCLUDE RULE LINES
  %% ALSO, SCREEN OUT FREE FOOTNOTES IN SECTION TITLES FROM TOC
  \renewcommand\tableofcontents{%
    \let\svfreefootnote\freefootnote%
    \def\freefootnote##1{}%
    \clearpage%
      \AddEverypageHook{\smash{\hspace*{%
          \dimexpr-\PageLeftMargin-\hoffset+1.5in\relax}%
        \raisebox{\dimexpr\PageTopMargin+\voffset-9.85in\relax}{%
          \textcolor{black!60}{%
        \parbox[t]{\textwidth}{\singlespacing\fontsize{7pt}{8pt}\selectfont%
          \sffamily\@abbrevdistribution}}}}}%
      \newgeometry{hoffset=0pt,textwidth=5.5in,left=1.5in,right=1.5in,%
                   top=1in,bottom=\dimexpr1in+10pt,%
                   footskip=\dimexpr\footskip+10pt}%
    \thetableofcontents%
    \let\freefootnote\svfreefootnote%
  }
  \newcommand\thetableofcontents{
    \section*{\contentsname}\vspace{-12pt}
    \renewcommand{\@pnumwidth}{1.0em}
    \begin{singlespace}
       \def\Letter@Gap{}
       \renewcommand\cftdotsep{10000}%
       \@starttoc{toc}%
    \end{singlespace}%
  }
  %%MODIFY \listoffigures" TO ADD TO TABLE OF CONTENTS & DRAW RULER LINES
  \renewcommand\listoffigures{\@ifstar{\@listoffigures[]}{\@listoffigures}}
  \newcommand\thelistoffigures{
    \section*{\listfigurename}\vspace{4pt}%
    \renewcommand{\@pnumwidth}{1.55em}
    \cftsetrmarg{0pt plus1fil}
    \if T\Single@Figure
        \def\numberline##1{\svnumberline{}}%
        \renewcommand{\cftfigpresnum}{Figure}
    \else
      \let\numberline\svnumberline
    \fi
    \begin{singlespace}
      \@starttoc{lof}%
    \end{singlespace}
  }
  %%MODIFY \listoftables" TO ADD TO TABLE OF CONTENTS & DRAW RULER LINES
  \renewcommand\listoftables{\@ifstar{\@listoftables[]}{\@listoftables}}
  \newcommand\thelistoftables{
    \section*{\listtablename}\vspace{4pt}%
    \renewcommand{\@pnumwidth}{1.55em}
    \cftsetrmarg{0pt plus1fil}
    \if T\Single@Table
      \def\numberline##1{\svnumberline{}}%
    \else
      \let\numberline\svnumberline
    \fi
    \begin{singlespace}
      \@starttoc{lot}%
    \end{singlespace}
  }
}
\newcommand\@listoffigures[1][\clearpage]{%
  #1%
  \addcontentsline{toc}{section}{\listfigurename}%
  \thelistoffigures%
}
\newcommand\@listoftables[1][\clearpage]{%
  #1%
  \addcontentsline{toc}{section}{\listtablename}%
  \thelistoftables%
}

\onehalfspacing
%% ARL FOOTNOTE STYLE
\usepackage[symbol,perpage]{footmisc}

%% NULLIFY ABBREVIATED DISTRIBUTION DEFAULT (5/22/19)
\AtBeginDocument{\def\@abbrevdistribution{}}%

\endinput
%%
%% End of file `arlticle.cls'.
