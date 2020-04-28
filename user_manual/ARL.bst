%
% ARL.bst, highly modified from ACS style "achemso".
% 2019/05/21, V43
% Maintained by Steven B. Segletes of ARL/WMRD
%
% V24 was the initial release attempting to conform to the new
% ARL style guidelines of 2014/07/01.
%%
%% This is file `achemso.bst',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% achemso.dtx  (with options: `huvudkod')
%% 
%% Copyright (c) 1995-98 by Mats Dahlgren <matsd@sssk.se>.
%% All rights reserved.  See the file `achemso.ins' for information
%% on how you may (re-)distribute the `achemso' package files.
%% You are not allowed to make any changes to this file without
%% explicit permission from the author.
%% 
ENTRY
  { accessed
    address
    author
    booktitle
    chapter
    doi
    edition
    editor
    howpublished
    institution
    journal
    key
    month
    note
    number
    pages
    publisher
    revised
    school
    series
    title
    titleclass
    referenceclass
    type
    venue
    volume
    website
    year
    audience
    conferenceyear
  }
  {}
  { label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
}

STRINGS { s t }

FUNCTION {output.nonnull}
{ 's :=
  output.state mid.sentence =
    { " " * write$ }
    { output.state after.block =
        { ", " * write$
     newline$
     "\newblock " write$
   }
   { output.state before.all =
       'write$
       { add.period$ " " * write$ }
     if$
   }
      if$
      mid.sentence 'output.state :=
    }
  if$
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {fin.entry.noperiod}
{ write$
  newline$
}

FUNCTION {fin.entry}
{ add.period$
  write$
  newline$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {new.sentence}
{ "~" * skip$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}

FUNCTION {new.block.checka}
{ empty$
    'skip$
    'new.block
  if$
}

FUNCTION {new.block.checkb}
{ empty$
  swap$ empty$
  and
    'skip$
    'new.block
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {commafield.or.null}
{ duplicate$ empty$
    { pop$ "" }
    {", " *}
  if$
}

FUNCTION {periodfield.or.null}
{ duplicate$ empty$
    { pop$ "" }
    {". " *}
  if$
}

FUNCTION {parenfield.or.null}
{ duplicate$ empty$
    { pop$ "" }
    {"(" swap$ * ")" *}
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "\textit{" swap$ * "}" * }
  if$
}

FUNCTION {emphasizec}
{ duplicate$ empty$
    { pop$ "" }
    { "\textit{" swap$ * "}," * }
  if$
}

FUNCTION {emphasizesc}
{ duplicate$ empty$
    { pop$ "" }
    { "\textit{" swap$ * "};" * }
  if$
}

FUNCTION {roman}
{ duplicate$ empty$
    { pop$ "" }
    { "\textrm{" swap$ * "}" * }
  if$
}

FUNCTION {romanp}
{ duplicate$ empty$
    { pop$ "" }
    { "\textrm{" swap$ * "}." * }
  if$
}

FUNCTION {romanc}
{ duplicate$ empty$
    { pop$ "" }
    { "\textrm{" swap$ * "}," * }
  if$
}

FUNCTION {romansc}
{ duplicate$ empty$
    { pop$ "" }
    { "\textrm{" swap$ * "};" * }
  if$
}

FUNCTION {slantc}
{ duplicate$ empty$
    { pop$ "" }
    { "\textsl{" swap$ * "}," * }
  if$
}

FUNCTION {slant}
{ duplicate$ empty$
    { pop$ "" }
    { "\textsl{" swap$ * "}" * }
  if$
}

FUNCTION {emboldenc}
{ duplicate$ empty$
    { pop$ "" }
    { "\textbf{" swap$ * "}," * }
  if$
}

FUNCTION {embolden}
{ duplicate$ empty$
    { pop$ "" }
    { "\textbf{" swap$ * "}" * }
  if$
}

INTEGERS { nameptr namesleft numnames }

INTEGERS { etal }

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames #15 >
  s numnames "{ll}" format.name$ "others" = numnames #1 > and
  or 'etal :=
  etal
    { #1 #1 + 'namesleft := }
    { numnames 'namesleft := }
  if$
    { namesleft #0 > }
    { s nameptr "{vv~}{ll}{~f{}}{~jj}" format.name$ 't :=
      nameptr #1 >
   { namesleft #1 >
       { ",\ " * t * }
       { nameptr #2 >
      { "" * }
      'skip$
         if$
         t "others" =
         etal or
      { "~et~al." * }
      { ",\ " * t * }
         if$
       }
     if$
   }
   't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {format.ednames}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames #15 >
  s numnames "{ll}" format.name$ "others" = numnames #1 > and
  or 'etal :=
  etal
    { #1 #1 + 'namesleft := }
    { numnames 'namesleft := }
  if$
    { namesleft #0 > }
    { s nameptr "{vv~}{ll}{~f{}}{~jj}" format.name$ 't :=
      nameptr #1 >
   { namesleft #1 >
       { ",\ " * t * }
       { nameptr #2 >
      { "" * }
      'skip$
         if$
         t "others" =
         etal or
      { "~et~al." * }
      { ",\ " * t * }
         if$
       }
     if$
   }
   't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {format.authors}
{ author empty$
    { "" }
    { author format.names add.period$ "\," * }
  if$
}

FUNCTION {format.editors_clean}
{ editor empty$
    { "" }
    { editor format.ednames ", editor." * output}
  if$
}

FUNCTION {format.editors}
{ editor empty$
    { "" }
    { editor format.ednames 
      editor num.names$ #1 >
      { ",\ editors." * }
      { ",\ editor." * }
      if$
    }
  if$
}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
   { t #1 #2 substring$ "--" = not
       { "--" *
         t #2 global.max$ substring$ 't :=
       }
       {   { t #1 #1 substring$ "-" = }
      { "-" *
        t #2 global.max$ substring$ 't :=
      }
         while$
       }
     if$
   }
   { t #1 #1 substring$ *
     t #2 global.max$ substring$ 't :=
   }
      if$
    }
  while$
}


FUNCTION {format.type}
{ type empty$
    { "Report No.: " }
    { type " No.: " * }
  if$
}

FUNCTION {format.address}
{ address empty$
    { "" }
    { address ": " * }
  if$
}

FUNCTION {format.publisher}
{ publisher empty$
    { "" }
    { publisher "; " * }
  if$
}

FUNCTION {format.referenceclass}
{ referenceclass empty$
    { "" }
    { "(" referenceclass * ")" * }
  if$
}

FUNCTION {format.titleclass}
{ titleclass empty$
    { "" }
    { "(" titleclass * ") " * }
  if$
}

FUNCTION {format.venue}
{ venue empty$
    { "" }
    { venue add.period$ }
  if$
}

FUNCTION {format.venuenodot}
{ venue empty$
    { "" }
    { venue }
  if$
}

FUNCTION {format.month}
{ month empty$
    { "" }
    'month
  if$
}

FUNCTION {format.space.month}
{ month empty$
    { "" }
    { " " * month }
  if$
}

FUNCTION {format.date}
{ year empty$
    { "" }
    'year
  if$
}

FUNCTION {format.conferencedate}
{ conferenceyear empty$
    { format.date }
    'conferenceyear
  if$
}

FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}

FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}

FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { series empty$
        { " (" }
        { " (" * series * "; " * }
      if$
      "vol." volume tie.or.space.connect
      "volume and number" number either.or.check
      ")" *
    }
  if$
}

FUNCTION {format.number.series}
{ volume empty$
    { number empty$
        { series empty$
            { "" }
            { "(" series * ")" * }
          if$
        }
        { series empty$
            { "there's a number but no series in " cite$ * warning$ }
            { "(" series * "; no. " * number * ")" * }
          if$
        }
      if$
    }
    { "" }
  if$
}

FUNCTION {format.edition}
{ edition empty$
    { "" }
    { edition " ed." * }
  if$
}

INTEGERS { multiresult }

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
   { #1 'multiresult := }
   { t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}

FUNCTION {format.pages}
{ pages empty$
    { "" }
    { pages multi.page.check
   { "p." pages n.dashify tie.or.space.connect }
   { "p." pages tie.or.space.connect }
      if$
    }
  if$
}

FUNCTION {format.pages.a}
{ pages empty$
    { "" }
    { "p" pages tie.or.space.connect }
  if$
}

FUNCTION {format.yr.vol.pages}
{ year romansc field.or.null 
  " " swap$ * *
  volume field.or.null number parenfield.or.null *
  "" swap$ * *
  pages empty$
    'skip$
    %{ commafield.or.null duplicate$ empty$   % dups the ','
    { duplicate$ empty$   
   { pop$ format.pages.a }
   { ":" * pages * }
      if$
    }
  if$
}

FUNCTION {format.yr.pages}
{ year field.or.null
  " " swap$ * *
  pages empty$
    'skip$
    { duplicate$ empty$
   { pop$ format.pages }
   { "; " * format.pages * }
      if$
    }
  if$
}

FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
   { "Chapter" }
   { type "l" change.case$ }
      if$
      chapter tie.or.space.connect
      pages empty$
   'skip$
   { ", " * format.pages * }
      if$
    }
  if$
}

FUNCTION {empty.misc.check}
{ author empty$ title empty$ howpublished empty$
  month empty$ year empty$ note empty$
  and and and and and
    { "all relevant fields are empty in " cite$ * warning$ }
    'skip$
  if$
}

FUNCTION {format.thesis}
{ type empty$
    {"[thesis]."}
    {"[" type * " thesis]." * }
  if$
}

FUNCTION {format.article.crossref}
{ key empty$
    { journal empty$
   { "need key or journal for " cite$ * " to crossref " * crossref *
     warning$
     ""
   }
   { "In " journal * }
      if$
    }
    { "In " key * }
  if$
  " \cite{" * crossref * "}" *
}

FUNCTION {format.crossref.editor}
{ editor #1 "{vv~}{ll}" format.name$
  editor num.names$ duplicate$
  #2 >
    { pop$ "~et~al." * }
    { #2 <
   'skip$
   { editor #2 "{vv~}{ll}{~f{}}{~jj}" format.name$ "others" =
       { "~et~al." * }
       { " and " * editor #2 "{vv~}{ll}" format.name$ * }
     if$
   }
      if$
    }
  if$
}

FUNCTION {format.book.crossref}
{ volume empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      "In "
    }
    { "Volume" volume tie.or.space.connect
      " of " *
    }
  if$
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
   { series empty$
       { "need editor, key, or series for " cite$ * " to crossref " *
         crossref * warning$
         "" *
       }
       { "\textit{" * series * "}" * }
     if$
   }
   { key * }
      if$
    }
    { format.crossref.editor * }
  if$
  " \cite{" * crossref * "}" *
}

FUNCTION {format.incoll.inproc.crossref}
{ editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
   { booktitle empty$
       { "need editor, key, or booktitle for " cite$ * " to crossref " *
         crossref * warning$
         ""
       }
       { "In \textit{" booktitle * ",}" * }
     if$
   }
   { "In " key * }
      if$
    }
    { "In " format.crossref.editor * }
  if$
  " \cite{" * crossref * "}" *
}

FUNCTION {in.book}
  { "In: " output
  booktitle roman
  edition empty$
    { skip$ }
    {";" * output
     edition " ed." * }
  if$
  volume empty$
    { skip$ }
    {"; Vol.~" * volume *}
  if$
  editor missing$
    {";" * output}
    {";" * output format.editors output }
  if$
  format.number.series output
  new.sentence
  format.address "address" output.check
  format.publisher "publisher" output.check
}

FUNCTION {proc.data}
  {format.editors output
  booktitle empty$ { title }{ booktitle } if$ roman
  edition empty$
    { skip$ }
    {";" * output
     edition " ed." * }
  if$
  volume empty$
    { skip$ }
    {"; Vol.~" * volume *}
  if$
  ";" * output
  venue empty$
    { " " output }
    { format.conferencedate output format.month ";" * output }
  if$
  format.venue output
  format.number.series output
  format.address output
  format.publisher output
  publisher empty$ { }{ format.date "." * output} if$
}

FUNCTION {conf.data}
  {"Paper presented at: " audience * "." * output
  format.conferencedate output format.month ";" * output
  format.venuenodot "venue" output.check
}

FUNCTION {book.vol.ed}
{ "" volume empty$
    {skip$}
    {" Vol.~" * volume *}
  if$
  edition empty$
    {";" * output}
    {"," * output
     edition " ed.;\" * output}
  if$
}

FUNCTION {thesis.affilation}
{ address empty$
  'skip$
  { "[" address *  "]:" * "address" output.check }
  if$
  institution empty$
    {school empty$
      'skip$
      {school ";" * output}
    if$ }
    {institution ";" * output}
  if$
}

FUNCTION {article}
{ output.bibitem
  format.titleclass output
  format.authors "~" * "author" output.check
  title missing$
  {}{
  " " title "t" change.case$ * ". " * "title" output.check
  }
  if$
  crossref missing$
    { 
      journal field.or.null roman add.period$
      note missing$
        {format.yr.vol.pages}
        {format.yr.vol.pages "; " *}
      if$
      output
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  note output
  add.period$
  format.referenceclass output
  fin.entry.noperiod
}

FUNCTION {submitted}
{ output.bibitem
  format.authors "author" output.check
  new.block
  "submitted for publication in" output
  note empty$
  { journal emphasize "journal" output.check }
  { journal emphasizec "journal" output.check
    note output }
  if$
  fin.entry
}

FUNCTION {inpress}
{ output.bibitem
  format.authors "author" output.check
  journal emphasizec "journal" output.check
  "in press" output
  note output
  fin.entry
}

FUNCTION {book}
{ output.bibitem
  format.titleclass output
  author empty$
    { format.editors "author and editor" output.check 
      editor empty$
      { }
      {"\, " * }
      if$
    }
    { format.authors output.nonnull "\, " *  }
  if$
  booktitle empty$
    { title "t" change.case$ add.period$ output }
    { booktitle "t" change.case$ add.period$ output }
  if$
  author empty$
    { }
    {
      editor empty$
        { }
        { format.editors output }
      if$
    }
  if$
  format.edition output
  crossref missing$
    { new.sentence
      address ":" * "address" output.check
      publisher "; " * "publisher" output.check
    }
    { format.book.crossref output.nonnull }
  if$
  format.date add.period$ "year" output.check
  format.bvolume output
  format.number.series output
  note output
  add.period$
  format.referenceclass output
  fin.entry.noperiod
}

FUNCTION {booklet}
{ output.bibitem
  format.authors "author" output.check
  " ``" title * "'',  " * "title" output.check
  howpublished empty$
    { skip$ }
    { howpublished "," * output}
  if$
  address empty$
    { skip$ }
    { address "," * output}
  if$
  format.date "year" output.check
  note output
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  format.titleclass output
  format.authors "author" output.check
  " " title "t" change.case$ * ".  " * "title" output.check
  "In: " output proc.data
  chapter empty$ {
      pages empty$ { "" }{ format.pages add.period$ } if$ output
    }
    {
      pages empty$ { "" }{ format.chapter.pages} if$ output
    }
  if$
  note output
  add.period$
  format.referenceclass output
  fin.entry.noperiod
}

FUNCTION {presentation}
{ output.bibitem
  format.titleclass output
  format.authors "author" output.check
  " " title "t" change.case$ * ".  " * "title" output.check
  conf.data
  note empty$
   { "" add.period$ }
   { note add.period$ }
  if$
  output
  add.period$
  format.referenceclass output
  fin.entry.noperiod
}

FUNCTION {proceedings}
{ output.bibitem
  format.titleclass output
  proc.data
  chapter empty$ {
      pages empty$ { "" }{ format.pages} if$ output
    }
    {
      pages empty$ { "" }{ format.chapter.pages} if$ output
    }
  if$
  note output
  add.period$
  format.referenceclass output
  fin.entry.noperiod
}

FUNCTION { inbook }{ inproceedings }

FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  " " title * ".  " * "title" output.check
  in.book
  format.yr.pages
  note empty$
    {skip$}
    {"," * note output}
  if$
  fin.entry
}

FUNCTION {conference} { presentation }

FUNCTION {manual}
{ output.bibitem
  format.titleclass output
  author empty$
    'skip$
    { format.authors output.nonnull }
  if$
  title "t" change.case$ roman ".~ " * "title" output.check
  address empty$
    'skip$
    { address ":" * output }
  if$
  institution empty$
    'skip$
    { institution ";" * output}
  if$
  format.edition output
  format.date 
  format.space.month * 
  note empty$
    { output }
    { ". " * note * output }
  if$
  add.period$
  format.referenceclass output
  fin.entry.noperiod
}

FUNCTION {mastersthesis}
{ output.bibitem
  format.titleclass output
  format.authors "author" output.check
  title "title" output.check
  "[master's thesis]." output
  thesis.affilation
  format.date "year" output.check
  note output
  add.period$
  format.referenceclass output 
  fin.entry.noperiod
}

FUNCTION {misc}
{ output.bibitem
  format.titleclass output
  format.authors output
  title empty$
    'skip$
    { " " title * ". " * output }
  if$
  howpublished empty$
    { institution empty$
        'skip$
        {institution ";" * output }
      if$
    }
    {howpublished ";" * output }
  if$
  format.date output
  note output
  add.period$
  format.referenceclass output
  fin.entry.noperiod
  empty.misc.check
}

FUNCTION {program}
{ output.bibitem
  format.authors output
  title empty$
    'skip$
    { " " title slantc * "~" * output }
  if$
  howpublished empty$
    'skip$
    {howpublished "," * output }
  if$
  format.date output
  note output
  fin.entry
  empty.misc.check
}

FUNCTION { thesis }
{ output.bibitem
  format.titleclass output
  format.authors "author" output.check
  title "title" output.check
  format.thesis output.nonnull
  thesis.affilation
  format.date "year" output.check
  note output
  add.period$
  format.referenceclass output
  fin.entry.noperiod
}

FUNCTION {phdthesis}
{ output.bibitem
  format.titleclass output
  format.authors "author" output.check
  title "title" output.check
  format.thesis output.nonnull
  thesis.affilation
  format.date "year" output.check
  note output
  add.period$
  format.referenceclass output
  fin.entry.noperiod
}

FUNCTION {techreport}
{ output.bibitem
  format.titleclass output
  format.authors "author" output.check
   title "t" change.case$ roman ". " * "title"  output.check
   format.address  output
   institution "; " * "institution" output.check
   format.date    format.space.month *  add.period$ "year" output.check
   format.type output
   number "." * output
  note output
  add.period$
  format.referenceclass output
  fin.entry.noperiod
}

FUNCTION {unpublished}
{ output.bibitem
  format.authors "author" output.check
  note empty$
    { title empty$
        { "Unpublished data" output }
        { year empty$
             {title output}
             {title ", " * year * output }
        }
        if$
    }
  {  address empty$
      { note * ".~" output }
      { note * ".~" * address * "\unskip,~" output }
      if$
      format.date "year" output.check
  }
  if$
  fin.entry
}


FUNCTION {privatecommunication}
{ output.bibitem
  format.authors "author" output.check
    address empty$
      { "Private communication," * output}
      { "~" * address * ".~~" * "Private communication," * output }
    if$
    month empty$
      {format.date "year" output.check
      note output}
      {format.month output
      format.date "year" output.check
      note output}
    if$
  fin.entry
}

FUNCTION {milstd}
{ output.bibitem
  format.titleclass output
   number ". " * "number" output.check
   title "t" change.case$ roman ". " * "title"  output.check
   format.address output
   Institution empty$ {} { institution "; " * output } if$
   format.month output
   format.date "." * "year" output.check
  note output
  add.period$
  format.referenceclass output
  fin.entry.noperiod
}

FUNCTION {internet}
{ output.bibitem
  format.titleclass output
  format.authors " " * "author" output.check
  title empty$
    {""}
    { title "t" change.case$ * ".\,\ " *}
  if$
  format.address output
  publisher empty$
    { 
      institution empty$
        { }
        { institution "; " * output}
      if$
    }
    { publisher "; " * output}
  if$
  year empty$
    { " [date unknown; " }
    { year 
      month empty$
        { " [\ignorespaces " * }
        { "~" * month * " [\ignorespaces " *}
      if$
    }
  if$
  revised empty$
    { "" * }
    { "revised~" * revised * "; " *}
    if$
  accessed empty$
    { "" * }
    { "accessed~"  * accessed * "]. " *}
    if$
  website * 
  doi empty$
    {}
    { add.period$ " " * doi * }
  if$
  output
  add.period$
  format.referenceclass output
  fin.entry.noperiod
}

FUNCTION {remark}
{ output.bibitem
  note "note" output.check
  fin.entry
}

FUNCTION {default.type} { misc }

MACRO {ARL} {"U.S. Army Research Laboratory: Aberdeen Proving Ground, MD"}

MACRO {jan} {"Jan"}
MACRO {feb} {"Feb"}
MACRO {mar} {"Mar"}
MACRO {apr} {"Apr"}
MACRO {may} {"May"}
MACRO {jun} {"June"}
MACRO {jul} {"July"}
MACRO {aug} {"Aug"}
MACRO {sep} {"Sep"}
MACRO {oct} {"Oct"}
MACRO {nov} {"Nov"}
MACRO {dec} {"Dec"}

MACRO {acchemr} {"Acc.\ Chem.\ Res."}
MACRO {aacsa} {"Adv.\ {ACS} Abstr."}
MACRO {anchem} {"Anal.\ Chem."}
MACRO {bioch} {"Biochemistry"}
MACRO {bicoc} {"Bioconj.\ Chem."}  % ***
MACRO {bitech} {"Biotechnol.\ Progr."}  % ***
MACRO {chemeng} {"Chem.\ Eng.\ News"}
MACRO {chs} {"Chem.\ Health Safety"} % ***
MACRO {crt} {"Chem.\ Res.\ Tox."} % ***
MACRO {chemrev} {"Chem.\ Rev."} % ***
MACRO {cmat} {"Chem.\ Mat."} % ***
MACRO {chemtech} {"{CHEMTECH}"} % ***
MACRO {enfu} {"Energy \&{} Fuels"} % ***
MACRO {envst} {"Environ.\ Sci.\ Technol."}
MACRO {iecf} {"Ind.\ Eng.\ Chem.\ Fundam."}
MACRO {iecpdd} {"Ind.\ Eng.\ Chem.\ Proc.\ Des.\ Dev."}
MACRO {iecprd} {"Ind.\ Eng.\ Chem.\ Prod.\ Res.\ Dev."}
MACRO {iecr} {"Ind.\ Eng.\ Chem.\ Res."} % ***
MACRO {inor} {"Inorg.\ Chem."}
MACRO {jafc} {"J.~Agric.\ Food Chem."}
MACRO {jacs} {"J.~Am.\ Chem.\ Soc."}
MACRO {jced} {"J.~Chem.\ Eng.\ Data"}
MACRO {jcics} {"J.~Chem.\ Inf.\ Comput.\ Sci."}
MACRO {jmc} {"J.~Med.\ Chem."}
MACRO {joc} {"J.~Org.\ Chem."}
MACRO {jps} {"J.~Pharm.\ Sci."}
MACRO {jpcrd} {"J.~Phys.\ Chem.\ Ref.\ Data"} % ***
MACRO {jpc} {"J.~Phys.\ Chem."}
MACRO {jpca} {"J.~Phys.\ Chem.~A"}
MACRO {jpcb} {"J.~Phys.\ Chem.~B"}
MACRO {lang} {"Langmuir"}
MACRO {macro} {"Macromolecules"}
MACRO {orgmet} {"Organometallics"}

MACRO {jft} {"J.~Chem.\ Soc., Faraday Trans."}
MACRO {jft1} {"J.~Chem.\ Soc., Faraday Trans. 1"}
MACRO {jft2} {"J.~Chem.\ Soc., Faraday Trans. 2"}
MACRO {tfs} {"Trans.\ Faraday Soc."}
MACRO {jcis} {"J.~Colloid Interface Sci."}
MACRO {acis} {"Adv.~Colloid Interface Sci."}
MACRO {cs} {"Colloids Surf."}
MACRO {csa} {"Colloids Surf.\ A:\ Physicochem.\ Eng.\ Apects"}
MACRO {csb} {"Colloids Surf.\ B:\ Biointerfaces"}
MACRO {pcps} {"Progr.\ Colloid Polym.\ Sci."}
MACRO {jmr} {"J.~Magn.\ Reson."}
MACRO {jmra} {"J.~Magn.\ Reson.\ A"}
MACRO {jmrb} {"J.~Magn.\ Reson.\ B"}
MACRO {sci} {"Science"}
MACRO {nat} {"Nature (London)"}
MACRO {jcch} {"J.~Comput.\ Chem."}
MACRO {cca} {"Croat.\ Chem.\ Acta"}

MACRO {poly} {"Polymer"}
MACRO {ajp} {"Am.\ J.\ Phys."}
MACRO {rsi} {"Rev.\ Sci.\ Instrum."}
MACRO {jcp} {"J.~Chem.\ Phys."}
MACRO {cpl} {"Chem.\ Phys.\ Lett."}
MACRO {molph} {"Mol.\ Phys."}
MACRO {pac} {"Pure Appl.\ Chem."}
MACRO {jbc} {"J.~Biol.\ Chem."}
MACRO {tl} {"Tetrahedron Lett.\"}
MACRO {psisoe} {"Proc.\ SPIE-Int.\ Soc.\ Opt.\ Eng."}
MACRO {prb} {"Phys.\ Rev.\ B:\ Condens.\ Matter"}
MACRO {jap} {"J.~Appl.\ Phys."}
MACRO {pnac} {"Proc.\ Natl.\ Acad.\ Sci.\ U.~S.~A. "}
MACRO {bba} {"Biochim.\ Biophys.\ Acta"}
MACRO {nar} {"Nucleic.\ Acid Res."}

READ

STRINGS { longest.label }

INTEGERS { number.label longest.label.width }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #1 'number.label :=
  #0 'longest.label.width :=
}

FUNCTION {longest.label.pass}
{ number.label int.to.str$ 'label :=
  number.label #1 + 'number.label :=
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
}

EXECUTE {initialize.longest.label}

ITERATE {longest.label.pass}

FUNCTION {begin.bib}
{ preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\providecommand{\refin}[1]{\\ \textbf{Referenced in:} #1}"
  write$ newline$
  "\begin{thebibliography}{"  longest.label  * "}" * write$ newline$
}

EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}
%% 
%%
%% End of file `achemso.bst'.
