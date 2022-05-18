package bindgen

// Copyright (c) 2018 Bhojpur Consulting Private Limited, India. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import (
	"fmt"
	"io"
	"os"
	"sort"

	"modernc.org/cc"
	"modernc.org/xc"
)

// FilterFunc is a function to filter types
type FilterFunc func(*cc.Declarator) bool

// Parse parses with the given model, as well as having some hard coded predefined definitions
// that are useful for translating C to Go code
func Parse(model *cc.Model, paths ...string) (*cc.TranslationUnit, error) {
	predefined, includePaths, sysIncludePaths, err := cc.HostConfig("-I/usr/local/cuda/include")
	if err != nil {
		return nil, fmt.Errorf("binding: failed to get host config: %v", err)
	}

	return cc.Parse(
		predefined+`
#define __const const
#define __attribute__(...)
#define __extension__
#define __inline
#define __restrict
unsigned __builtin_bswap32 (unsigned x);
unsigned long long __builtin_bswap64 (unsigned long long x);
`,
		paths,
		model,
		cc.IncludePaths(includePaths),
		cc.SysIncludePaths(sysIncludePaths),
	)
}

// Get returns a list of declarations given the filter function
func Get(t *cc.TranslationUnit, filter FilterFunc) ([]Declaration, error) {
	var decls []Declaration
	for ; t != nil; t = t.TranslationUnit {
		if t.ExternalDeclaration.Case != 1 { /* Declaration */
			continue
		}

		d := t.ExternalDeclaration.Declaration
		if d.Case != 0 {
			// Other case is 1: StaticAssertDeclaration.
			continue
		}

		init := d.InitDeclaratorListOpt
		if init == nil {
			continue
		}
		idl := init.InitDeclaratorList
		if idl.InitDeclaratorList != nil {
			// We do not want comma-separated lists.
			continue
		}
		id := idl.InitDeclarator
		if id.Case != 0 {
			// We do not want assignments.
			continue
		}

		declarator := id.Declarator
		name := NameOf(declarator)

		if !filter(declarator) {
			continue
		}
		params, variadic := declarator.Type.Parameters()

		var retType cc.Type
		var decl Declaration
		switch declarator.Type.Kind() {
		case cc.Function:
			retType = declarator.Type.Result()
			decl = &CSignature{
				Pos:         declarator.Pos(),
				Name:        name,
				Return:      retType,
				CParameters: params,
				Variadic:    variadic,
				Declarator:  declarator,
			}
		case cc.Enum:
			decl = &Enum{
				Pos:        declarator.Pos(),
				Name:       name,
				Type:       declarator.Type,
				Declarator: declarator,
			}
		default:
			decl = &Other{
				Pos:        declarator.Pos(),
				Name:       name,
				Declarator: declarator,
			}
		}
		decls = append(decls, decl)
	}

	sort.Sort(byName(decls))

	return decls, nil
}

// NameOf returns the name of a C declarator
func NameOf(any interface{}) (name string) {
	switch a := any.(type) {
	case Namer:
		return a.Name()
	case *cc.Declarator:
		var id int
		id, _ = a.Identifier()
		return string(xc.Dict.S(id))
	case *CSignature:
		return a.Name
	case *Enum:
		return a.Name
	case *Other:
		return a.Name
	default:
		return ""
	}
}

// TypeDefOf returns the type def name of a type. If a type is not a typedef'd type, it returns "".
func TypeDefOf(t cc.Type) (name string) {
	id := t.Declarator().RawSpecifier().TypedefName()
	return string(xc.Dict.S(id))
}

// Explore is a function used to iterate quickly on a project to translate C functions/types to Go functions/types
func Explore(t *cc.TranslationUnit, filters ...FilterFunc) error {
	pre := func(w io.Writer, a string) {}
	format := func(w io.Writer, a string) { fmt.Fprintf(w, "%v\n", a) }
	post := func(w io.Writer, a string) { fmt.Fprintln(w) }

	return exploration(os.Stdout, t, pre, format, post, filters...)
}

// GenIgnored generates Go code for a const data structure that contains all the ignored functions/types
//
// Filename indicates what file needs to be parsed, not the output file.
func GenIgnored(buf io.Writer, t *cc.TranslationUnit, filters ...FilterFunc) error {
	pre := func(w io.Writer, a string) { fmt.Fprint(w, "var ignored = map[string]struct{}{\n") }
	format := func(w io.Writer, a string) { fmt.Fprintf(w, "%q:{},\n", a) }
	post := func(w io.Writer, a string) { fmt.Fprint(w, "}\n") }

	return exploration(buf, t, pre, format, post, filters...)
}

// GenNameMap generates Go code representing a name mapping scheme
//
// filename indicates the file to be parsed, varname indicates the name of the variable.
// 	- fn is the transformation function.
// 	- init indicates if the mapping should be generated in a func init(){}
func GenNameMap(buf io.Writer, t *cc.TranslationUnit, varname string, fn func(string) string, filter FilterFunc, init bool) error {
	varstr := "var "
	if init {
		varstr = ""
	}
	pre := func(w io.Writer, a string) { fmt.Fprintf(w, "%v%v = map[string]string{\n", varstr, varname) }
	format := func(w io.Writer, a string) {
		fmt.Fprintf(w, "%q: %q,\n", a, fn(a))
	}
	post := func(w io.Writer, a string) { fmt.Fprint(w, "}\n") }
	return exploration(buf, t, pre, format, post, filter)
}

func exploration(w io.Writer, t *cc.TranslationUnit, pre, format, post func(io.Writer, string), filters ...FilterFunc) error {
	for _, f := range filters {
		decls, err := Get(t, f)
		if err != nil {
			return err
		}

		pre(w, "")
		for _, d := range decls {
			format(w, NameOf(d))
		}
		post(w, "")
	}
	return nil
}
