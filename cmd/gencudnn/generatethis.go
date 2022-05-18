package main

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
	"strings"

	"github.com/bhojpur/neuron/pkg/bindgen"
	"github.com/kr/pretty"
	"modernc.org/cc"
)

// generate this contains function to generate for THIS package (main)

// generateMappings is used to generate the mappings
func generateMappings(appendCurrent bool, fns ...func(buf io.WriteCloser, t *cc.TranslationUnit)) {
	hdr := "package main\n"

	initfn := `
	func init() {


	`
	t, err := bindgen.Parse(model, hdrfile)
	handleErr(err)

	var buf io.WriteCloser
	if appendCurrent {
		buf, err = os.OpenFile("mappings.go", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	} else {
		buf, err = os.OpenFile("mappings.go", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	}
	handleErr(err)
	defer buf.Close()

	if !appendCurrent {
		fmt.Fprintln(buf, hdr)
		bindgen.GenIgnored(buf, t, functions)
		fmt.Fprintln(buf, initfn)
	}

	for _, fn := range fns {
		fn(buf, t)
	}
	fmt.Fprintln(buf, "}\n")
}

// generateCRUD creates lists of CRUD functions for the generateStubs function to consume
func generateCRUD(buf io.Writer, t *cc.TranslationUnit, fnType string) {
	decls, err := bindgen.Get(t, functions)
	handleErr(err)

	var a = make(map[string][]string)
	switch fnType {
	case "create":
		fmt.Fprintf(buf, "creations = ")
	case "get":
		fmt.Fprintf(buf, "getFns = ")
	case "set":
		fmt.Fprintf(buf, "setFns = ")
	case "destroy":
		fmt.Fprintf(buf, "destructions = ")
	case "methods":
		fmt.Fprintf(buf, "methods = ")
	}

	var b map[string]struct{}

	for _, d := range decls {
		cs := d.(*bindgen.CSignature)
		params := cs.Parameters()

		if fnType != "methods" && !strings.Contains(strings.ToLower(cs.Name), fnType) {
			continue
		}
		if fnType == "methods" && len(params) == 0 {
			continue
		}

		switch fnType {
		case "create":
			for i := len(params) - 1; i >= 0; i-- {
				p := params[i]
				if !bindgen.IsConstType(p.Type()) && bindgen.IsPointer(p.Type()) {
					typ := nameOfType(p.Type())
					a[typ] = append(a[typ], cs.Name)
				}
			}

		case "set", "destroy":
			p := params[0]
			typ := nameOfType(p.Type())
			if typ == "cudnnHandle_t" && len(params) > 1 {
				p = params[1]
				typ = nameOfType(p.Type())
			}
			a[typ] = append(a[typ], cs.Name)
		case "methods":
			// if strings.Contains(strings.ToLower(cs.Name), "get") {
			// 	continue
			// }
			if _, ok := ignored[cs.Name]; ok {
				continue
			}
			if alreadyGenIn(cs.Name, creations, setFns, destructions) {
				continue
			}

			p := params[0]
			typ := nameOfType(p.Type())
			a[typ] = append(a[typ], cs.Name)
		}
	}
	fmt.Fprintf(buf, "%# v\n\n", pretty.Formatter(a))

	// set the actual thing if not set
	// lisp users just shake their head in disappointment
	switch fnType {
	case "create":
		creations = a
	case "set":
		setFns = a
	case "destroy":
		destructions = a
	case "methods":
		methods = a
		if b != nil {
			fmt.Fprintf(buf, "orphaned = %# v\n\n", pretty.Formatter(b))
			orphaned = b
		}
	}
}

func generateAlphaBeta(buf io.Writer, t *cc.TranslationUnit) {
	decls, err := bindgen.Get(t, functions)
	handleErr(err)
	fmt.Fprint(buf, "alphaBetas = map[string]map[int]string {\n")
	for _, d := range decls {
		cs := d.(*bindgen.CSignature)
		params := cs.Parameters()

		var printedName bool
		for i := len(params) - 1; i >= 0; i-- {
			p := params[i]
			if !(bindgen.IsConstType(p.Type()) && bindgen.IsPointer(p.Type())) {
				continue
			}
			if inList(p.Name(), alphaBetaParams) {
				if !printedName {
					printedName = true
					fmt.Fprintf(buf, "%q: {", cs.Name)
				}
				fmt.Fprintf(buf, "%d: %q, ", i, p.Name())
			}
		}
		if printedName {
			fmt.Fprint(buf, "},\n") // close the first
		}
	}
	fmt.Fprint(buf, "}\n")
}
