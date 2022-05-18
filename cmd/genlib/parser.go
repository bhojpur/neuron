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
	"strings"

	"github.com/bhojpur/neuron/pkg/bindgen"
	"modernc.org/cc"
)

func Parse() (retVal []*CSignature) {
	t, err := bindgen.Parse(bindgen.Model(), "cuda.h")
	if err != nil {
		panic(err)
	}

	decls, err := functions(t)
	if err != nil {
		panic(err)
	}

	for _, d := range decls {
		retVal = append(retVal, decl2csig(d.(*bindgen.CSignature)))
	}
	return
}

func functions(t *cc.TranslationUnit) ([]bindgen.Declaration, error) {
	filter := func(decl *cc.Declarator) bool {
		name := bindgen.NameOf(decl)
		if !strings.HasPrefix(name, "cu") {
			return false
		}
		if _, ok := ignoredFunctions[name]; ok {
			return false
		}
		if decl.Type.Kind() == cc.Function {
			return true
		}
		return false
	}
	return bindgen.Get(t, filter)
}

func decl2csig(d *bindgen.CSignature) *CSignature {
	retVal := new(CSignature)
	retVal.Name = d.Name
	var params []*Param
	for _, p := range d.Parameters() {
		params = append(params, bgparam2param(p))
	}
	retVal.Params = params
	retVal.Fix()
	return retVal
}

// bgparam2cparam transforms bindgen parameter to *Param
func bgparam2param(p bindgen.Parameter) *Param {
	name := p.Name()
	typ := cleanType(p.Type())
	isPtr := bindgen.IsPointer(p.Type())
	return NewParam(name, typ, isPtr)
}

func cleanType(t cc.Type) string {
	typ := t.String()
	if td := bindgen.TypeDefOf(t); td != "" {
		typ = td
	}

	if bindgen.IsConstType(t) {
		typ = strings.TrimPrefix(typ, "const ")
	}

	if bindgen.IsPointer(t) {
		typ = strings.TrimSuffix(typ, "*")
	}
	return typ
}
