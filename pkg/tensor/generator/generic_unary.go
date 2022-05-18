package generator

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
	"io"
	"text/template"
)

type GenericUnary struct {
	TypedUnaryOp
	Iter bool
	Cond bool
}

func (fn *GenericUnary) Name() string {
	if fn.Iter {
		return fn.TypedUnaryOp.Name() + "Iter"
	}
	return fn.TypedUnaryOp.Name()
}

func (fn *GenericUnary) Signature() *Signature {
	paramNames := []string{"a"}
	paramTemplates := []*template.Template{sliceType}
	var err bool
	if fn.Iter {
		paramNames = append(paramNames, "ait")
		paramTemplates = append(paramTemplates, iteratorType)
		err = true
	}
	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   typeAnnotatedName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,

		Kind: fn.Kind(),
		Err:  err,
	}
}

func (fn *GenericUnary) WriteBody(w io.Writer) {
	var IterName0 string
	T := template.New(fn.Name()).Funcs(funcs)

	if fn.Iter {
		T = template.Must(T.Parse(genericUnaryIterLoopRaw))
		IterName0 = "ait"
	} else {
		T = template.Must(T.Parse(genericLoopRaw))
	}
	if fn.Cond {
		template.Must(T.New("loopbody").Parse(fn.SymbolTemplate()))
	} else {
		template.Must(T.New("loopbody").Parse(basicSet))
		template.Must(T.New("symbol").Parse(fn.SymbolTemplate()))
	}
	template.Must(T.New("opDo").Parse(unaryOpDo))
	template.Must(T.New("callFunc").Parse(unaryOpCallFunc))
	template.Must(T.New("check").Parse(""))

	lb := LoopBody{
		TypedOp:   fn.TypedUnaryOp,
		Range:     "a",
		Left:      "a",
		Index0:    "i",
		IterName0: IterName0,
	}
	T.Execute(w, lb)
}

func (fn *GenericUnary) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func "))
	sig.Write(w)
	w.Write([]byte("{\n"))
	fn.WriteBody(w)
	if sig.Err {
		w.Write([]byte("\nreturn\n"))
	}
	w.Write([]byte("}\n\n"))
}

func GenerateGenericUncondUnary(f io.Writer, ak Kinds) {
	var gen []*GenericUnary
	importStmt := `
import (
	"math"
	"math/cmplx"

	"github.com/bhojpur/neuron/pkg/math32"
	"github.com/bhojpur/neuron/pkg/vector/vecf32"
	"github.com/bhojpur/neuron/pkg/vector/vecf64"
)
`
	f.Write([]byte(importStmt))
	for _, tu := range typedUncondUnaries {
		if tc := tu.TypeClass(); tc != nil && !tc(tu.Kind()) {
			continue
		}
		fn := &GenericUnary{
			TypedUnaryOp: tu,
		}
		gen = append(gen, fn)
	}

	for _, g := range gen {
		g.Write(f)
		g.Iter = true
	}
	for _, g := range gen {
		g.Write(f)
	}
}

func GenerateGenericCondUnary(f io.Writer, ak Kinds) {
	var gen []*GenericUnary
	for _, tu := range typedCondUnaries {
		if tc := tu.TypeClass(); tc != nil && !tc(tu.Kind()) {
			continue
		}
		// special case for cmplx
		if isComplex(tu.Kind()) {
			continue
		}

		fn := &GenericUnary{
			TypedUnaryOp: tu,
			Cond:         true,
		}
		gen = append(gen, fn)
	}
	for _, g := range gen {
		g.Write(f)
		g.Iter = true
	}
	for _, g := range gen {
		g.Write(f)
	}
}

/*
SPECIAL CASES
*/

type GenericUnarySpecial struct {
	*GenericUnary
	AdditionalParams         []string
	AdditionalParamTemplates []*template.Template
}

func (fn *GenericUnarySpecial) Signature() *Signature {
	sig := fn.GenericUnary.Signature()
	sig.ParamNames = append(sig.ParamNames, fn.AdditionalParams...)
	sig.ParamTemplates = append(sig.ParamTemplates, fn.AdditionalParamTemplates...)
	return sig
}

func (fn *GenericUnarySpecial) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func "))
	sig.Write(w)
	w.Write([]byte("{\n"))
	fn.WriteBody(w)
	if sig.Err {
		w.Write([]byte("\nreturn\n"))
	}
	w.Write([]byte("}\n\n"))
}

func (fn *GenericUnarySpecial) WriteBody(w io.Writer) {
	var IterName0 string
	T := template.New(fn.Name()).Funcs(funcs)

	if fn.Iter {
		T = template.Must(T.Parse(genericUnaryIterLoopRaw))
		IterName0 = "ait"
	} else {
		T = template.Must(T.Parse(genericLoopRaw))
	}
	template.Must(T.New("loopbody").Parse(clampBody))
	template.Must(T.New("opDo").Parse(unaryOpDo))
	template.Must(T.New("callFunc").Parse(unaryOpCallFunc))
	template.Must(T.New("check").Parse(""))

	lb := LoopBody{
		TypedOp:   fn.TypedUnaryOp,
		Range:     "a",
		Left:      "a",
		Index0:    "i",
		IterName0: IterName0,
	}
	T.Execute(w, lb)
}

func GenerateSpecialGenericUnaries(f io.Writer, ak Kinds) {
	var gen []*GenericUnarySpecial
	for _, tu := range typedSpecialUnaries {
		if tc := tu.TypeClass(); tc != nil && !tc(tu.Kind()) {
			continue
		}

		additional := tu.UnaryOp.(specialUnaryOp).additionalParams
		tmpls := make([]*template.Template, len(additional))
		for i := range tmpls {
			tmpls[i] = scalarType
		}
		fn := &GenericUnarySpecial{
			GenericUnary: &GenericUnary{
				TypedUnaryOp: tu,
			},
			AdditionalParams:         additional,
			AdditionalParamTemplates: tmpls,
		}
		gen = append(gen, fn)
	}

	for _, fn := range gen {
		fn.Write(f)
		fn.Iter = true
	}

	for _, fn := range gen {
		fn.Write(f)
	}
}
