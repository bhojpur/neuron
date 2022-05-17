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
	"reflect"
	"text/template"
)

const (
	fnErrSet = `if {{.Range}}[i], err = {{template "callFunc" .}}; handleNoOp(err) != nil {
		return
	}`
	fnErrIncr = `var x {{asType .Kind}}
	if x, err = {{template "callFunc" .}}; err != nil {
		if err = handleNoOp(err);err != nil {
			return
		}
	}
	{{.Range}}[i] = x
	`
	simpleUnaryCallFunc = `{{template "symbol" .}}({{.Left}}[{{.Index0}}])`
)

type Map struct {
	k    reflect.Kind
	Iter bool
	Incr bool
	Err  bool
}

func (fn *Map) Name() string {
	switch {
	case fn.Iter && fn.Incr && fn.Err:
		return "MapIterIncrErr"
	case fn.Iter && fn.Incr && !fn.Err:
		return "MapIterIncr"
	case fn.Iter && !fn.Incr && fn.Err:
		return "MapIterErr"
	case fn.Iter && !fn.Incr && !fn.Err:
		return "MapIter"
	case !fn.Iter && fn.Incr && fn.Err:
		return "MapIncrErr"
	case !fn.Iter && fn.Incr && !fn.Err:
		return "MapIncr"
	case !fn.Iter && !fn.Incr && fn.Err:
		return "MapErr"
	default:
		return "Map"
	}
}

func (fn *Map) Arity() int             { return 1 }
func (fn *Map) SymbolTemplate() string { return "fn" }
func (fn *Map) TypeClass() TypeClass   { return nil }
func (fn *Map) IsFunc() bool           { return true }
func (fn *Map) Kind() reflect.Kind     { return fn.k }

func (fn *Map) Signature() *Signature {
	var retErr bool
	paramNames := []string{"fn", "a"}
	paramTemplates := []*template.Template{unaryFuncType, sliceType}
	if fn.Iter {
		paramNames = append(paramNames, "ait")
		paramTemplates = append(paramTemplates, iteratorType)
		retErr = true
	}
	if fn.Err {
		paramTemplates[0] = unaryFuncErrType
		retErr = true
	}

	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   typeAnnotatedName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,
		Kind:           fn.Kind(),
		Err:            retErr,
	}
}

func (fn *Map) WriteBody(w io.Writer) {
	Range := "a"
	Left := "a"

	var T *template.Template
	var IterName0 string
	if fn.Iter {
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(genericUnaryIterLoopRaw))
		IterName0 = "ait"
	} else {
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(genericLoopRaw))
	}

	switch {
	case fn.Incr && fn.Err:
		template.Must(T.New("loopbody").Funcs(funcs).Parse(fnErrIncr))
	case fn.Incr && !fn.Err:
		template.Must(T.New("loopbody").Funcs(funcs).Parse(basicIncr))
	case !fn.Incr && fn.Err:
		template.Must(T.New("loopbody").Funcs(funcs).Parse(fnErrSet))
	default:
		template.Must(T.New("loopbody").Funcs(funcs).Parse(basicSet))
	}
	template.Must(T.New("callFunc").Funcs(funcs).Parse(simpleUnaryCallFunc))
	template.Must(T.New("symbol").Funcs(funcs).Parse("fn"))
	template.Must(T.New("opDo").Funcs(funcs).Parse(""))
	template.Must(T.New("check").Funcs(funcs).Parse(""))

	lb := LoopBody{
		TypedOp:   fn,
		Range:     Range,
		Left:      Left,
		Index0:    "i",
		IterName0: IterName0,
	}
	T.Execute(w, lb)
}

func (fn *Map) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func "))
	sig.Write(w)
	w.Write([]byte("{\n"))
	fn.WriteBody(w)
	w.Write([]byte("\nreturn \n"))
	w.Write([]byte("}\n\n"))
}

func makeGenericMaps(incr bool) (retVal []*Map) {
	for _, k := range AllKinds {
		if incr {
			if !isAddable(k) {
				continue
			}
		}
		if isParameterized(k) {
			continue
		}

		m := &Map{k: k}
		if incr {
			m.Incr = true
		}
		retVal = append(retVal, m)
	}
	return
}

func GenerateGenericMap(f io.Writer, ak Kinds) {
	gen0 := makeGenericMaps(false)
	for _, m := range gen0 {
		m.Write(f)
		m.Err = true
	}
	for _, m := range gen0 {
		m.Write(f)
		m.Err = false
		m.Iter = true
	}
	for _, m := range gen0 {
		m.Write(f)
		m.Err = true
	}
	for _, m := range gen0 {
		m.Write(f)
	}

	gen1 := makeGenericMaps(true)
	for _, m := range gen1 {
		m.Write(f)
		m.Err = true
	}
	for _, m := range gen1 {
		m.Write(f)
		m.Err = false
		m.Iter = true
	}
	for _, m := range gen1 {
		m.Write(f)
		m.Err = true
	}
	for _, m := range gen1 {
		m.Write(f)
	}
}
