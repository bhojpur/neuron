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

type APIUnary struct {
	UnaryOp
}

func (fn *APIUnary) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template
	switch {
	case fn.UnaryOp.Name() == "Clamp":
		paramNames = []string{"a", "min", "max", "opts"}
		paramTemplates = []*template.Template{tensorType, interfaceType, interfaceType, splatFuncOptType}
	default:
		paramNames = []string{"a", "opts"}
		paramTemplates = []*template.Template{tensorType, splatFuncOptType}
	}
	return &Signature{
		Name:            fn.Name(),
		NameTemplate:    plainName,
		ParamNames:      paramNames,
		ParamTemplates:  paramTemplates,
		RetVals:         []string{"retVal"},
		RetValTemplates: []*template.Template{tensorType},
		Err:             true,
	}
}

func (fn *APIUnary) WriteBody(w io.Writer) {
	body := `e := a.Engine()
	if {{interfaceName .Name | lower}}, ok := e.({{interfaceName .Name}}); ok {
		{{if eq .Name "Clamp" -}}
		return clamper.Clamp(a, min, max, opts...)
		{{else -}}
		return {{interfaceName .Name|lower}}.{{.Name}}(a, opts...)
		{{end -}}
	}
	err = errors.Errorf("Engine does not perform {{.Name}}")
	return
	`

	T := template.Must(template.New("body").Funcs(funcs).Parse(body))
	T.Execute(w, fn)
}

func (fn *APIUnary) Write(w io.Writer) {
	w.Write([]byte("func "))
	sig := fn.Signature()
	sig.Write(w)
	w.Write([]byte("{ \n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func GenerateUncondUnaryAPI(f io.Writer, kinds Kinds) {
	var unaries []*APIUnary
	for _, u := range unconditionalUnaries {
		fn := &APIUnary{
			UnaryOp: u,
		}
		unaries = append(unaries, fn)
	}
	for _, u := range unaries {
		u.Write(f)
	}
}

func GenerateCondUnaryAPI(f io.Writer, kinds Kinds) {
	var unaries []*APIUnary
	for _, u := range conditionalUnaries {
		fn := &APIUnary{
			UnaryOp: u,
		}
		unaries = append(unaries, fn)
	}
	for _, u := range unaries {
		u.Write(f)
	}
}

func GenerateSpecialUnaryAPI(f io.Writer, kinds Kinds) {
	var unaries []*APIUnary

	for _, u := range specialUnaries {
		fn := &APIUnary{
			UnaryOp: u,
		}
		unaries = append(unaries, fn)
	}
	for _, u := range unaries {
		u.Write(f)
	}
}
